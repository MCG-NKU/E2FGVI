import os
import glob
import logging
import importlib
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from core.lr_scheduler import MultiStepRestartLR, CosineAnnealingRestartLR
from core.loss import AdversarialLoss
from core.dataset import TrainDataset, TrainDataset_Mem
from model.modules.flow_comp import FlowCompletionLoss


class Trainer:
    def __init__(self, config):
        self.config = config
        self.epoch = 0
        self.iteration = 0
        self.num_local_frames = config['train_data_loader']['num_local_frames']
        self.num_ref_frames = config['train_data_loader']['num_ref_frames']
        self.spynet_lr = config['trainer'].get('spynet_lr', 1.0)

        # setup data set and data loader
        if config['train_data_loader']['sequence_load'] == 0:
            # 默认的训练loader，非序列化输入训练
            self.train_dataset = TrainDataset(config['train_data_loader'])
            self.same_mask = False
        else:
            # 记忆训练的序列化训练loader
            # 是否在切换到下一个视频前使用相同的mask
            if config['train_data_loader']['sequence_load'] == 1:
                # 使用不同的重新生成的mask
                self.same_mask = False
            else:
                # 同一个视频使用相同的mask
                self.same_mask = True
            self.train_dataset = TrainDataset_Mem(config['train_data_loader'],
                                                  batch_size=config['trainer']['batch_size'],
                                                  same_mask=self.same_mask)

        self.train_sampler = None
        self.train_args = config['trainer']
        if config['distributed']:
            self.train_sampler = DistributedSampler(
                self.train_dataset,
                num_replicas=config['world_size'],
                rank=config['global_rank'])

        if config['train_data_loader']['sequence_load'] == 0:
            # 默认的训练loader，非序列化shuffle输入训练
            self.train_loader = DataLoader(
                self.train_dataset,
                batch_size=self.train_args['batch_size'] // config['world_size'],
                shuffle=(self.train_sampler is None),
                num_workers=self.train_args['num_workers'],
                sampler=self.train_sampler)
        else:
            # 记忆训练的序列化训练loader
            self.train_loader = DataLoader(
                self.train_dataset,
                batch_size=self.train_args['batch_size'] // config['world_size'],
                num_workers=self.train_args['num_workers'],
                sampler=self.train_sampler)

        # set loss functions
        self.adversarial_loss = AdversarialLoss(
            type=self.config['losses']['GAN_LOSS'])
        self.adversarial_loss = self.adversarial_loss.to(self.config['device'])
        self.l1_loss = nn.L1Loss()

        if config['model']['net'] == 'lite-MFN' or config['model']['net'] == 'large-MFN':
            self.flow_comp_loss = FlowCompletionLoss(estimator='mfn').to(self.config['device'])
        else:
            self.flow_comp_loss = FlowCompletionLoss(estimator='spy').to(self.config['device'])     # default

        # setup models including generator and discriminator
        net = importlib.import_module('model.' + config['model']['net'])

        if config['model']['memory'] != 0:
            self.memory = True
        else:
            self.memory = False

        if config['model']['net'] == 'lite-MFN' or config['model']['net'] == 'large-MFN':

            if config['model']['skip_dcn'] != 0:
                self.skip_dcn = True
            else:
                self.skip_dcn = False

            if config['model']['flow_guide'] != 0:
                self.flow_guide = True
            else:
                self.flow_guide = False

            if config['model']['token_fusion'] != 0:
                self.token_fusion = True
            else:
                self.token_fusion = False

            if config['model']['token_fusion_simple'] != 0:
                self.token_fusion_simple = True
            else:
                self.token_fusion_simple = False

            if config['model']['fusion_skip_connect'] != 0:
                self.fusion_skip_connect = True
            else:
                self.fusion_skip_connect = False

            if self.memory:
                # 额外输入记忆力需要的参数
                # 是否使用空间池化压缩记忆缓存
                if config['model']['mem_pool'] != 0:
                    self.mem_pool = True
                else:
                    self.mem_pool = False

                # 是否仅存储局部帧的记忆kv
                if config['model']['store_lf'] != 0:
                    self.store_lf = True
                else:
                    self.store_lf = False

                # 是否在增强前对齐缓存和当前帧的kv
                if config['model']['align_cache'] != 0:
                    self.align_cache = True
                else:
                    self.align_cache = False

                # 是否在对齐时对token通道分组进行，来实现sub-token的对齐
                if config['model']['sub_token_align'] != 0:
                    self.sub_token_align = True
                    self.sub_factor = config['model']['sub_token_align']
                else:
                    self.sub_token_align = False
                    self.sub_factor = 1

                # 是否只为一半的层装备记忆力来节省显存消耗
                if config['model']['half_memory'] != 0:
                    self.half_memory = True
                else:
                    self.half_memory = False

                # 是否只有最后一层blk装备记忆力来节省显存消耗，避免记忆干扰当前帧的特征提取
                if config['model']['last_memory'] != 0:
                    self.last_memory = True
                else:
                    self.last_memory = False

                # 是否使用cross attention融合记忆与当前特征(在Nh Nw维度流动信息)
                if config['model']['cross_att'] != 0:
                    self.cross_att = True
                else:
                    self.cross_att = False

                # 是否对时序上的信息也使用cross attention融合(额外在T维度流动信息)
                if config['model']['time_att'] != 0:
                    self.time_att = True
                else:
                    self.time_att = False

                # 是否在时序融合信息的时候解耦时空，降低计算复杂度
                if config['model']['time_deco'] != 0:
                    self.time_deco = True
                else:
                    self.time_deco = False

                # 是否在聚合时空记忆时使用temporal focal attention
                if config['model']['temp_focal'] != 0:
                    self.temp_focal = True
                else:
                    self.temp_focal = False

                # 是否在聚合时空记忆时使用cswin attention
                if config['model']['cs_win'] != 0:
                    self.cs_win = True
                    # if config['model']['cs_win'] == 2:
                    #     # cs_win_strip决定了cswin的条带宽度，默认为1
                    #     self.cs_win_strip = 2
                    # else:
                    #     self.cs_win_strip = 1
                else:
                    self.cs_win = False
                    # self.cs_win_strip = 1

                # 是否使用attention聚合不同时间的记忆和当前特征，而不是使用线性层聚合记忆再attention
                if config['model']['mem_att'] != 0:
                    self.mem_att = True
                else:
                    self.mem_att = False

                # 是否为cswin引入类似temporal focal的机制来增强注意力
                if config['model']['cs_focal'] != 0:
                    self.cs_focal = True
                    if config['model']['cs_focal'] == 2:
                        # 改进的正交全局滑窗策略，取到non-local的focal窗口
                        # 现在默认都是v2了，v1已经被淘汰
                        self.cs_focal_v2 = True
                    else:
                        raise Exception('Focal v1 has been given up.')
                else:
                    self.cs_focal = False
                    self.cs_focal_v2 = False

            # 是否使用3D deco focav2 cswin替换temporal focal trans主干
            if config['model']['cs_trans'] != 0:
                self.cs_trans = True
            else:
                self.cs_trans = False

            if self.cs_trans:
                # cs trans 主干需要的参数
                # 是否使用MixF3N代替F3N，目前仅对cs win trans block生效
                if config['model']['mix_f3n'] != 0:
                    self.mix_f3n = True
                else:
                    self.mix_f3n = False

                # 是否给attention加一个CONV path，目前仅对cs win trans block生效
                if config['model']['conv_path'] != 0:
                    self.conv_path = True
                else:
                    self.conv_path = False

                # 是否使用滑窗逻辑强化cs win，只对于条带宽度不为1时生效
                # 顺便更改了条带宽度不为1的池化逻辑，直接池化到条带的宽度，提高数据利用率(原来补0)
                if config['model']['cs_sw'] != 0:
                    self.cs_sw = True
                else:
                    self.cs_sw = False

                # 是否为cswin引入不同宽度条带池化的机制来增强注意力，只对初始条带宽度1有效
                if config['model']['pool_strip'] != 0:
                    self.pool_strip = True
                    if config['model']['pool_strip'] == 1:
                        # 使用什么宽度的条带来池化增强当前窗口
                        self.pool_sw = 1
                    elif config['model']['pool_strip'] == 2:
                        self.pool_sw = 2
                    elif config['model']['pool_strip'] == 4:
                        self.pool_sw = 4
                    else:
                        raise Exception('Not implement.')
                else:
                    self.pool_strip = False
                    self.pool_sw = 2

                # 定义transformer的深度
                if config['model']['depths'] != 0:
                    self.depths = config['model']['depths']
                else:
                    # 使用网络默认的深度
                    self.depths = None

                # 定义新trans主干不同层的条带宽度
                if config['model']['sw_list'] != 0:
                    self.sw_list = config['model']['sw_list']
                else:
                    # 使用网络默认的深度
                    self.sw_list = []

                # 定义新trans主干不同层的head数量
                if config['model']['head_list'] != 0:
                    self.head_list = config['model']['head_list']
                else:
                    # 使用网络默认的head数量，也就是每层4个
                    self.head_list = []

                # 定义不同的stage拥有多少个block
                if config['model']['blk_list'] != 0:
                    self.blk_list = config['model']['blk_list']
                else:
                    # 使用网络默认的blk数量，也就是深度的数量
                    self.blk_list = []

                self.netG = net.InpaintGenerator(
                    skip_dcn=self.skip_dcn, flow_guide=self.flow_guide, token_fusion=self.token_fusion,
                    token_fusion_simple=self.token_fusion_simple, fusion_skip_connect=self.fusion_skip_connect,
                    memory=self.memory, max_mem_len=config['model']['max_mem_len'],
                    compression_factor=config['model']['compression_factor'], mem_pool=self.mem_pool,
                    store_lf=self.store_lf, align_cache=self.align_cache, sub_token_align=self.sub_token_align,
                    sub_factor=self.sub_factor, half_memory=self.half_memory, last_memory=self.last_memory,
                    cross_att=self.cross_att, time_att=self.time_att, time_deco=self.time_deco,
                    temp_focal=self.temp_focal, cs_win=self.cs_win, mem_att=self.mem_att, cs_focal=self.cs_focal,
                    cs_focal_v2=self.cs_focal_v2,
                    cs_trans=self.cs_trans, mix_f3n=self.mix_f3n, conv_path=self.conv_path, cs_sw=self.cs_sw,
                    pool_strip=self.pool_strip, pool_sw=self.pool_sw, depths=self.depths, sw_list=self.sw_list,
                    head_list=self.head_list, blk_list=self.blk_list)
            else:
                self.netG = net.InpaintGenerator(
                    skip_dcn=self.skip_dcn, flow_guide=self.flow_guide, token_fusion=self.token_fusion,
                    token_fusion_simple=self.token_fusion_simple, fusion_skip_connect=self.fusion_skip_connect,
                    memory=self.memory, max_mem_len=config['model']['max_mem_len'],
                    compression_factor=config['model']['compression_factor'], mem_pool=self.mem_pool,
                    store_lf=self.store_lf, align_cache=self.align_cache, sub_token_align=self.sub_token_align,
                    sub_factor=self.sub_factor, half_memory=self.half_memory, last_memory=self.last_memory,
                    cross_att=self.cross_att, time_att=self.time_att, time_deco=self.time_deco,
                    temp_focal=self.temp_focal, cs_win=self.cs_win, mem_att=self.mem_att, cs_focal=self.cs_focal,
                    cs_focal_v2=self.cs_focal_v2,
                    cs_trans=self.cs_trans)
        else:
            self.netG = net.InpaintGenerator()
        print(self.netG)
        self.netG = self.netG.to(self.config['device'])
        if not self.config['model']['no_dis']:
            self.netD = net.Discriminator(
                in_channels=3,
                use_sigmoid=config['losses']['GAN_LOSS'] != 'hinge')
            self.netD = self.netD.to(self.config['device'])

        # setup optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()
        self.load()

        if config['distributed']:
            self.netG = DDP(self.netG,
                            device_ids=[self.config['local_rank']],
                            output_device=self.config['local_rank'],
                            broadcast_buffers=True,
                            find_unused_parameters=True)
            if not self.config['model']['no_dis']:
                self.netD = DDP(self.netD,
                                device_ids=[self.config['local_rank']],
                                output_device=self.config['local_rank'],
                                broadcast_buffers=True,
                                find_unused_parameters=False)

        # set summary writer
        self.dis_writer = None
        self.gen_writer = None
        self.summary = {}
        if self.config['global_rank'] == 0 or (not config['distributed']):
            self.dis_writer = SummaryWriter(
                os.path.join(config['save_dir'], 'dis'))
            self.gen_writer = SummaryWriter(
                os.path.join(config['save_dir'], 'gen'))

    def setup_optimizers(self):
        """Set up optimizers."""
        backbone_params = []
        spynet_params = []
        for name, param in self.netG.named_parameters():
            if 'update_spynet' in name:
                spynet_params.append(param)
            else:
                backbone_params.append(param)

        optim_params = [
            {
                'params': backbone_params,
                'lr': self.config['trainer']['lr']
            },
            {  # finetuning learning rate for spynet
                'params': spynet_params,
                'lr': self.config['trainer']['lr'] * self.spynet_lr
            },
        ]

        self.optimG = torch.optim.Adam(optim_params,
                                       betas=(self.config['trainer']['beta1'],
                                              self.config['trainer']['beta2']))

        if not self.config['model']['no_dis']:
            self.optimD = torch.optim.Adam(
                self.netD.parameters(),
                lr=self.config['trainer']['lr'],
                betas=(self.config['trainer']['beta1'],
                       self.config['trainer']['beta2']))

    def setup_schedulers(self):
        """Set up schedulers."""
        scheduler_opt = self.config['trainer']['scheduler']
        scheduler_type = scheduler_opt.pop('type')

        if scheduler_type in ['MultiStepLR', 'MultiStepRestartLR']:
            self.scheG = MultiStepRestartLR(
                self.optimG,
                milestones=scheduler_opt['milestones'],
                gamma=scheduler_opt['gamma'])
            self.scheD = MultiStepRestartLR(
                self.optimD,
                milestones=scheduler_opt['milestones'],
                gamma=scheduler_opt['gamma'])
        elif scheduler_type == 'CosineAnnealingRestartLR':
            self.scheG = CosineAnnealingRestartLR(
                self.optimG,
                periods=scheduler_opt['periods'],
                restart_weights=scheduler_opt['restart_weights'])
            self.scheD = CosineAnnealingRestartLR(
                self.optimD,
                periods=scheduler_opt['periods'],
                restart_weights=scheduler_opt['restart_weights'])
        else:
            raise NotImplementedError(
                f'Scheduler {scheduler_type} is not implemented yet.')

    def update_learning_rate(self):
        """Update learning rate."""
        self.scheG.step()
        self.scheD.step()

    def get_lr(self):
        """Get current learning rate."""
        return self.optimG.param_groups[0]['lr']

    def add_summary(self, writer, name, val):
        """Add tensorboard summary."""
        if name not in self.summary:
            self.summary[name] = 0
        self.summary[name] += val
        if writer is not None and self.iteration % 100 == 0:
            writer.add_scalar(name, self.summary[name] / 100, self.iteration)
            self.summary[name] = 0

    def load(self):
        """Load netG (and netD)."""
        # get the latest checkpoint
        model_path = self.config['save_dir']
        if os.path.isfile(os.path.join(model_path, 'latest.ckpt')):
            latest_epoch = open(os.path.join(model_path, 'latest.ckpt'),
                                'r').read().splitlines()[-1]
        else:
            ckpts = [
                os.path.basename(i).split('.pth')[0]
                for i in glob.glob(os.path.join(model_path, '*.pth'))
            ]
            ckpts.sort()
            latest_epoch = ckpts[-1] if len(ckpts) > 0 else None

        if latest_epoch is not None:
            gen_path = os.path.join(model_path,
                                    f'gen_{int(latest_epoch):06d}.pth')
            dis_path = os.path.join(model_path,
                                    f'dis_{int(latest_epoch):06d}.pth')
            opt_path = os.path.join(model_path,
                                    f'opt_{int(latest_epoch):06d}.pth')

            if self.config['global_rank'] == 0:
                print(f'Loading model from {gen_path}...')
            dataG = torch.load(gen_path, map_location=self.config['device'])
            self.netG.load_state_dict(dataG)
            if not self.config['model']['no_dis']:
                dataD = torch.load(dis_path,
                                   map_location=self.config['device'])
                self.netD.load_state_dict(dataD)

            data_opt = torch.load(opt_path, map_location=self.config['device'])
            self.optimG.load_state_dict(data_opt['optimG'])
            self.scheG.load_state_dict(data_opt['scheG'])
            if not self.config['model']['no_dis']:
                self.optimD.load_state_dict(data_opt['optimD'])
                self.scheD.load_state_dict(data_opt['scheD'])
            self.epoch = data_opt['epoch']
            self.iteration = data_opt['iteration']

        else:
            if self.config['global_rank'] == 0:
                print('Warnning: There is no trained model found.'
                      'An initialized model will be used.')

    def save(self, it):
        """Save parameters every eval_epoch"""
        if self.config['global_rank'] == 0:
            # configure path
            gen_path = os.path.join(self.config['save_dir'],
                                    f'gen_{it:06d}.pth')
            dis_path = os.path.join(self.config['save_dir'],
                                    f'dis_{it:06d}.pth')
            opt_path = os.path.join(self.config['save_dir'],
                                    f'opt_{it:06d}.pth')
            print(f'\nsaving model to {gen_path} ...')

            # remove .module for saving
            if isinstance(self.netG, torch.nn.DataParallel) \
               or isinstance(self.netG, DDP):
                netG = self.netG.module
                if not self.config['model']['no_dis']:
                    netD = self.netD.module
            else:
                netG = self.netG
                if not self.config['model']['no_dis']:
                    netD = self.netD

            # save checkpoints
            torch.save(netG.state_dict(), gen_path)
            if not self.config['model']['no_dis']:
                # 可以选择不存储判别器，opt存下来的是优化器的状态
                torch.save(netD.state_dict(), dis_path)
                torch.save(
                    {
                        'epoch': self.epoch,
                        'iteration': self.iteration,
                        'optimG': self.optimG.state_dict(),
                        'optimD': self.optimD.state_dict(),
                        'scheG': self.scheG.state_dict(),
                        'scheD': self.scheD.state_dict()
                    }, opt_path)
            else:
                torch.save(
                    {
                        'epoch': self.epoch,
                        'iteration': self.iteration,
                        'optimG': self.optimG.state_dict(),
                        'scheG': self.scheG.state_dict()
                    }, opt_path)

            latest_path = os.path.join(self.config['save_dir'], 'latest.ckpt')
            os.system(f"echo {it:06d} > {latest_path}")

    def train(self):
        """training entry"""
        pbar = range(int(self.train_args['iterations']))
        if self.config['global_rank'] == 0:
            pbar = tqdm(pbar,
                        initial=self.iteration,
                        dynamic_ncols=True,
                        smoothing=0.01)

        os.makedirs('logs', exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(filename)s[line:%(lineno)d]"
            "%(levelname)s %(message)s",
            datefmt="%a, %d %b %Y %H:%M:%S",
            filename=f"logs/{self.config['save_dir'].split('/')[-1]}.log",
            filemode='w')

        while True:
            self.epoch += 1
            if self.config['distributed']:
                self.train_sampler.set_epoch(self.epoch)

            if not self.memory:
                self._train_epoch(pbar)
            else:
                # 序列视频输入用于记忆力训练
                if not self.same_mask:
                    # 使用随机mask训练
                    self._train_epoch_mem(pbar)
                else:
                    # 使用固定mask训练，需要返回mask的字典
                    self._train_epoch_mem_mask(pbar)

            if self.iteration > self.train_args['iterations']:
                break
        print('\nEnd training....')

    def _train_epoch(self, pbar):
        """Process input and calculate loss every training epoch"""
        device = self.config['device']

        # temp for compare
        # for frames, masks, _ in self.train_loader:
        for frames, masks, video_name, index, start_index in self.train_loader:
            self.iteration += 1

            frames, masks = frames.to(device), masks.to(device)
            l_t = self.num_local_frames
            b, t, c, h, w = frames.size()

            masked_frames = (frames * (1 - masks).float())
            gt_local_frames = (frames[:, :l_t, ...] + 1) / 2

            pred_imgs, pred_flows = self.netG(masked_frames, l_t)
            pred_imgs = pred_imgs.view(b, -1, c, h, w)
            comp_imgs = frames * (1. - masks) + masks * pred_imgs

            # compute flow completion loss
            flow_loss = self.flow_comp_loss(pred_flows, gt_local_frames)

            gen_loss = 0
            dis_loss = 0

            if not self.config['model']['no_dis']:
                # discriminator adversarial loss
                real_clip = self.netD(frames)
                fake_clip = self.netD(comp_imgs.detach())
                dis_real_loss = self.adversarial_loss(real_clip, True, True)
                dis_fake_loss = self.adversarial_loss(fake_clip, False, True)
                dis_loss += (dis_real_loss + dis_fake_loss) / 2
                self.add_summary(self.dis_writer, 'loss/dis_vid_fake',
                                 dis_fake_loss.item())
                self.add_summary(self.dis_writer, 'loss/dis_vid_real',
                                 dis_real_loss.item())
                self.optimD.zero_grad()
                dis_loss.backward()
                self.optimD.step()

                # generator adversarial loss
                gen_clip = self.netD(comp_imgs)
                gan_loss = self.adversarial_loss(gen_clip, True, False)
                gan_loss = gan_loss \
                    * self.config['losses']['adversarial_weight']
                gen_loss += gan_loss
                self.add_summary(self.gen_writer, 'loss/gan_loss',
                                 gan_loss.item())

            flow_loss = flow_loss * self.config['losses']['flow_weight']
            gen_loss += flow_loss
            self.add_summary(self.gen_writer, 'loss/flow_loss',
                             flow_loss.item())

            # generator l1 loss
            hole_loss = self.l1_loss(pred_imgs * masks, frames * masks)
            # 空洞内的loss
            hole_loss = hole_loss / torch.mean(masks) \
                * self.config['losses']['hole_weight']
            gen_loss += hole_loss
            self.add_summary(self.gen_writer, 'loss/hole_loss',
                             hole_loss.item())

            valid_loss = self.l1_loss(pred_imgs * (1 - masks),
                                      frames * (1 - masks))
            # 非遮挡区域的loss
            valid_loss = valid_loss / torch.mean(1-masks) \
                * self.config['losses']['valid_weight']
            gen_loss += valid_loss
            self.add_summary(self.gen_writer, 'loss/valid_loss',
                             valid_loss.item())

            self.optimG.zero_grad()
            # gen loss 是对抗、光流、空洞和非遮挡区域loss之和
            gen_loss.backward()
            self.optimG.step()

            self.update_learning_rate()

            # console logs
            if self.config['global_rank'] == 0:
                pbar.update(1)
                if not self.config['model']['no_dis']:
                    pbar.set_description((f"flow: {flow_loss.item():.3f}; "
                                          f"d: {dis_loss.item():.3f}; "
                                          f"hole: {hole_loss.item():.3f}; "
                                          f"valid: {valid_loss.item():.3f}"))
                else:
                    pbar.set_description((f"flow: {flow_loss.item():.3f}; "
                                          f"hole: {hole_loss.item():.3f}; "
                                          f"valid: {valid_loss.item():.3f}"))

                if self.iteration % self.train_args['log_freq'] == 0:
                    if not self.config['model']['no_dis']:
                        logging.info(f"[Iter {self.iteration}] "
                                     f"flow: {flow_loss.item():.4f}; "
                                     f"d: {dis_loss.item():.4f}; "
                                     f"hole: {hole_loss.item():.4f}; "
                                     f"valid: {valid_loss.item():.4f}")
                    else:
                        logging.info(f"[Iter {self.iteration}] "
                                     f"flow: {flow_loss.item():.4f}; "
                                     f"hole: {hole_loss.item():.4f}; "
                                     f"valid: {valid_loss.item():.4f}")

            # saving models
            if self.iteration % self.train_args['save_freq'] == 0:
                self.save(int(self.iteration))

            if self.iteration > self.train_args['iterations']:
                break

    def _train_epoch_mem(self, pbar):
        """Process input and calculate loss every training epoch in a sequence manner with memory"""
        device = self.config['device']

        # debug
        # video_index_list = []
        # start_index_list = []
        # video_name_list = []
        # ii = 0
        # # torch.autograd.set_detect_anomaly(True)

        for frames, masks, video_name, index, start_index in self.train_loader:
            self.iteration += 1

            # 当有新视频出现时，即start_index为0时，清空记忆缓存
            for start_idx in start_index:
                if start_idx == 0:
                    for blk in self.netG.transformer:
                        try:
                            # 清空有记忆力的层的记忆缓存
                            blk.attn.m_k = []
                            blk.attn.m_v = []
                        except:
                            pass

            # debug
            # video_index_list.append(index)
            # start_index_list.append(start_index)
            # video_name_list.append(video_name)
            # ii += 1
            # try:
            #     print('-' * 50)
            #     print('[Bacth 0] Video Index: %d, Start Frame Index: %d || [Bacth 1] Video Index: %d, Start Frame Index: %d || iter: %d'
            #           % (index[0], start_index[0], index[1], start_index[1], ii))
            #     # print('[Bacth 2] Video Index: %d, Start Frame Index: %d || [Bacth 3] Video Index: %d, Start Frame Index: %d || iter: %d'
            #     #       % (index[2], start_index[2], index[3], start_index[3], ii))
            #     print('-'*50)
            # except:
            #     pass
            # if ii > 10000:
            #     break

            frames, masks = frames.to(device), masks.to(device)
            l_t = self.num_local_frames
            b, t, c, h, w = frames.size()

            masked_frames = (frames * (1 - masks).float())
            gt_local_frames = (frames[:, :l_t, ...] + 1) / 2

            pred_imgs, pred_flows = self.netG(masked_frames, l_t)
            pred_imgs = pred_imgs.view(b, -1, c, h, w)
            comp_imgs = frames * (1. - masks) + masks * pred_imgs

            # compute flow completion loss
            flow_loss = self.flow_comp_loss(pred_flows, gt_local_frames)

            gen_loss = 0
            dis_loss = 0

            if not self.config['model']['no_dis']:
                # discriminator adversarial loss
                real_clip = self.netD(frames)
                fake_clip = self.netD(comp_imgs.detach())
                dis_real_loss = self.adversarial_loss(real_clip, True, True)
                dis_fake_loss = self.adversarial_loss(fake_clip, False, True)
                dis_loss += (dis_real_loss + dis_fake_loss) / 2
                self.add_summary(self.dis_writer, 'loss/dis_vid_fake',
                                 dis_fake_loss.item())
                self.add_summary(self.dis_writer, 'loss/dis_vid_real',
                                 dis_real_loss.item())
                self.optimD.zero_grad()
                dis_loss.backward()
                self.optimD.step()

                # generator adversarial loss
                gen_clip = self.netD(comp_imgs)
                gan_loss = self.adversarial_loss(gen_clip, True, False)
                gan_loss = gan_loss \
                           * self.config['losses']['adversarial_weight']
                gen_loss += gan_loss
                self.add_summary(self.gen_writer, 'loss/gan_loss',
                                 gan_loss.item())

            flow_loss = flow_loss * self.config['losses']['flow_weight']
            gen_loss += flow_loss
            self.add_summary(self.gen_writer, 'loss/flow_loss',
                             flow_loss.item())

            # generator l1 loss
            hole_loss = self.l1_loss(pred_imgs * masks, frames * masks)
            # 空洞内的loss
            hole_loss = hole_loss / torch.mean(masks) \
                        * self.config['losses']['hole_weight']
            gen_loss += hole_loss
            self.add_summary(self.gen_writer, 'loss/hole_loss',
                             hole_loss.item())

            valid_loss = self.l1_loss(pred_imgs * (1 - masks),
                                      frames * (1 - masks))
            # 非遮挡区域的loss
            valid_loss = valid_loss / torch.mean(1 - masks) \
                         * self.config['losses']['valid_weight']
            gen_loss += valid_loss
            self.add_summary(self.gen_writer, 'loss/valid_loss',
                             valid_loss.item())

            self.optimG.zero_grad()
            # gen loss 是对抗、光流、空洞和非遮挡区域loss之和
            gen_loss.backward()
            self.optimG.step()

            self.update_learning_rate()

            # console logs
            if self.config['global_rank'] == 0:
                pbar.update(1)
                if not self.config['model']['no_dis']:
                    pbar.set_description((f"flow: {flow_loss.item():.3f}; "
                                          f"d: {dis_loss.item():.3f}; "
                                          f"hole: {hole_loss.item():.3f}; "
                                          f"valid: {valid_loss.item():.3f}"))
                else:
                    pbar.set_description((f"flow: {flow_loss.item():.3f}; "
                                          f"hole: {hole_loss.item():.3f}; "
                                          f"valid: {valid_loss.item():.3f}"))

                if self.iteration % self.train_args['log_freq'] == 0:
                    if not self.config['model']['no_dis']:
                        logging.info(f"[Iter {self.iteration}] "
                                     f"flow: {flow_loss.item():.4f}; "
                                     f"d: {dis_loss.item():.4f}; "
                                     f"hole: {hole_loss.item():.4f}; "
                                     f"valid: {valid_loss.item():.4f}")
                    else:
                        logging.info(f"[Iter {self.iteration}] "
                                     f"flow: {flow_loss.item():.4f}; "
                                     f"hole: {hole_loss.item():.4f}; "
                                     f"valid: {valid_loss.item():.4f}")

            # saving models
            if self.iteration % self.train_args['save_freq'] == 0:
                self.save(int(self.iteration))

            if self.iteration > self.train_args['iterations']:
                break

    def _train_epoch_mem_mask(self, pbar):
        """Process input and calculate loss every training epoch in a sequence manner with memory"""
        device = self.config['device']

        # # debug
        # video_index_list = []
        # start_index_list = []
        # video_name_list = []
        # ii = 0
        # torch.autograd.set_detect_anomaly(True)

        for frames, masks, video_name, index, start_index, new_mask, mask_dict in self.train_loader:
            self.iteration += 1

            # 当有新视频出现时，即start_index为0时，清空记忆缓存
            for start_idx in start_index:
                if start_idx == 0:
                    # 清空记忆缓存
                    for blk in self.netG.transformer:
                        try:
                            # 清空有记忆力的层的记忆缓存
                            blk.attn.m_k = []
                            blk.attn.m_v = []
                        except:
                            pass

            # # 就让你等于之前的mask_dict就完事了，反正到了新视频会重新生成
            # self.train_loader.dataset.random_dict_list = mask_dict
            # self.train_dataset.random_dict_list = mask_dict

            # # debug
            # video_index_list.append(index)
            # start_index_list.append(start_index)
            # video_name_list.append(video_name)
            # ii += 1
            # if new_mask[0]:
            #     new_flag_0 = 1
            # else:
            #     new_flag_0 = 0
            # if new_mask[1]:
            #     new_flag_1 = 1
            # else:
            #     new_flag_1 = 0
            # try:
            #     print('-' * 50)
            #     print('[Bacth 0] Video Index: %d, Start Frame Index: %d, New Mask: %d '
            #           '|| [Bacth 1] Video Index: %d, Start Frame Index: %d, New Mask: %d|| iter: %d'
            #           % (index[0], start_index[0], new_flag_0, index[1], start_index[1], new_flag_1, ii))
            #     # print('[Bacth 2] Video Index: %d, Start Frame Index: %d || [Bacth 3] Video Index: %d, Start Frame Index: %d || iter: %d'
            #     #       % (index[2], start_index[2], index[3], start_index[3], ii))
            #     print('-'*50)
            # except:
            #     pass
            # if ii > 100:
            #     break

            frames, masks = frames.to(device), masks.to(device)
            l_t = self.num_local_frames
            b, t, c, h, w = frames.size()

            masked_frames = (frames * (1 - masks).float())
            gt_local_frames = (frames[:, :l_t, ...] + 1) / 2

            pred_imgs, pred_flows = self.netG(masked_frames, l_t)
            pred_imgs = pred_imgs.view(b, -1, c, h, w)
            comp_imgs = frames * (1. - masks) + masks * pred_imgs

            # compute flow completion loss
            flow_loss = self.flow_comp_loss(pred_flows, gt_local_frames)

            gen_loss = 0
            dis_loss = 0

            if not self.config['model']['no_dis']:
                # discriminator adversarial loss
                real_clip = self.netD(frames)
                fake_clip = self.netD(comp_imgs.detach())
                dis_real_loss = self.adversarial_loss(real_clip, True, True)
                dis_fake_loss = self.adversarial_loss(fake_clip, False, True)
                dis_loss += (dis_real_loss + dis_fake_loss) / 2
                self.add_summary(self.dis_writer, 'loss/dis_vid_fake',
                                 dis_fake_loss.item())
                self.add_summary(self.dis_writer, 'loss/dis_vid_real',
                                 dis_real_loss.item())
                self.optimD.zero_grad()
                dis_loss.backward()
                self.optimD.step()

                # generator adversarial loss
                gen_clip = self.netD(comp_imgs)
                gan_loss = self.adversarial_loss(gen_clip, True, False)
                gan_loss = gan_loss \
                           * self.config['losses']['adversarial_weight']
                gen_loss += gan_loss
                self.add_summary(self.gen_writer, 'loss/gan_loss',
                                 gan_loss.item())

            flow_loss = flow_loss * self.config['losses']['flow_weight']
            gen_loss += flow_loss
            self.add_summary(self.gen_writer, 'loss/flow_loss',
                             flow_loss.item())

            # generator l1 loss
            hole_loss = self.l1_loss(pred_imgs * masks, frames * masks)
            # 空洞内的loss
            hole_loss = hole_loss / torch.mean(masks) \
                        * self.config['losses']['hole_weight']
            gen_loss += hole_loss
            self.add_summary(self.gen_writer, 'loss/hole_loss',
                             hole_loss.item())

            valid_loss = self.l1_loss(pred_imgs * (1 - masks),
                                      frames * (1 - masks))
            # 非遮挡区域的loss
            valid_loss = valid_loss / torch.mean(1 - masks) \
                         * self.config['losses']['valid_weight']
            gen_loss += valid_loss
            self.add_summary(self.gen_writer, 'loss/valid_loss',
                             valid_loss.item())

            self.optimG.zero_grad()
            # gen loss 是对抗、光流、空洞和非遮挡区域loss之和
            gen_loss.backward()
            self.optimG.step()

            self.update_learning_rate()

            # console logs
            if self.config['global_rank'] == 0:
                pbar.update(1)
                if not self.config['model']['no_dis']:
                    pbar.set_description((f"flow: {flow_loss.item():.3f}; "
                                          f"d: {dis_loss.item():.3f}; "
                                          f"hole: {hole_loss.item():.3f}; "
                                          f"valid: {valid_loss.item():.3f}"))
                else:
                    pbar.set_description((f"flow: {flow_loss.item():.3f}; "
                                          f"hole: {hole_loss.item():.3f}; "
                                          f"valid: {valid_loss.item():.3f}"))

                if self.iteration % self.train_args['log_freq'] == 0:
                    if not self.config['model']['no_dis']:
                        logging.info(f"[Iter {self.iteration}] "
                                     f"flow: {flow_loss.item():.4f}; "
                                     f"d: {dis_loss.item():.4f}; "
                                     f"hole: {hole_loss.item():.4f}; "
                                     f"valid: {valid_loss.item():.4f}")
                    else:
                        logging.info(f"[Iter {self.iteration}] "
                                     f"flow: {flow_loss.item():.4f}; "
                                     f"hole: {hole_loss.item():.4f}; "
                                     f"valid: {valid_loss.item():.4f}")

            # saving models
            if self.iteration % self.train_args['save_freq'] == 0:
                self.save(int(self.iteration))

            if self.iteration > self.train_args['iterations']:
                break
