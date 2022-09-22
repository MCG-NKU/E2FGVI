''' Towards An End-to-End Framework for Video Inpainting
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

# from model.modules.flow_comp import SPyNet
from model.modules.flow_comp_MFN import MaskFlowNetS
from model.modules.feat_prop import BidirectionalPropagation, SecondOrderDeformableAlignment
from model.modules.tfocal_transformer_hq import TemporalFocalTransformerBlock, SoftSplit, SoftComp,\
    SoftSplit_FlowGuide, TokenSlimmingModule, ReverseTSM, ReverseTSM_v2, Decoupled3DFocalTransformerBlock
from model.modules.spectral_norm import spectral_norm as _spectral_norm


class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def print_network(self):
        if isinstance(self, list):
            self = self[0]
        num_params = 0
        for param in self.parameters():
            num_params += param.numel()
        print(
            'Network [%s] was created. Total number of parameters: %.1f million. '
            'To see the architecture, do print(network).' %
            (type(self).__name__, num_params / 1000000))

    def init_weights(self, init_type='normal', gain=0.02):
        '''
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        '''
        def init_func(m):
            classname = m.__class__.__name__
            if classname.find('InstanceNorm2d') != -1:
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.constant_(m.weight.data, 1.0)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif hasattr(m, 'weight') and (classname.find('Conv') != -1
                                           or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'xavier_uniform':
                    nn.init.xavier_uniform_(m.weight.data, gain=1.0)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)
                elif init_type == 'none':  # uses pytorch's default init method
                    m.reset_parameters()
                else:
                    raise NotImplementedError(
                        'initialization method [%s] is not implemented' %
                        init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

        # propagate to children
        for m in self.children():
            if hasattr(m, 'init_weights'):
                m.init_weights(init_type, gain)


class Encoder(nn.Module):
    # def __init__(self):     # default, out channel不会随着channel改变导致bug
    def __init__(self, out_channel=128, reduction=1):
        super(Encoder, self).__init__()
        self.group = [1, 2, 4, 8, 1]
        self.layers = nn.ModuleList([
            nn.Conv2d(3, 64//reduction, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64//reduction, 64//reduction, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64//reduction, 128//reduction, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128//reduction, 256//reduction, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256//reduction, 384//reduction, kernel_size=3, stride=1, padding=1, groups=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(640//reduction, 512//reduction, kernel_size=3, stride=1, padding=1, groups=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(768//reduction, 384//reduction, kernel_size=3, stride=1, padding=1, groups=4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(640//reduction, 256//reduction, kernel_size=3, stride=1, padding=1, groups=8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512//reduction, out_channel, kernel_size=3, stride=1, padding=1, groups=1),
            nn.LeakyReLU(0.2, inplace=True)
        ])

    def forward(self, x):
        bt, c, _, _ = x.size()
        # h, w = h//4, w//4
        out = x
        for i, layer in enumerate(self.layers):
            if i == 8:
                x0 = out
                _, _, h, w = x0.size()
            if i > 8 and i % 2 == 0:
                g = self.group[(i - 8) // 2]
                x = x0.view(bt, g, -1, h, w)
                o = out.view(bt, g, -1, h, w)
                out = torch.cat([x, o], 2).view(bt, -1, h, w)
            out = layer(out)
        return out


class deconv(nn.Module):
    def __init__(self,
                 input_channel,
                 output_channel,
                 kernel_size=3,
                 padding=0):
        super().__init__()
        self.conv = nn.Conv2d(input_channel,
                              output_channel,
                              kernel_size=kernel_size,
                              stride=1,
                              padding=padding)

    def forward(self, x):
        x = F.interpolate(x,
                          scale_factor=2,
                          mode='bilinear',
                          align_corners=True)
        return self.conv(x)


class InpaintGenerator(BaseNetwork):
    def __init__(self, init_weights=True, flow_align=True, skip_dcn=False, flow_guide=False,
                 token_fusion=False, token_fusion_simple=False, fusion_skip_connect=False,
                 memory=False, max_mem_len=1, compression_factor=1, mem_pool=False, store_lf=False, align_cache=False,
                 sub_token_align=False, sub_factor=1, half_memory=False, last_memory=False,
                 cross_att=False, time_att=False, time_deco=False, temp_focal=False, cs_win=False, mem_att=False,
                 cs_focal=False, cs_focal_v2=False, cs_trans=False, mix_f3n=False, conv_path=False,
                 cs_sw=False, pool_strip=False, pool_sw=1, depths=None, sw_list=[], head_list=[], blk_list=[]):
        super(InpaintGenerator, self).__init__()

        # large model:
        channel = 256   # default
        hidden = 512    # default
        reduction = 1   # default

        # small model
        # channel = 64
        # hidden = 128
        # reduction = 2

        # 设置transformer参数
        # 设置trans block的数量
        if depths is None:
            # depths = 8    # default
            # depths = 4       # for cswin tiny d4 model
            depths = 2  # 0.08s/frame, 0.07s/frame with hidden = 128,
        else:
            depths = depths

        # 只有一个stage
        if not blk_list:
            # 设置不同层的条带宽度
            if not sw_list:
                # 默认条件下不同层的条带宽度都是1
                for sw_idx in range(0, depths):
                    sw_list.append(1)
            else:
                # 不同层使用不同的宽度
                sw_list = sw_list

            # 设置不同层的head数量
            if not head_list:
                # 默认条件下每层都使用4个head，对于cswin相当于宽度和高度各2个head
                num_heads = [4] * depths
            else:
                # 不同层使用不同的head数量
                num_heads = head_list
        else:
            # 有多个stage
            depth_cnt = 0
            self.sw_list = sw_list  # 存一下不同stage的条带宽度设置
            self.head_list = head_list  # 存一下不同stage的head设置
            sw_list = []
            num_heads = []
            for stage_idx in range(0, len(blk_list)):
                depth_cnt += blk_list[stage_idx]
                # 对于每一层都有一个循环
                for depth_idx in range(0, blk_list[stage_idx]):
                    # 设置不同层的条带宽度
                    sw_list.append(self.sw_list[stage_idx])
                    # 设置不同层的head数量
                    num_heads.append(self.head_list[stage_idx])

            # 检查一下深度的数量和stage定义的数量是否相等
            if depth_cnt != depths:
                raise Exception('Wrong transformer structure config.')

        # 光流引导特征嵌入
        self.flow_guide = flow_guide
        # token空间缩减
        self.token_fusion = token_fusion
        # 共用token空间缩减和扩展模块
        self.token_fusion_simple = token_fusion_simple
        # 在token空间扩展时使用缩减前的特征进行跳跃连接
        self.fusion_skip_connect = fusion_skip_connect
        # 引入Memory机制存储上次的补全feat
        self.memory = memory

        # if self.memory:
        max_mem_len = max_mem_len                   # 记忆的最长存储时间，以forward次数为单位
        compression_factor = compression_factor     # 记忆张量的压缩系数，通道以及空间共用
        mem_pool = mem_pool                         # 是否使用池化来进一步在空间上压缩记忆张量
        store_lf = store_lf                         # 是否仅在记忆缓存中存储局部帧的kv张量
        align_cache = align_cache                   # 是否在增强 k v 前对齐缓存和当前帧
        sub_token_align = sub_token_align           # 是否在对齐缓存和当前帧时对token通道分组来实现sub-token对齐
        sub_factor = sub_factor                     # sub-token对齐的分组系数，分组系数越大计算损耗越高，分辨率精度越高
        half_memory = half_memory                   # 如果为True，则只有一半的block有记忆力
        last_memory = last_memory                   # 如果为True，则只有最后一层的block有记忆力
        cross_att = cross_att                       # 如果为True，使用cross attention融合记忆与当前帧
        time_att = time_att                         # 如果为True，使用cross attention额外在T维度融合记忆与当前帧
        time_deco = time_deco                       # 如果为True，则cross attention会把时间和空间解耦
        temp_focal = temp_focal                     # 如果为True，则cross attention的时空记忆聚合基于temp focal att实现
        cs_win = cs_win                             # 如果为True，则cross attention的时空记忆聚合基于cswin att实现
        mem_att = mem_att                           # 如果为True，则使用cross att直接聚合不同迭代的记忆和当前特征
        cs_focal = cs_focal                         # 如果为True，则为cs win增强池化的focal机制
        cs_focal_v2 = cs_focal_v2                   # 如果为True，则cs win的focal基于与池化完的张量方向相同的滑窗实现
        # cs_win_strip = cs_win_strip                 # 决定了 cs win 的条带宽度，默认为1，目前已被sw_list替换；tf主干默认用条带宽度1的
        cs_trans = cs_trans                         # 如果为True，则使用我们增强的cswin替代temporal focal作为trans主干
        mix_f3n = mix_f3n                           # 如果为True，则使用MixF3N代替原本的F3N，目前仅对于cswin主干生效
        conv_path = conv_path                       # 如果为True，则给attention额外引入CONV path，目前仅对于cswin主干生效
        cs_sw = cs_sw                               # 如果为True，使用滑窗逻辑强化cswin，只对于条带宽度大于1和cswin主干生效
        pool_strip = pool_strip                     # 如果为True，则将不同宽度的条带池化到1来增强当前窗口，只对初始条带为1有效
        pool_sw = pool_sw                           # 用来池化增强当前条带的条带宽度

        # encoder
        # self.encoder = Encoder()    # default
        self.encoder = Encoder(out_channel=channel // 2, reduction=reduction)

        # decoder-default
        self.decoder = nn.Sequential(
            deconv(channel // 2, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            deconv(64, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1))

        # # decoder-lite
        # self.decoder = nn.Sequential(
        #     deconv(channel // 2, 128//reduction, kernel_size=3, padding=1),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Conv2d(128//reduction, 64//reduction, kernel_size=3, stride=1, padding=1),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     deconv(64//reduction, 64//reduction, kernel_size=3, padding=1),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Conv2d(64//reduction, 3, kernel_size=3, stride=1, padding=1))

        # feature propagation module
        self.feat_prop_module = BidirectionalPropagation(channel // 2, flow_align=flow_align, skip_dcn=skip_dcn)

        # soft split and soft composition
        kernel_size = (7, 7)    # 滑块的大小
        padding = (3, 3)    # 两个方向上隐式填0的数量
        stride = (3, 3)     # 滑块的步长
        output_size = (60, 108)
        t2t_params = {
            'kernel_size': kernel_size,
            'stride': stride,
            'padding': padding
        }
        if not self.flow_guide:
            self.ss = SoftSplit(channel // 2,
                                hidden,
                                kernel_size,
                                stride,
                                padding,
                                t2t_param=t2t_params)
        else:
            # 使用光流引导patch embedding
            self.ss = SoftSplit_FlowGuide(channel // 2,
                                hidden,
                                kernel_size,
                                stride,
                                padding,
                                t2t_param=t2t_params)
        self.sc = SoftComp(channel // 2, hidden, kernel_size, stride, padding)

        n_vecs = 1  # 计算token的数量
        for i, d in enumerate(kernel_size):
            n_vecs *= int((output_size[i] + 2 * padding[i] -
                           (d - 1) - 1) / stride[i] + 1)

        blocks = []
        window_size = [(5, 9)] * depths
        focal_windows = [(5, 9)] * depths
        focal_levels = [2] * depths
        pool_method = "fc"

        if self.token_fusion:
            # 融合相似度高的token来降低计算复杂度
            # 以下为token缩减层和token恢复层
            self.rtsm = nn.ModuleList()
            self.tsm = nn.ModuleList()

            num_patches = n_vecs    # 原始的token数量
            keeping_ratio = 0.5625  # 0.75*0.75

            # self.keeped_patches = [int(num_patches * ((keeping_ratio) ** i)) for i in range(depths)]
            self.keeped_patches = [int(num_patches * keeping_ratio)] * depths
            self.layer_patches = []
            self.stage_blocks = 1
            self.slim_index = []

            for i in range(depths):
                self.layer_patches += self.stage_blocks * [self.keeped_patches[i]]
            self.layer_patches += self.stage_blocks * [self.keeped_patches[-1]]

            if not token_fusion_simple:
                # 每个trans block前面一个token缩减，后面一个token复原
                for i in range(1, depths):
                    self.tsm.append(TokenSlimmingModule(hidden, self.keeped_patches[i]))
                    self.slim_index.append(i)

                dropped_token = 0
                if not self.fusion_skip_connect:
                    # 仅使用缩减后的token进行恢复
                    for i in range(depths):
                        dropped_token = max(dropped_token, num_patches - self.layer_patches[i + 1])
                        if dropped_token > 0:
                            self.rtsm.append(ReverseTSM(hidden, self.layer_patches[i + 1], num_patches))
                        else:
                            self.rtsm.append(nn.Identity())
                else:
                    # 融合缩减前的 trans feat 进行 token 恢复
                    for i in range(depths):
                        dropped_token = max(dropped_token, num_patches - self.layer_patches[i + 1])
                        if dropped_token > 0:
                            self.rtsm.append(ReverseTSM_v2(hidden, self.layer_patches[i + 1], num_patches))
                        else:
                            self.rtsm.append(nn.Identity())
            else:
                # 所有的trans block共用一个token缩减和一个token复原
                self.tsm.append(TokenSlimmingModule(hidden, self.keeped_patches[1]))
                self.slim_index.append(1)
                dropped_token = max(0, num_patches - self.layer_patches[1])

                if not self.fusion_skip_connect:
                    # 仅使用缩减后的token进行恢复
                    if dropped_token > 0:
                        self.rtsm.append(ReverseTSM(hidden, self.layer_patches[1], num_patches))
                    else:
                        self.rtsm.append(nn.Identity())
                else:
                    # 融合缩减前的 trans feat 进行 token 恢复
                    if dropped_token > 0:
                        self.rtsm.append(ReverseTSM_v2(hidden, self.layer_patches[1], num_patches))
                    else:
                        self.rtsm.append(nn.Identity())

        if not self.token_fusion:
            # default temporal focal transformer
            for i in range(depths):
                if not (half_memory or last_memory):
                    # 所有的层都有记忆
                    if not cs_trans:
                        # 使用tf trans主干
                        blocks.append(
                            TemporalFocalTransformerBlock(dim=hidden,
                                                          num_heads=num_heads[i],
                                                          window_size=window_size[i],
                                                          focal_level=focal_levels[i],
                                                          focal_window=focal_windows[i],
                                                          n_vecs=n_vecs,
                                                          t2t_params=t2t_params,
                                                          pool_method=pool_method,
                                                          memory=self.memory,
                                                          max_mem_len=max_mem_len,
                                                          compression_factor=compression_factor,
                                                          mem_pool=mem_pool,
                                                          store_lf=store_lf,
                                                          align_cache=align_cache,
                                                          sub_token_align=sub_token_align,
                                                          sub_factor=sub_factor,
                                                          cross_att=cross_att,
                                                          time_att=time_att,
                                                          time_deco=time_deco,
                                                          temp_focal=temp_focal,
                                                          cs_win=cs_win,
                                                          mem_att=mem_att,
                                                          cs_focal=cs_focal,
                                                          cs_focal_v2=cs_focal_v2,
                                                          cs_win_strip=1),)
                    else:
                        # 使用cs win主干
                        blocks.append(
                            Decoupled3DFocalTransformerBlock(dim=hidden,
                                                             num_heads=num_heads[i],
                                                             window_size=window_size[i],
                                                             focal_level=focal_levels[i],
                                                             focal_window=focal_windows[i],
                                                             n_vecs=n_vecs,
                                                             t2t_params=t2t_params,
                                                             memory=self.memory,
                                                             max_mem_len=max_mem_len,
                                                             compression_factor=compression_factor,
                                                             mem_pool=mem_pool,
                                                             store_lf=store_lf,
                                                             align_cache=align_cache,
                                                             sub_token_align=sub_token_align,
                                                             sub_factor=sub_factor,
                                                             cross_att=cross_att,
                                                             time_att=time_att,
                                                             time_deco=time_deco,
                                                             temp_focal=temp_focal,
                                                             cs_win=cs_win,
                                                             mem_att=mem_att,
                                                             cs_focal=cs_focal,
                                                             cs_focal_v2=cs_focal_v2,
                                                             cs_win_strip=sw_list[i],
                                                             mix_f3n=mix_f3n,
                                                             conv_path=conv_path,
                                                             cs_sw=cs_sw,
                                                             pool_strip=pool_strip,
                                                             pool_sw=pool_sw),)
                elif half_memory:
                    # 只有一半的层有记忆
                    if (i + 1) % 2 == 0:
                        # 偶数层(包括最后一层有记忆力)
                        if not cs_trans:
                            blocks.append(
                                TemporalFocalTransformerBlock(dim=hidden,
                                                              num_heads=num_heads[i],
                                                              window_size=window_size[i],
                                                              focal_level=focal_levels[i],
                                                              focal_window=focal_windows[i],
                                                              n_vecs=n_vecs,
                                                              t2t_params=t2t_params,
                                                              pool_method=pool_method,
                                                              memory=self.memory,
                                                              max_mem_len=max_mem_len,
                                                              compression_factor=compression_factor,
                                                              mem_pool=mem_pool,
                                                              store_lf=store_lf,
                                                              align_cache=align_cache,
                                                              sub_token_align=sub_token_align,
                                                              sub_factor=sub_factor,
                                                              cross_att=cross_att,
                                                              time_att=time_att,
                                                              time_deco=time_deco,
                                                              temp_focal=temp_focal,
                                                              cs_win=cs_win,
                                                              mem_att=mem_att,
                                                              cs_focal=cs_focal,
                                                              cs_focal_v2=cs_focal_v2,
                                                              cs_win_strip=1), )
                        else:
                            # 使用cs win主干
                            blocks.append(
                                Decoupled3DFocalTransformerBlock(dim=hidden,
                                                                 num_heads=num_heads[i],
                                                                 window_size=window_size[i],
                                                                 focal_level=focal_levels[i],
                                                                 focal_window=focal_windows[i],
                                                                 n_vecs=n_vecs,
                                                                 t2t_params=t2t_params,
                                                                 memory=self.memory,
                                                                 max_mem_len=max_mem_len,
                                                                 compression_factor=compression_factor,
                                                                 mem_pool=mem_pool,
                                                                 store_lf=store_lf,
                                                                 align_cache=align_cache,
                                                                 sub_token_align=sub_token_align,
                                                                 sub_factor=sub_factor,
                                                                 cross_att=cross_att,
                                                                 time_att=time_att,
                                                                 time_deco=time_deco,
                                                                 temp_focal=temp_focal,
                                                                 cs_win=cs_win,
                                                                 mem_att=mem_att,
                                                                 cs_focal=cs_focal,
                                                                 cs_focal_v2=cs_focal_v2,
                                                                 cs_win_strip=sw_list[i],
                                                                 mix_f3n=mix_f3n,
                                                                 conv_path=conv_path,
                                                                 cs_sw=cs_sw,
                                                                 pool_strip=pool_strip,
                                                                 pool_sw=pool_sw), )
                    else:
                        # 奇数层没有记忆
                        if not cs_trans:
                            blocks.append(
                                TemporalFocalTransformerBlock(dim=hidden,
                                                              num_heads=num_heads[i],
                                                              window_size=window_size[i],
                                                              focal_level=focal_levels[i],
                                                              focal_window=focal_windows[i],
                                                              n_vecs=n_vecs,
                                                              t2t_params=t2t_params,
                                                              pool_method=pool_method,
                                                              memory=False))
                        else:
                            # 使用cs win主干
                            blocks.append(
                                Decoupled3DFocalTransformerBlock(dim=hidden,
                                                                 num_heads=num_heads[i],
                                                                 window_size=window_size[i],
                                                                 focal_level=focal_levels[i],
                                                                 focal_window=focal_windows[i],
                                                                 n_vecs=n_vecs,
                                                                 t2t_params=t2t_params,
                                                                 memory=False,
                                                                 max_mem_len=max_mem_len,
                                                                 compression_factor=compression_factor,
                                                                 mem_pool=mem_pool,
                                                                 store_lf=store_lf,
                                                                 align_cache=align_cache,
                                                                 sub_token_align=sub_token_align,
                                                                 sub_factor=sub_factor,
                                                                 cross_att=cross_att,
                                                                 time_att=time_att,
                                                                 time_deco=time_deco,
                                                                 temp_focal=temp_focal,
                                                                 cs_win=cs_win,
                                                                 mem_att=mem_att,
                                                                 cs_focal=cs_focal,
                                                                 cs_focal_v2=cs_focal_v2,
                                                                 cs_win_strip=sw_list[i],
                                                                 mix_f3n=mix_f3n,
                                                                 conv_path=conv_path,
                                                                 cs_sw=cs_sw,
                                                                 pool_strip=pool_strip,
                                                                 pool_sw=pool_sw), )
                elif last_memory:
                    # 只有最后一层有记忆力
                    if (i + 1) == depths:
                        # 最后一层有记忆力
                        if not cs_trans:
                            blocks.append(
                                TemporalFocalTransformerBlock(dim=hidden,
                                                              num_heads=num_heads[i],
                                                              window_size=window_size[i],
                                                              focal_level=focal_levels[i],
                                                              focal_window=focal_windows[i],
                                                              n_vecs=n_vecs,
                                                              t2t_params=t2t_params,
                                                              pool_method=pool_method,
                                                              memory=self.memory,
                                                              max_mem_len=max_mem_len,
                                                              compression_factor=compression_factor,
                                                              mem_pool=mem_pool,
                                                              store_lf=store_lf,
                                                              align_cache=align_cache,
                                                              sub_token_align=sub_token_align,
                                                              sub_factor=sub_factor,
                                                              cross_att=cross_att,
                                                              time_att=time_att,
                                                              time_deco=time_deco,
                                                              temp_focal=temp_focal,
                                                              cs_win=cs_win,
                                                              mem_att=mem_att,
                                                              cs_focal=cs_focal,
                                                              cs_focal_v2=cs_focal_v2,
                                                              cs_win_strip=1), )
                        else:
                            # 使用cs win主干
                            blocks.append(
                                Decoupled3DFocalTransformerBlock(dim=hidden,
                                                                 num_heads=num_heads[i],
                                                                 window_size=window_size[i],
                                                                 focal_level=focal_levels[i],
                                                                 focal_window=focal_windows[i],
                                                                 n_vecs=n_vecs,
                                                                 t2t_params=t2t_params,
                                                                 memory=self.memory,
                                                                 max_mem_len=max_mem_len,
                                                                 compression_factor=compression_factor,
                                                                 mem_pool=mem_pool,
                                                                 store_lf=store_lf,
                                                                 align_cache=align_cache,
                                                                 sub_token_align=sub_token_align,
                                                                 sub_factor=sub_factor,
                                                                 cross_att=cross_att,
                                                                 time_att=time_att,
                                                                 time_deco=time_deco,
                                                                 temp_focal=temp_focal,
                                                                 cs_win=cs_win,
                                                                 mem_att=mem_att,
                                                                 cs_focal=cs_focal,
                                                                 cs_focal_v2=cs_focal_v2,
                                                                 cs_win_strip=sw_list[i],
                                                                 mix_f3n=mix_f3n,
                                                                 conv_path=conv_path,
                                                                 cs_sw=cs_sw,
                                                                 pool_strip=pool_strip,
                                                                 pool_sw=pool_sw), )
                    else:
                        # 前面的层没有记忆
                        if not cs_trans:
                            blocks.append(
                                TemporalFocalTransformerBlock(dim=hidden,
                                                              num_heads=num_heads[i],
                                                              window_size=window_size[i],
                                                              focal_level=focal_levels[i],
                                                              focal_window=focal_windows[i],
                                                              n_vecs=n_vecs,
                                                              t2t_params=t2t_params,
                                                              pool_method=pool_method,
                                                              memory=False))
                        else:
                            # 使用cs win主干
                            # 虽然没有记忆，但是self attention也需要用到cswin的一系列参数！
                            blocks.append(
                                Decoupled3DFocalTransformerBlock(dim=hidden,
                                                                 num_heads=num_heads[i],
                                                                 window_size=window_size[i],
                                                                 focal_level=focal_levels[i],
                                                                 focal_window=focal_windows[i],
                                                                 n_vecs=n_vecs,
                                                                 t2t_params=t2t_params,
                                                                 memory=False,
                                                                 max_mem_len=max_mem_len,
                                                                 compression_factor=compression_factor,
                                                                 mem_pool=mem_pool,
                                                                 store_lf=store_lf,
                                                                 align_cache=align_cache,
                                                                 sub_token_align=sub_token_align,
                                                                 sub_factor=sub_factor,
                                                                 cross_att=cross_att,
                                                                 time_att=time_att,
                                                                 time_deco=time_deco,
                                                                 temp_focal=temp_focal,
                                                                 cs_win=cs_win,
                                                                 mem_att=mem_att,
                                                                 cs_focal=cs_focal,
                                                                 cs_focal_v2=cs_focal_v2,
                                                                 cs_win_strip=sw_list[i],
                                                                 mix_f3n=mix_f3n,
                                                                 conv_path=conv_path,
                                                                 cs_sw=cs_sw,
                                                                 pool_strip=pool_strip,
                                                                 pool_sw=pool_sw), )

            self.transformer = nn.Sequential(*blocks)
        else:
            # 根据token聚合指数修改temporal focal transformer
            for i in range(depths):
                blocks.append(
                    TemporalFocalTransformerBlock(dim=hidden,
                                                  num_heads=num_heads[i],
                                                  window_size=window_size[i],
                                                  focal_level=focal_levels[i],
                                                  focal_window=focal_windows[i],
                                                  n_vecs=self.keeped_patches,
                                                  t2t_params=t2t_params,
                                                  pool_method=pool_method,
                                                  token_fusion=True))
            self.transformer = nn.Sequential(*blocks)

        if init_weights:
            self.init_weights()
            # Need to initial the weights of MSDeformAttn specifically
            for m in self.modules():
                if isinstance(m, SecondOrderDeformableAlignment):
                    m.init_offset()

        # flow completion network
        self.update_MFN = MaskFlowNetS()

    def forward_bidirect_flow(self, masked_local_frames):
        b, l_t, c, h, w = masked_local_frames.size()

        # compute forward and backward flows of masked frames
        masked_local_frames = F.interpolate(masked_local_frames.view(
            -1, c, h, w),
                                            scale_factor=1 / 4,
                                            mode='bilinear',
                                            align_corners=True,
                                            recompute_scale_factor=True)
        masked_local_frames = masked_local_frames.view(b, l_t, c, h // 4,
                                                       w // 4)
        mlf_1 = masked_local_frames[:, :-1, :, :, :].reshape(
            -1, c, h // 4, w // 4)
        mlf_2 = masked_local_frames[:, 1:, :, :, :].reshape(
            -1, c, h // 4, w // 4)
        pred_flows_forward = self.update_MFN(mlf_1, mlf_2)
        pred_flows_backward = self.update_MFN(mlf_2, mlf_1)

        pred_flows_forward = pred_flows_forward.view(b, l_t - 1, 2, h // 4,
                                                     w // 4)
        pred_flows_backward = pred_flows_backward.view(b, l_t - 1, 2, h // 4,
                                                       w // 4)

        return pred_flows_forward, pred_flows_backward

    def forward(self, masked_frames, num_local_frames):
        l_t = num_local_frames
        b, t, ori_c, ori_h, ori_w = masked_frames.size()

        # normalization before feeding into the flow completion module
        masked_local_frames = (masked_frames[:, :l_t, ...] + 1) / 2
        pred_flows = self.forward_bidirect_flow(masked_local_frames)

        # extracting features and performing the feature propagation on local features
        enc_feat = self.encoder(masked_frames.view(b * t, ori_c, ori_h, ori_w))
        _, c, h, w = enc_feat.size()
        fold_output_size = (h, w)
        local_feat = enc_feat.view(b, t, c, h, w)[:, :l_t, ...]
        ref_feat = enc_feat.view(b, t, c, h, w)[:, l_t:, ...]

        # local_feat = self.feat_prop_module(local_feat, pred_flows[0],
        #                                    pred_flows[1])       # pred_flows[0]的位置应当输入后向光流，pred_flows[1]的位置应当输入前向光流
        # 可是self.forward_bidirect_flow返回来0是前向光流，1是反向光流。。。
        # 更正前后向光流：
        local_feat = self.feat_prop_module(local_feat, pred_flows[1],
                                           pred_flows[0])

        enc_feat = torch.cat((local_feat, ref_feat), dim=1)

        # content hallucination through stacking multiple temporal focal transformer blocks
        if not self.flow_guide:
            trans_feat = self.ss(enc_feat.view(-1, c, h, w), b, fold_output_size)   # [B, t, f_h, f_w, hidden]
        else:
            trans_feat = self.ss(enc_feat, b, c, fold_output_size,
                                 pred_flows[0], pred_flows[1], l_t)

        if self.token_fusion:
            # 融合相似度高的token来降低计算复杂度
            tokens = trans_feat
            if not self.token_fusion_simple:
                # 每个trans block前面一个token缩减，后面一个token复原
                for i, blk in enumerate(self.transformer):
                    # tokens, fold_output_size = blk([tokens, fold_output_size])
                    if (i + 1) in self.slim_index:
                        tsm = self.tsm[self.slim_index.index(i + 1)]
                        if self.fusion_skip_connect:
                            # 存储聚合前的tokens, 用于渐进式地跳连融合
                            tokens_prior = tokens
                        # token 聚合
                        tokens = tsm(tokens)
                        # trans block
                        tokens, fold_output_size = blk([tokens, fold_output_size])
                        if not self.fusion_skip_connect:
                            # token 恢复
                            tokens = self.rtsm[i](tokens)
                        else:
                            # 融合缩减前的 tokens_prior 进行 token 恢复
                            tokens = self.rtsm[i](tokens, tokens_prior)
            else:
                # 所有的trans block共用一个token缩减和一个token复原
                tsm = self.tsm[self.slim_index.index(1)]
                # token 聚合
                tokens = tsm(tokens)
                # trans block
                tokens, fold_output_size = self.transformer([tokens, fold_output_size])   # temporal focal trans block
                if not self.fusion_skip_connect:
                    # token 恢复
                    tokens = self.rtsm[0](tokens)
                else:
                    # 融合缩减前的 trans feat 进行 token 恢复
                    tokens = self.rtsm[0](tokens, trans_feat)

            trans_feat = tokens
            trans_feat = self.sc(trans_feat, t, fold_output_size)
        else:
            # if not self.memory:
            #     trans_feat = self.transformer([trans_feat, fold_output_size])   # default temporal focal trans block
            # else:
            #     trans_feat = self.transformer([trans_feat, fold_output_size, l_t])  # add local frame nums as input
            trans_feat = self.transformer([trans_feat, fold_output_size, l_t])  # 比默认行为多传一个lt

            # 软组合
            trans_feat = self.sc(trans_feat[0], t, fold_output_size)

        trans_feat = trans_feat.view(b, t, -1, h, w)
        enc_feat = enc_feat + trans_feat    # 残差链接

        # decode frames from features
        output = self.decoder(enc_feat.view(b * t, c, h, w))
        output = torch.tanh(output)
        return output, pred_flows


# ######################################################################
#  Discriminator for Temporal Patch GAN
# ######################################################################


class Discriminator(BaseNetwork):
    def __init__(self,
                 in_channels=3,
                 use_sigmoid=False,
                 use_spectral_norm=True,
                 init_weights=True):
        super(Discriminator, self).__init__()
        self.use_sigmoid = use_sigmoid
        nf = 32

        self.conv = nn.Sequential(
            spectral_norm(
                nn.Conv3d(in_channels=in_channels,
                          out_channels=nf * 1,
                          kernel_size=(3, 5, 5),
                          stride=(1, 2, 2),
                          padding=1,
                          bias=not use_spectral_norm), use_spectral_norm),
            # nn.InstanceNorm2d(64, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(
                nn.Conv3d(nf * 1,
                          nf * 2,
                          kernel_size=(3, 5, 5),
                          stride=(1, 2, 2),
                          padding=(1, 2, 2),
                          bias=not use_spectral_norm), use_spectral_norm),
            # nn.InstanceNorm2d(128, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(
                nn.Conv3d(nf * 2,
                          nf * 4,
                          kernel_size=(3, 5, 5),
                          stride=(1, 2, 2),
                          padding=(1, 2, 2),
                          bias=not use_spectral_norm), use_spectral_norm),
            # nn.InstanceNorm2d(256, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(
                nn.Conv3d(nf * 4,
                          nf * 4,
                          kernel_size=(3, 5, 5),
                          stride=(1, 2, 2),
                          padding=(1, 2, 2),
                          bias=not use_spectral_norm), use_spectral_norm),
            # nn.InstanceNorm2d(256, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(
                nn.Conv3d(nf * 4,
                          nf * 4,
                          kernel_size=(3, 5, 5),
                          stride=(1, 2, 2),
                          padding=(1, 2, 2),
                          bias=not use_spectral_norm), use_spectral_norm),
            # nn.InstanceNorm2d(256, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(nf * 4,
                      nf * 4,
                      kernel_size=(3, 5, 5),
                      stride=(1, 2, 2),
                      padding=(1, 2, 2)))

        if init_weights:
            self.init_weights()

    def forward(self, xs):
        # T, C, H, W = xs.shape (old)
        # B, T, C, H, W (new)
        xs_t = torch.transpose(xs, 1, 2)
        feat = self.conv(xs_t)
        if self.use_sigmoid:
            feat = torch.sigmoid(feat)
        out = torch.transpose(feat, 1, 2)  # B, T, C, H, W
        return out


def spectral_norm(module, mode=True):
    if mode:
        return _spectral_norm(module)
    return module
