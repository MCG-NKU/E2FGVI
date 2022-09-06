# -*- coding: utf-8 -*-
import cv2
import numpy as np
import importlib
import sys
import os
import time
import random
import argparse
import line_profiler
from PIL import Image

import torch
from torch.utils.data import DataLoader

from core.dataset import TestDataset
from core.metrics import calc_psnr_and_ssim, calculate_i3d_activations, calculate_vfid, init_i3d_model

# global variables
w, h = 432, 240     # default acc. test setting in e2fgvi for davis dataset
# w, h = 864, 480     # davis res 480x854
# w, h = 320, 240     # pal test
ref_length = 10     # non-local frames的步幅间隔，此处为每10帧取1帧NLF
neighbor_stride = 5     # local frames的窗口大小，加上自身则窗口大小为6
default_fps = 24


# sample reference frames from the whole video
def get_ref_index(neighbor_ids, length):
    ref_index = []
    for i in range(0, length, ref_length):
        if i not in neighbor_ids:
            ref_index.append(i)
    return ref_index


# sample reference frames from the whole video with mem support
# 允许相同的局部帧和非局部帧id，保证时间维度的一致性，但是引入了冗余计算？
def get_ref_index_mem(length):
    ref_index = []
    for i in range(0, length, ref_length):
        ref_index.append(i)
    return ref_index


# sample reference frames from the remain frames with random behavior like trainning
def get_ref_index_mem_random(neighbor_ids, video_length, num_ref_frame=3):
    complete_idx_set = list(range(video_length))
    remain_idx = list(set(complete_idx_set) - set(neighbor_ids))
    ref_index = sorted(random.sample(remain_idx, num_ref_frame))
    return ref_index


def main_worker(args):
    args.size = (w, h)
    # set up datasets and data loader
    # default result
    # assert (args.dataset == 'davis') or args.dataset == 'youtube-vos', \
    #     f"{args.dataset} dataset is not supported"
    test_dataset = TestDataset(args)

    test_loader = DataLoader(test_dataset,
                             batch_size=1,
                             shuffle=False,
                             num_workers=args.num_workers)

    # set up models
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = importlib.import_module('model.' + args.model)
    model = net.InpaintGenerator().to(device)
    if args.ckpt is not None:
        data = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(data)
        print(f'Loading from: {args.ckpt}')
    model.eval()

    total_frame_psnr = []
    total_frame_ssim = []

    output_i3d_activations = []
    real_i3d_activations = []

    print('Start evaluation...')

    if args.timing:
        time_all = 0
        len_all = 0

    # create results directory
    result_path = os.path.join('results', f'{args.model}_{args.dataset}')
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    eval_summary = open(
        os.path.join(result_path, f"{args.model}_{args.dataset}_metrics.txt"),
        "w")

    i3d_model = init_i3d_model()

    for index, items in enumerate(test_loader):

        if args.memory:
            # 进入新的视频时清空记忆缓存
            for blk in model.transformer:
                try:
                    blk.attn.m_k = []
                    blk.attn.m_v = []
                except:
                    pass

        frames, masks, video_name, frames_PIL = items

        video_length = frames.size(1)
        frames, masks = frames.to(device), masks.to(device)
        ori_frames = frames_PIL     # 原始帧，可视为真值
        ori_frames = [
            ori_frames[i].squeeze().cpu().numpy() for i in range(video_length)
        ]
        comp_frames = [None] * video_length     # 补全帧

        if args.timing:
            len_all += video_length

        # complete holes by our model
        # 当这个循环走完的时候，一段视频已经被补全了
        for f in range(0, video_length, neighbor_stride):
            if not args.memory:
                # default id with different T
                neighbor_ids = [
                    i for i in range(max(0, f - neighbor_stride),
                                     min(video_length, f + neighbor_stride + 1))
                ]   # neighbor_ids即为Local Frames, 局部帧
            else:
                if args.same_memory:
                    # 尽可能与e2fgvi的原测试逻辑一致
                    # 输入的时间维度T保持一致
                    if (f - neighbor_stride > 0) and (f + neighbor_stride + 1 < video_length):
                        # 视频首尾均不会越界，不需要补充额外帧
                        neighbor_ids = [
                            i for i in range(max(0, f - neighbor_stride),
                                             min(video_length, f + neighbor_stride + 1))
                        ]  # neighbor_ids即为Local Frames, 局部帧
                    else:
                        # 视频越界，补充额外帧保证记忆缓存的时间通道维度一致，后面也可以尝试放到trans里直接复制特征的时间维度
                        neighbor_ids = [
                            i for i in range(max(0, f - neighbor_stride),
                                             min(video_length, f + neighbor_stride + 1))
                        ]  # neighbor_ids即为Local Frames, 局部帧
                        repeat_num = (neighbor_stride * 2 + 1) - len(neighbor_ids)
                        for ii in range(0, repeat_num):
                            # 复制最后一帧
                            neighbor_ids.append(neighbor_ids[-1])

                else:
                    # 与记忆力模型的训练逻辑一致
                    if video_length < (f + neighbor_stride):
                        neighbor_ids = [
                            i for i in range(f, video_length)
                        ]  # 时间上不重叠的窗口，每个局部帧只会被计算一次，视频尾部可能不足5帧局部帧，复制最后一帧补全数量
                        for repeat_idx in range(0, neighbor_stride - len(neighbor_ids)):
                            neighbor_ids.append(neighbor_ids[-1])
                    else:
                        neighbor_ids = [
                            i for i in range(f, f + neighbor_stride)
                        ]  # 时间上不重叠的窗口，每个局部帧只会被计算一次

            if not args.memory:
                # default test set, 局部帧与非局部帧不会输入同样id的帧
                ref_ids = get_ref_index(neighbor_ids, video_length)  # ref_ids即为Non-Local Frames, 非局部帧

                selected_imgs = frames[:1, neighbor_ids + ref_ids, :, :, :]
                selected_masks = masks[:1, neighbor_ids + ref_ids, :, :, :]
            else:
                # 为了保证时间维度一致, 允许输入相同id的帧
                if args.same_memory:
                    ref_ids = get_ref_index_mem(video_length)  # ref_ids即为Non-Local Frames, 非局部帧
                else:
                    ref_ids = get_ref_index_mem_random(neighbor_ids, video_length, num_ref_frame=3)  # 与序列训练同样的非局部帧输入逻辑

                selected_imgs_lf = frames[:1, neighbor_ids, :, :, :]
                selected_imgs_nlf = frames[:1, ref_ids, :, :, :]
                selected_imgs = torch.cat((selected_imgs_lf, selected_imgs_nlf), dim=1)
                selected_masks_lf = masks[:1, neighbor_ids, :, :, :]
                selected_masks_nlf = masks[:1, ref_ids, :, :, :]
                selected_masks = torch.cat((selected_masks_lf, selected_masks_nlf), dim=1)

            with torch.no_grad():
                masked_frames = selected_imgs * (1 - selected_masks)

                if args.timing:
                    torch.cuda.synchronize()
                    time_start = time.time()
                pred_img, _ = model(masked_frames, len(neighbor_ids))   # forward里会输入局部帧数量来对两种数据分开处理
                if args.timing:
                    torch.cuda.synchronize()
                    time_end = time.time()
                    time_sum = time_end - time_start
                    time_all += time_sum
                    # print('Run Time: '
                    #       f'{time_sum/len(neighbor_ids)}')

                pred_img = (pred_img + 1) / 2
                pred_img = pred_img.cpu().permute(0, 2, 3, 1).numpy() * 255
                binary_masks = masks[0, neighbor_ids, :, :, :].cpu().permute(
                    0, 2, 3, 1).numpy().astype(np.uint8)
                for i in range(len(neighbor_ids)):
                    idx = neighbor_ids[i]
                    img = np.array(pred_img[i]).astype(np.uint8) * binary_masks[i] \
                        + ori_frames[idx] * (1 - binary_masks[i])

                    if not args.good_fusion:
                        if comp_frames[idx] is None:
                            # 如果第一次补全Local Frame中的某帧，直接记录到补全帧list (comp_frames) 里
                            comp_frames[idx] = img

                        else:   # default 融合策略：不合理，neighbor_stride倍数的LF的中间帧权重为0.25，应当为0.5
                            # 如果不是第一次补全Local Frame中的某帧，即该帧已补全过，则把此前结果与当前帧结果简单加和平均
                            comp_frames[idx] = comp_frames[idx].astype(
                                np.float32) * 0.5 + img.astype(np.float32) * 0.5
                        ########################################################################################
                    else:
                        if comp_frames[idx] is None:
                            # 如果第一次补全Local Frame中的某帧，直接记录到补全帧list (comp_frames) 里
                            comp_frames[idx] = img

                        elif idx == (neighbor_ids[0] + neighbor_ids[-1])/2:
                            # 如果是中间帧，记录下来
                            medium_frame = img
                        elif (idx != 0) & (idx == neighbor_ids[0]):
                            # 如果是第三次出现，加权平均
                            comp_frames[idx] = comp_frames[idx].astype(
                                np.float32) * 0.25 + medium_frame.astype(np.float32) * 0.5 + img.astype(np.float32) * 0.25
                        else:
                            # 如果是不是中间帧，权重为0.5
                            comp_frames[idx] = comp_frames[idx].astype(
                                np.float32) * 0.5 + img.astype(np.float32) * 0.5
                        ########################################################################################

        if args.memory_double:
            for f in range(neighbor_stride//2, video_length, neighbor_stride):
                if not args.memory:
                    # default id with different T
                    neighbor_ids = [
                        i for i in range(max(neighbor_stride//2, f - neighbor_stride),
                                         min(video_length, f + neighbor_stride + 1))
                    ]  # neighbor_ids即为Local Frames, 局部帧
                else:
                    if args.same_memory:
                        # 尽可能与e2fgvi的原测试逻辑一致
                        # 输入的时间维度T保持一致
                        if (f - neighbor_stride > 0) and (f + neighbor_stride + 1 < video_length):
                            # 视频首尾均不会越界，不需要补充额外帧
                            neighbor_ids = [
                                i for i in range(max(0, f - neighbor_stride),
                                                 min(video_length, f + neighbor_stride + 1))
                            ]   # neighbor_ids即为Local Frames, 局部帧
                        else:
                            # 视频越界，补充额外帧保证记忆缓存的时间通道维度一致，后面也可以尝试放到trans里直接复制特征的时间维度
                            neighbor_ids = [
                                i for i in range(max(0, f - neighbor_stride),
                                                 min(video_length, f + neighbor_stride + 1))
                            ]  # neighbor_ids即为Local Frames, 局部帧
                            repeat_num = (neighbor_stride * 2 + 1) - len(neighbor_ids)
                            for ii in range(0, repeat_num):
                                # 复制最后一帧
                                neighbor_ids.append(neighbor_ids[-1])

                    else:
                        # 与记忆力模型的训练逻辑一致
                        if video_length < (f + neighbor_stride):
                            neighbor_ids = [
                                i for i in range(f, video_length)
                            ]  # 时间上不重叠的窗口，每个局部帧只会被计算一次，视频尾部可能不足5帧局部帧，复制最后一帧补全数量
                            for repeat_idx in range(0, neighbor_stride - len(neighbor_ids)):
                                neighbor_ids.append(neighbor_ids[-1])
                        else:
                            neighbor_ids = [
                                i for i in range(f, f + neighbor_stride)
                            ]  # 时间上不重叠的窗口，每个局部帧只会被计算一次

                if not args.memory:
                    # default test set, 局部帧与非局部帧不会输入同样id的帧
                    ref_ids = get_ref_index(neighbor_ids, video_length)  # ref_ids即为Non-Local Frames, 非局部帧

                    selected_imgs = frames[:1, neighbor_ids + ref_ids, :, :, :]
                    selected_masks = masks[:1, neighbor_ids + ref_ids, :, :, :]
                else:
                    # 为了保证时间维度一致, 允许输入相同id的帧
                    if args.same_memory:
                        ref_ids = get_ref_index_mem(video_length)  # ref_ids即为Non-Local Frames, 非局部帧
                    else:
                        ref_ids = get_ref_index_mem_random(neighbor_ids, video_length, num_ref_frame=3)  # 与序列训练同样的非局部帧输入逻辑

                    selected_imgs_lf = frames[:1, neighbor_ids, :, :, :]
                    selected_imgs_nlf = frames[:1, ref_ids, :, :, :]
                    selected_imgs = torch.cat((selected_imgs_lf, selected_imgs_nlf), dim=1)
                    selected_masks_lf = masks[:1, neighbor_ids, :, :, :]
                    selected_masks_nlf = masks[:1, ref_ids, :, :, :]
                    selected_masks = torch.cat((selected_masks_lf, selected_masks_nlf), dim=1)

                with torch.no_grad():
                    masked_frames = selected_imgs * (1 - selected_masks)

                    if args.timing:
                        torch.cuda.synchronize()
                        time_start = time.time()
                    pred_img, _ = model(masked_frames, len(neighbor_ids))  # forward里会输入局部帧数量来对两种数据分开处理
                    if args.timing:
                        torch.cuda.synchronize()
                        time_end = time.time()
                        time_sum = time_end - time_start
                        time_all += time_sum
                        # print('Run Time: '
                        #       f'{time_sum/len(neighbor_ids)}')

                    pred_img = (pred_img + 1) / 2
                    pred_img = pred_img.cpu().permute(0, 2, 3, 1).numpy() * 255
                    binary_masks = masks[0, neighbor_ids, :, :, :].cpu().permute(
                        0, 2, 3, 1).numpy().astype(np.uint8)
                    for i in range(len(neighbor_ids)):
                        idx = neighbor_ids[i]
                        img = np.array(pred_img[i]).astype(np.uint8) * binary_masks[i] \
                              + ori_frames[idx] * (1 - binary_masks[i])

                        if not args.good_fusion:
                            if comp_frames[idx] is None:
                                # 如果第一次补全Local Frame中的某帧，直接记录到补全帧list (comp_frames) 里
                                comp_frames[idx] = img

                            else:  # default 融合策略：不合理，neighbor_stride倍数的LF的中间帧权重为0.25，应当为0.5
                                # 如果不是第一次补全Local Frame中的某帧，即该帧已补全过，则把此前结果与当前帧结果简单加和平均
                                comp_frames[idx] = comp_frames[idx].astype(
                                    np.float32) * 0.5 + img.astype(np.float32) * 0.5
                            ########################################################################################
                        else:
                            if comp_frames[idx] is None:
                                # 如果第一次补全Local Frame中的某帧，直接记录到补全帧list (comp_frames) 里
                                comp_frames[idx] = img

                            elif idx == (neighbor_ids[0] + neighbor_ids[-1]) / 2:
                                # 如果是中间帧，记录下来
                                medium_frame = img
                            elif (idx != 0) & (idx == neighbor_ids[0]):
                                # 如果是第三次出现，加权平均
                                comp_frames[idx] = comp_frames[idx].astype(
                                    np.float32) * 0.25 + medium_frame.astype(np.float32) * 0.5 + img.astype(
                                    np.float32) * 0.25
                            else:
                                # 如果是不是中间帧，权重为0.5
                                comp_frames[idx] = comp_frames[idx].astype(
                                    np.float32) * 0.5 + img.astype(np.float32) * 0.5
                            ########################################################################################

        # calculate metrics
        cur_video_psnr = []
        cur_video_ssim = []
        comp_PIL = []  # to calculate VFID
        frames_PIL = []
        for ori, comp in zip(ori_frames, comp_frames):
            psnr, ssim = calc_psnr_and_ssim(ori, comp)

            cur_video_psnr.append(psnr)
            cur_video_ssim.append(ssim)

            total_frame_psnr.append(psnr)
            total_frame_ssim.append(ssim)

            frames_PIL.append(Image.fromarray(ori.astype(np.uint8)))
            comp_PIL.append(Image.fromarray(comp.astype(np.uint8)))
        cur_psnr = sum(cur_video_psnr) / len(cur_video_psnr)
        cur_ssim = sum(cur_video_ssim) / len(cur_video_ssim)

        # saving i3d activations
        frames_i3d, comp_i3d = calculate_i3d_activations(frames_PIL,
                                                         comp_PIL,
                                                         i3d_model,
                                                         device=device)
        real_i3d_activations.append(frames_i3d)
        output_i3d_activations.append(comp_i3d)

        print(
            f'[{index+1:3}/{len(test_loader)}] Name: {str(video_name):25} | PSNR/SSIM: {cur_psnr:.4f}/{cur_ssim:.4f}'
        )
        eval_summary.write(
            f'[{index+1:3}/{len(test_loader)}] Name: {str(video_name):25} | PSNR/SSIM: {cur_psnr:.4f}/{cur_ssim:.4f}\n'
        )

        if args.timing:
            print('Average run time: (%f) per frame' % (time_all/len_all))

        # saving images for evaluating warpping errors
        if args.save_results:
            save_frame_path = os.path.join(result_path, video_name[0])
            os.makedirs(save_frame_path, exist_ok=False)

            for i, frame in enumerate(comp_frames):
                cv2.imwrite(
                    os.path.join(save_frame_path,
                                 str(i).zfill(5) + '.png'),
                    cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_RGB2BGR))

    avg_frame_psnr = sum(total_frame_psnr) / len(total_frame_psnr)
    avg_frame_ssim = sum(total_frame_ssim) / len(total_frame_ssim)

    fid_score = calculate_vfid(real_i3d_activations, output_i3d_activations)
    print('Finish evaluation... Average Frame PSNR/SSIM/VFID: '
          f'{avg_frame_psnr:.2f}/{avg_frame_ssim:.4f}/{fid_score:.3f}')
    eval_summary.write(
        'Finish evaluation... Average Frame PSNR/SSIM/VFID: '
        f'{avg_frame_psnr:.2f}/{avg_frame_ssim:.4f}/{fid_score:.3f}')
    eval_summary.close()

    if args.timing:
        print('All average forward run time: (%f) per frame' % (time_all / len_all))

    return len(total_frame_psnr)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='E2FGVI')
    parser.add_argument('--dataset',
                        choices=['davis', 'youtube-vos', 'pal'],
                        type=str)
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--model', choices=['e2fgvi', 'e2fgvi_hq', 'e2fgvi_hq-lite', 'lite-MFN', 'large-MFN'], type=str)
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--save_results', action='store_true', default=False)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--timing', action='store_true', default=False)
    parser.add_argument('--profile', action='store_true', default=False)
    parser.add_argument('--good_fusion', action='store_true', default=False, help='using my fusion strategy')
    parser.add_argument('--memory', action='store_true', default=False, help='test with memory ability')
    parser.add_argument('--same_memory', action='store_true', default=False,
                        help='test with memory ability in E2FGVI style, not work with --memory_double')
    # TODO: 这里的memory double逻辑还可以把前面两帧也再次估计一遍提升精度
    parser.add_argument('--memory_double', action='store_true', default=False, help='test with memory ability twice')
    args = parser.parse_args()

    if args.profile:
        profile = line_profiler.LineProfiler(main_worker)  # 把函数传递到性能分析器
        profile.enable()  # 开始分析

    # if args.timing:
    #     torch.cuda.synchronize()
    #     time_start = time.time()

    frame_num = main_worker(args)

    # if args.timing:
    #     torch.cuda.synchronize()
    #     time_end = time.time()
    #     time_sum = time_end - time_start
    #     print('Finish evaluation... Average Run Time: '
    #           f'{time_sum/frame_num}')

    if args.profile:
        profile.disable()  # 停止分析
        profile.print_stats(sys.stdout)  # 打印出性能分析结果
