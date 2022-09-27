import os
import json
import random
import math

import cv2
from PIL import Image
import numpy as np

import torch
import torchvision.transforms as transforms

from core.utils import (TrainZipReader, TestZipReader,
                        create_random_shape_with_random_motion, Stack,
                        ToTorchFormatTensor, GroupRandomHorizontalFlip,
                        create_random_shape_with_random_motion_seq)


class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, args: dict, debug=False):
        self.args = args
        self.num_local_frames = args['num_local_frames']
        self.num_ref_frames = args['num_ref_frames']
        self.size = self.w, self.h = (args['w'], args['h'])

        json_path = os.path.join(args['data_root'], args['name'], 'train.json')
        with open(json_path, 'r') as f:
            self.video_dict = json.load(f)
        self.video_names = list(self.video_dict.keys())
        if debug:
            self.video_names = self.video_names[:100]

        self._to_tensors = transforms.Compose([
            Stack(),
            ToTorchFormatTensor(),
        ])

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, index):
        item = self.load_item(index)
        return item

    def _sample_index(self, length, sample_length, num_ref_frame=3):
        complete_idx_set = list(range(length))
        pivot = random.randint(0, length - sample_length)
        local_idx = complete_idx_set[pivot:pivot + sample_length]
        remain_idx = list(set(complete_idx_set) - set(local_idx))
        ref_index = sorted(random.sample(remain_idx, num_ref_frame))

        return local_idx + ref_index

    def load_item(self, index):
        video_name = self.video_names[index]
        # create masks
        all_masks = create_random_shape_with_random_motion(
            self.video_dict[video_name], imageHeight=self.h, imageWidth=self.w)

        # create sample index
        selected_index = self._sample_index(self.video_dict[video_name],
                                            self.num_local_frames,
                                            self.num_ref_frames)

        # read video frames
        frames = []
        masks = []
        for idx in selected_index:
            video_path = os.path.join(self.args['data_root'],
                                      self.args['name'], 'JPEGImages',
                                      f'{video_name}.zip')
            img = TrainZipReader.imread(video_path, idx).convert('RGB')
            img = img.resize(self.size)
            frames.append(img)
            masks.append(all_masks[idx])

        # normalizate, to tensors
        frames = GroupRandomHorizontalFlip()(frames)
        frame_tensors = self._to_tensors(frames) * 2.0 - 1.0
        mask_tensors = self._to_tensors(masks)
        return frame_tensors, mask_tensors, video_name


class TrainDataset_Mem(torch.utils.data.Dataset):
    """
    Sequence Video Train Dataloader by Hao, based on E2FGVI train loader.
    same_mask(bool): If True, use same mask until video changes to the next video.
    """
    def __init__(self, args: dict, debug=False, start=0, end=1, batch_size=1, same_mask=False):
        self.args = args
        self.num_local_frames = args['num_local_frames']
        self.num_ref_frames = args['num_ref_frames']
        self.size = self.w, self.h = (args['w'], args['h'])

        if args['name'] != 'KITTI360-EX':
            json_path = os.path.join(args['data_root'], args['name'], 'train.json')
        else:
            json_path = os.path.join(args['data_root'], 'train.json')
            self.dataset_name = 'KITTI360-EX'

        with open(json_path, 'r') as f:
            self.video_dict = json.load(f)
        self.video_names = list(self.video_dict.keys())
        if args['name'] == 'KITTI360-EX':
            # 打乱数据顺序防止过拟合
            from random import shuffle
            shuffle(self.video_names)
        if debug:
            self.video_names = self.video_names[:100]

        self._to_tensors = transforms.Compose([
            Stack(),
            ToTorchFormatTensor(),
        ])

        self.neighbor_stride = 5    # 每隔5步采样一组LF和NLF
        self.start_index = 0        # 采样的起始点
        self.start = start          # 数据集迭代起点
        self.end = len(self.video_names)              # 数据集迭代终点
        self.batch_size = batch_size    # 用于计算数据集迭代器
        self.index = 0      # 自定义视频index
        self.batch_buffer = 0   # 用于随着batch size更新视频index
        self.new_video_flag = False     # 用于判断是否到了新视频
        self.worker_group = 0   # 用于随着每组worker更新

        self.same_mask = same_mask  # 如果为True, 在切换视频前使用相同的mask，这样的行为模式
        if self.same_mask:
            self.random_dict_list = []
            self.new_mask_list = []
            for i in range(0, self.batch_size):
                # 用于存储随机mask的参数字典, 不同batch不一样
                self.random_dict_list.append(None)
                # 当设置为True时，mask会重新随机生成
                self.new_mask_list.append(False)

        # 为每个batch创建独立的video index和start_index, 以及worker_group
        self.video_index_list = []
        for i in range(0, self.batch_size):
            # 初始化video id时将不同batch的错开防止数据重复和过拟合
            self.video_index_list.append(i * len(self.video_names)//self.batch_size)

        # debug
        # self.video_index_list[0] = 58
        # self.video_index_list[1] = 1793

        self.start_index_list = []
        for i in range(0, self.batch_size):
            self.start_index_list.append(0)

        self.worker_group_list = []
        for i in range(0, self.batch_size):
            self.worker_group_list.append(0)

    def __len__(self):
        # return len(self.video_names)
        # 视频切换等操作自定义完成，iter定义为一个大数避免与自定义index冲突
        return len(self.video_names)*1000

    # 两次迭代相距5帧, 不同batch通道是不同的视频, 视频index和起始帧index在batch之间独立, 避免数据浪费
    def __getitem__(self, index):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            # single-process data loading, skip idx rearrange
            print('Warning: Only one data worker was used, the manner has not been test!')
            pass
        else:
            if self.start_index_list[self.batch_buffer] == 0 and self.worker_group_list[self.batch_buffer] == 0:
                # 新视频, 初始化start index

                if self.same_mask:
                    # 在这种行为模式下，当切换到新视频时，我们重新生成mask
                    # 只有对于第一个worker，我们希望他可以成功生成新的mask，其他的worker最好和他用一样的mask
                    if worker_info.id == 0:
                        self.random_dict_list[self.batch_buffer] = None
                        self.new_mask_list[self.batch_buffer] = True

                # 判断start index有没有超出视频的长度
                if (self.neighbor_stride * worker_info.id) <= self.video_dict[self.video_names[self.video_index_list[self.batch_buffer]]]:
                    self.start_index_list[self.batch_buffer] = self.neighbor_stride * worker_info.id
                else:
                    # debug
                    # print('WHAT???????????????????????????????????????????????????????????????')

                    # 超出则等待第一个worker超出，并且start index置为当前worker的上一个没有超出的worker的start id
                    # 等待第一个worker超出后再更新到下一个视频，防止不同worker的start id错位
                    # 判断前面的worker有没有超出，用最近的且没有超出的worker的值替换
                    out_list = []
                    final_worker_idx = 0
                    for previous_worker in range(0, worker_info.num_workers):
                        out_list.append(
                            ((self.neighbor_stride * previous_worker) <= self.video_dict[self.video_names[self.video_index_list[self.batch_buffer]]])
                        )
                    out_list.reverse()
                    final_worker_idx = worker_info.num_workers - 1 - out_list.index(True)
                    self.start_index_list[self.batch_buffer] = self.neighbor_stride * final_worker_idx

            else:
                # 不是新视频
                if self.same_mask:
                    # 在这种行为模式下，不是新视频，直接用上一次的mask参数
                    self.new_mask_list[self.batch_buffer] = False

                # 判断start index有没有超出视频的长度
                if (self.start_index_list[self.batch_buffer] + self.neighbor_stride * worker_info.num_workers) <= self.video_dict[self.video_names[self.video_index_list[self.batch_buffer]]]:
                    self.start_index_list[self.batch_buffer] += self.neighbor_stride * worker_info.num_workers
                else:
                    # 超出则切换到下一个视频，并且start index置为0(每个worker的start仍然不同)
                    # 如果第一个worker没有超出，就复制当前worker的上一个worker的start id，等待第一个worker超出后再更新到下一个视频，
                    # 防止不同worker的start id错位
                    if (self.start_index_list[self.batch_buffer] + self.neighbor_stride * (worker_info.num_workers - worker_info.id)) <= self.video_dict[self.video_names[self.video_index_list[self.batch_buffer]]]:
                        # 判断前面的worker有没有超出，用最近的且没有超出的worker的值替换
                        out_list = []
                        final_worker_idx = 0
                        for previous_worker in range(0, worker_info.num_workers):
                            out_list.append(
                                ((self.start_index_list[self.batch_buffer] + self.neighbor_stride * (worker_info.num_workers - worker_info.id + previous_worker)) <= self.video_dict[self.video_names[self.video_index_list[self.batch_buffer]]])
                            )
                        out_list.reverse()
                        final_worker_idx = worker_info.num_workers - 1 - out_list.index(True)
                        self.start_index_list[self.batch_buffer] = self.start_index_list[self.batch_buffer] + self.neighbor_stride * (worker_info.num_workers - worker_info.id + final_worker_idx)
                    else:
                        # 如果第一个worker超出了，切换到下一个视频
                        self.new_video_flag = True

                        if self.same_mask:
                            # 在这种行为模式下，当切换到新视频时，我们重新生成mask
                            # 只有对于第一个worker，我们希望他可以成功生成新的mask，其他的worker最好和他用一样的mask
                            if worker_info.id == 0:
                                self.random_dict_list[self.batch_buffer] = None
                                self.new_mask_list[self.batch_buffer] = True

                        # self.worker_group = 0
                        self.worker_group_list[self.batch_buffer] = 0
                        self.start_index_list[self.batch_buffer] = self.neighbor_stride * worker_info.id
                        # 判断视频 index有没有超出视频的个数
                        if (self.video_index_list[self.batch_buffer] + 1) < len(self.video_names):
                            self.video_index_list[self.batch_buffer] += 1
                        else:
                            # 超出则切换回第一个视频
                            self.video_index_list[self.batch_buffer] = 0

        # 根据index和start index读取帧
        self.index = self.video_index_list[self.batch_buffer]
        self.start_index = self.start_index_list[self.batch_buffer]
        # item = self.load_item_v3(index=self.index)
        item = self.load_item_v4()

        # 更新woker group的index
        self.worker_group_list[self.batch_buffer] += 1

        self.batch_buffer += 1
        if self.batch_buffer == self.batch_size:
            # 重置batch缓存
            self.batch_buffer = 0

        return item

    def _sample_index_seq(self, length, sample_length, num_ref_frame=3, pivot=0, before_nlf=False):
        """

        Args:
            length:
            sample_length:
            num_ref_frame:
            pivot:
            before_nlf: If True, the non local frames will be sampled only from previous frames, not from future.

        Returns:

        """
        complete_idx_set = list(range(length))
        local_idx = complete_idx_set[pivot:pivot + sample_length]

        # 保证最后几帧返回的局部帧数量一致，也是5帧（步数），使得batch stack的时候不会出错:
        if len(local_idx) < self.neighbor_stride:
            for i in range(0, self.neighbor_stride-len(local_idx)):
                if local_idx:
                    local_idx.append(local_idx[-1])
                else:
                    # 恰好视频长度是局部帧步幅的整数倍，取local_idx为最后一帧5次
                    local_idx.append(complete_idx_set[-1])

        if before_nlf:
            # 非局部帧只会从过去的视频帧中选取，不会使用未来的信息
            complete_idx_set = complete_idx_set[:pivot + sample_length]

        remain_idx = list(set(complete_idx_set) - set(local_idx))

        # 当只用过去的帧作为非局部帧时，可能会出现过去的帧数量少于非局部帧需求的问题，比如视频的一开始
        if before_nlf:
            if len(remain_idx) < num_ref_frame:
                # 则我们允许从局部帧中采样非局部帧 转换为set可以去除重复元素
                remain_idx = list(set(remain_idx + local_idx))

        ref_index = sorted(random.sample(remain_idx, num_ref_frame))

        return local_idx + ref_index

    def load_item_v4(self):
        """避免dataloader的index和worker的index冲突"""
        video_name = self.video_names[self.index]

        # TODO: 保证每次切换到新视频前mask是一致的，这样记忆才有意义？ 这样逻辑也和测试逻辑一致了
        # create masks
        if self.dataset_name != 'KITTI360-EX':
            # 对于非KITTI360-EX数据集，随机创建mask
            if not self.same_mask:
                # 每次迭代都会生成新形状的随机mask
                # if self.dataset_name != 'KITTI360-EX':
                all_masks = create_random_shape_with_random_motion(
                    self.video_dict[video_name], imageHeight=self.h, imageWidth=self.w)
                # else:
                #     # 对于KITTI360训练，需要读取本地存储的mask
                #     all_masks = []
                #     for idx in range(0, self.video_dict[video_name]):
                #         mask_path = os.path.join(self.args['data_root'],
                #                                  'test_masks',
                #                                  video_name,
                #                                  str(idx).zfill(6) + '.png')
                #         mask = Image.open(mask_path).resize(self.size,
                #                                             Image.NEAREST).convert('L')
                #         mask = np.asarray(mask)
                #         m = np.array(mask > 0).astype(np.uint8)
                #         m = cv2.dilate(m,
                #                        cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)),
                #                        iterations=4)
                #         mask = Image.fromarray(m * 255)
                #         all_masks.append(mask)

            else:
                # 在切换新视频前使用一样的mask参数
                all_masks, random_dict = create_random_shape_with_random_motion_seq(
                    self.video_dict[video_name], imageHeight=self.h, imageWidth=self.w,
                    new_mask=self.new_mask_list[self.batch_buffer],
                    random_dict=self.random_dict_list[self.batch_buffer])
                # 更新随机mask的参数
                self.random_dict_list[self.batch_buffer] = random_dict

        # create sample index
        # 对于KITTI360-EX这样视场角扩展的场景，非局部帧只能从过去的信息中获取
        if self.dataset_name == 'KITTI360-EX':
            before_nlf = True
        else:
            # 默认视频补全可以用未来的信息
            before_nlf = False
        selected_index = self._sample_index_seq(self.video_dict[video_name],
                                                self.num_local_frames,
                                                self.num_ref_frames,
                                                pivot=self.start_index,
                                                before_nlf=before_nlf)

        # read video frames
        frames = []
        masks = []
        for idx in selected_index:
            if self.dataset_name != 'KITTI360-EX':
                video_path = os.path.join(self.args['data_root'],
                                          self.args['name'], 'JPEGImages',
                                          f'{video_name}.zip')
            else:
                video_path = os.path.join(self.args['data_root'],
                                          'JPEGImages',
                                          f'{video_name}.zip')
            img = TrainZipReader.imread(video_path, idx).convert('RGB')
            img = img.resize(self.size)
            frames.append(img)
            if self.dataset_name != 'KITTI360-EX':
                masks.append(all_masks[idx])
            else:
                # 对于KITTI360-EX数据集，读取zip中存储的mask
                mask_path = os.path.join(self.args['data_root'],
                                         'test_masks',
                                          f'{video_name}.zip')
                mask = TrainZipReader.imread(mask_path, idx)
                mask = mask.resize(self.size).convert('L')
                mask = np.asarray(mask)
                m = np.array(mask > 0).astype(np.uint8)
                # 不对训练的mask进行膨胀
                # m = cv2.dilate(m,
                #                cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)),
                #                iterations=4)
                mask = Image.fromarray(m * 255)
                masks.append(mask)

        # normalizate, to tensors
        frames = GroupRandomHorizontalFlip()(frames)
        if self.dataset_name == 'KITTI360-EX':
            # 对于本地读取的mask 也需要随着frame翻转
            masks = GroupRandomHorizontalFlip()(masks)
        frame_tensors = self._to_tensors(frames) * 2.0 - 1.0
        mask_tensors = self._to_tensors(masks)

        if not self.same_mask:
            # 每次生成新的随机mask，不需要返回字典
            return frame_tensors, mask_tensors, video_name, self.index, self.start_index
        else:
            # 要控制mask的行为一致，需要返回字典
            return frame_tensors, mask_tensors, video_name, self.index, self.start_index,\
                   self.new_mask_list[self.batch_buffer], self.random_dict_list


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, args):
        self.args = args
        self.size = self.w, self.h = args.size

        with open(os.path.join(args.data_root, args.dataset, 'test.json'),
                  'r') as f:
            self.video_dict = json.load(f)
        self.video_names = list(self.video_dict.keys())

        self._to_tensors = transforms.Compose([
            Stack(),
            ToTorchFormatTensor(),
        ])

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, index):
        item = self.load_item(index)
        return item

    def load_item(self, index):
        video_name = self.video_names[index]
        ref_index = list(range(self.video_dict[video_name]))

        # read video frames
        frames = []
        masks = []
        for idx in ref_index:
            video_path = os.path.join(self.args.data_root, self.args.dataset,
                                      'JPEGImages', f'{video_name}.zip')
            img = TestZipReader.imread(video_path, idx).convert('RGB')
            img = img.resize(self.size)
            frames.append(img)
            mask_path = os.path.join(self.args.data_root, self.args.dataset,
                                     'test_masks', video_name,
                                     str(idx).zfill(5) + '.png')
            mask = Image.open(mask_path).resize(self.size,
                                                Image.NEAREST).convert('L')
            # origin: 0 indicates missing. now: 1 indicates missing
            mask = np.asarray(mask)
            m = np.array(mask > 0).astype(np.uint8)
            m = cv2.dilate(m,
                           cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)),
                           iterations=4)
            mask = Image.fromarray(m * 255)
            masks.append(mask)

        # to tensors
        frames_PIL = [np.array(f).astype(np.uint8) for f in frames]
        frame_tensors = self._to_tensors(frames) * 2.0 - 1.0
        mask_tensors = self._to_tensors(masks)
        return frame_tensors, mask_tensors, video_name, frames_PIL
