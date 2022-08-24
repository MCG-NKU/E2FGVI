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
                        ToTorchFormatTensor, GroupRandomHorizontalFlip)


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
    """
    def __init__(self, args: dict, debug=False, start=0, end=1, batch_size=1):
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

        self.neighbor_stride = 5    # 每隔5步采样一组LF和NLF
        self.start_index = 0        # 采样的起始点
        self.start = start          # 数据集迭代起点
        self.end = len(self.video_names)              # 数据集迭代终点
        self.batch_size = batch_size    # 用于计算数据集迭代器
        self.index = 0      # 自定义视频index
        self.batch_buffer = 0   # 用于随着batch size更新视频index
        self.new_video_flag = False     # 用于判断是否到了新视频
        self.worker_group = 0   # 用于随着每组worker更新

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

    # 两次迭代相距5*bs帧, 不同batch通道是同一个视频
    # def __getitem__(self, index):
    #     # item = self.load_item(index)
    #     if self.new_video_flag is True:
    #         self.index += 1
    #         self.new_video_flag = False\
    #
    #     worker_info = torch.utils.data.get_worker_info()
    #     if worker_info is None:
    #         # single-process data loading, skip idx rearrange
    #         pass
    #     else:
    #         if self.batch_buffer == 0:
    #             if self.start_index == 0:
    #                 previous_start_idx = self.start_index
    #             else:
    #                 previous_start_idx = self.start_index - self.neighbor_stride
    #             for worker_idx in range(0, worker_info.id):
    #                 previous_start_idx += self.batch_size * self.neighbor_stride
    #             self.start_index = self.start_index + previous_start_idx
    #         else:
    #             pass
    #
    #     item = self.load_item_v2(index=self.index)
    #
    #     self.batch_buffer += 1
    #     if self.batch_buffer == self.batch_size:
    #         # 重置batch缓存
    #         self.batch_buffer = 0
    #     return item

    # # 两次迭代相距5帧, 不同batch通道是同一个视频
    # def __getitem__(self, index):
    #     # item = self.load_item(index)
    #     if self.new_video_flag is True:
    #         self.index += 1
    #         self.new_video_flag = False\
    #
    #     worker_info = torch.utils.data.get_worker_info()
    #     if worker_info is None:
    #         # single-process data loading, skip idx rearrange
    #         pass
    #     else:
    #         if self.batch_buffer == 0:
    #             if self.start_index == 0:
    #                 self.start_index += self.neighbor_stride * worker_info.id
    #             else:
    #                 self.start_index = (self.start_index - 5) + worker_info.num_workers*self.neighbor_stride*self.batch_buffer
    #         else:
    #             self.start_index = (self.start_index-5) + self.batch_buffer * self.neighbor_stride * worker_info.num_workers
    #
    #     item = self.load_item_v2(index=self.index)
    #
    #     self.batch_buffer += 1
    #     if self.batch_buffer == self.batch_size:
    #         # 重置batch缓存
    #         self.batch_buffer = 0
    #     return item

    # 两次迭代相距5帧, 不同batch通道是不同的视频, 视频index和起始帧index在batch之间共用
    # def __getitem__(self, index):
    #     # item = self.load_item(index)
    #     if self.new_video_flag is True:
    #         self.index += 1
    #         self.new_video_flag = False\
    #
    #     worker_info = torch.utils.data.get_worker_info()
    #     if worker_info is None:
    #         # single-process data loading, skip idx rearrange
    #         pass
    #     else:
    #         if self.batch_buffer == 0:
    #
    #             if self.start_index == 0 and self.worker_group == 0:
    #                 # 新视频, 初始化start index
    #                 # 判断start index有没有超出视频的长度
    #                 if (self.neighbor_stride * worker_info.id) <= self.video_dict[self.video_names[self.index]]:
    #                     self.start_index = self.neighbor_stride * worker_info.id
    #                 else:
    #                     # 超出则切换到下一个视频，并且start index置为0(每个worker的start仍然不同)
    #                     self.new_video_flag = True
    #                     self.worker_group = 0
    #                     # self.start_index = 0
    #                     self.start_index = self.neighbor_stride * worker_info.id
    #                     # 判断视频 index有没有超出视频的个数
    #                     if (self.index + 1) <= len(self.video_names):
    #                         self.index += 1
    #                     else:
    #                         # 超出则切换回第一个视频
    #                         self.index = 0
    #             else:
    #                 # 判断start index有没有超出视频的长度
    #                 if (self.start_index + self.neighbor_stride * worker_info.num_workers) <= self.video_dict[self.video_names[self.index]]:
    #                     self.start_index += self.neighbor_stride * worker_info.num_workers
    #                 else:
    #                     # 超出则切换到下一个视频，并且start index置为0(每个worker的start仍然不同)
    #                     self.new_video_flag = True
    #                     self.worker_group = 0
    #                     # self.start_index = 0
    #                     self.start_index = self.neighbor_stride * worker_info.id
    #                     # 判断视频 index有没有超出视频的个数
    #                     if (self.index + 1) <= len(self.video_names):
    #                         self.index += 1
    #                     else:
    #                         # 超出则切换回第一个视频
    #                         self.index = 0
    #
    #             # 下一组worker启动时恢复视频的index
    #             if self.index != 0:
    #                 self.index -= (self.batch_size - 1)
    #
    #         else:
    #             self.start_index = self.start_index
    #
    #         # 每个batch之间切换新的视频
    #         if (self.index + self.batch_buffer) <= len(self.video_names):
    #             self.index += self.batch_buffer
    #         else:
    #             self.index = 0
    #
    #     # 根据index和start index读取帧
    #     # item = self.load_item_v2(index=self.index)
    #     item = self.load_item_v3(index=self.index)
    #
    #     # 更新woker group的index
    #     if self.batch_buffer == (self.batch_size-1):
    #         self.worker_group += 1
    #
    #     # self.start_index += self.neighbor_stride
    #
    #     # 更新起始帧位置
    #     # if (self.start_index + 2*self.neighbor_stride) <= len(self.video_names):
    #     #     self.start_index += self.neighbor_stride
    #     # else:
    #     #     self.start_index = 0
    #     #     self.new_video_flag = True
    #
    #     self.batch_buffer += 1
    #     if self.batch_buffer == self.batch_size:
    #         # 重置batch缓存
    #         self.batch_buffer = 0
    #
    #     return item

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

    def _sample_index_seq(self, length, sample_length, num_ref_frame=3, pivot=0):
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

        remain_idx = list(set(complete_idx_set) - set(local_idx))
        ref_index = sorted(random.sample(remain_idx, num_ref_frame))

        return local_idx + ref_index

    def load_item(self, index):
        # 会导致batch不等于1的时候两个video的长度不一致，并且显存占用大
        video_name = self.video_names[index]
        # create masks
        all_masks = create_random_shape_with_random_motion(
            self.video_dict[video_name], imageHeight=self.h, imageWidth=self.w)

        # read video frames
        frames = []
        masks = []
        group_cnt = 0
        for i in list(range(self.video_dict[video_name])):
            if i % self.neighbor_stride != 0:
                # 每5帧采样一次
                pass
            else:
                # create sample index
                group_cnt += 1
                selected_index = self._sample_index_seq(self.video_dict[video_name],
                                                        self.num_local_frames,
                                                        self.num_ref_frames,
                                                        pivot=i)
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
        return frame_tensors, mask_tensors, video_name, group_cnt

    def load_item_v2(self, index):
        video_name = self.video_names[index]
        # create masks
        all_masks = create_random_shape_with_random_motion(
            self.video_dict[video_name], imageHeight=self.h, imageWidth=self.w)

        # create sample index
        selected_index = self._sample_index_seq(self.video_dict[video_name],
                                                self.num_local_frames,
                                                self.num_ref_frames,
                                                pivot=self.start_index)

        # 更新起始帧位置
        if (self.start_index + 2*self.neighbor_stride) <= self.video_dict[video_name]:
            self.start_index += self.neighbor_stride
        else:
            self.start_index = 0
            self.new_video_flag = True

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

        if self.new_video_flag:
            return frame_tensors, mask_tensors, video_name, index, 0
        else:
            return frame_tensors, mask_tensors, video_name, index, self.start_index-self.neighbor_stride

    def load_item_v3(self, index):
        video_name = self.video_names[index]
        # create masks
        all_masks = create_random_shape_with_random_motion(
            self.video_dict[video_name], imageHeight=self.h, imageWidth=self.w)

        # create sample index
        selected_index = self._sample_index_seq(self.video_dict[video_name],
                                                self.num_local_frames,
                                                self.num_ref_frames,
                                                pivot=self.start_index)

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

        return frame_tensors, mask_tensors, video_name, index, self.start_index

    def load_item_v4(self):
        """避免dataloader的index和worker的index冲突"""
        video_name = self.video_names[self.index]
        # create masks
        all_masks = create_random_shape_with_random_motion(
            self.video_dict[video_name], imageHeight=self.h, imageWidth=self.w)

        # create sample index
        selected_index = self._sample_index_seq(self.video_dict[video_name],
                                                self.num_local_frames,
                                                self.num_ref_frames,
                                                pivot=self.start_index)

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

        return frame_tensors, mask_tensors, video_name, self.index, self.start_index


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
