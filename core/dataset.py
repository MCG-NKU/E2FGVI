import os
import json
import random

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
