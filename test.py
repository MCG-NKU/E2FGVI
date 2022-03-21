# -*- coding: utf-8 -*-
import cv2
from PIL import Image
import numpy as np
import importlib
import os
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import animation
import torch

from core.utils import to_tensors


parser = argparse.ArgumentParser(description="E2FGVI")
parser.add_argument("-v", "--video", type=str, required=True)
parser.add_argument("-m", "--mask",   type=str, required=True)
parser.add_argument("-c", "--ckpt",   type=str, required=True)
parser.add_argument("--model", type=str, default='e2fgvi')
parser.add_argument("--width", type=int, default=432)
parser.add_argument("--height", type=int, default=240)
parser.add_argument("--outw", type=int, default=432)
parser.add_argument("--outh", type=int, default=240)
parser.add_argument("--step", type=int, default=10)
parser.add_argument("--num_ref", type=int, default=-1)
parser.add_argument("--neighbor_stride", type=int, default=5)
parser.add_argument("--savefps", type=int, default=24)
args = parser.parse_args()


w, h = args.width, args.height
ref_length = args.step  # ref_step
num_ref = args.num_ref
neighbor_stride = args.neighbor_stride
default_fps = args.savefps


# sample reference frames from the whole video 
def get_ref_index(f, neighbor_ids, length):
    ref_index = []
    if num_ref == -1:
        for i in range(0, length, ref_length):
            if i not in neighbor_ids:
                ref_index.append(i)
    else:
        start_idx = max(0, f - ref_length * (num_ref//2))
        end_idx = min(length, f + ref_length * (num_ref//2))
        for i in range(start_idx, end_idx+1, ref_length):
            if i not in neighbor_ids:
                if len(ref_index) > num_ref:
                    break
                ref_index.append(i)
    return ref_index


# read frame-wise masks
def read_mask(mpath):
    masks = []
    mnames = os.listdir(mpath)
    mnames.sort()
    for mp in mnames:
        m = Image.open(os.path.join(mpath, mp))
        m = m.resize((w, h), Image.NEAREST)
        m = np.array(m.convert('L'))
        m = np.array(m > 0).astype(np.uint8)
        m = cv2.dilate(m, cv2.getStructuringElement(
            cv2.MORPH_CROSS, (3, 3)), iterations=4)
        masks.append(Image.fromarray(m*255))
    return masks


#  read frames from video
def read_frame_from_videos(args):
    vname = args.video
    frames = []
    if args.use_mp4:
        vidcap = cv2.VideoCapture(vname)
        success, image = vidcap.read()
        count = 0
        while success:
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            frames.append(image.resize((w, h)))
            success, image = vidcap.read()
            count += 1
    else:
        lst = os.listdir(vname)
        lst.sort()
        fr_lst = [vname+'/'+name for name in lst]
        for fr in fr_lst:
            image = cv2.imread(fr)
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            frames.append(image.resize((w, h)))
    return frames


def main_worker():
    # set up models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = importlib.import_module('model.' + args.model)
    model = net.InpaintGenerator().to(device)
    data = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(data)
    print(f'Loading model from: {args.ckpt}')
    model.eval()

    # prepare datset
    args.use_mp4 = True if args.video.endswith('.mp4') else False
    print(f'Loading videos and masks from: {args.video} | INPUT MP4 format: {args.use_mp4}')
    frames = read_frame_from_videos(args)
    video_length = len(frames)
    imgs = to_tensors()(frames).unsqueeze(0) * 2 - 1
    frames = [np.array(f).astype(np.uint8) for f in frames]

    masks = read_mask(args.mask)
    binary_masks = [np.expand_dims((np.array(m) != 0).astype(np.uint8), 2)
                    for m in masks]
    masks = to_tensors()(masks).unsqueeze(0)
    imgs, masks = imgs.to(device), masks.to(device)
    comp_frames = [None] * video_length

    # completing holes by e2fgvi
    print(f'Start test...')
    for f in tqdm(range(0, video_length, neighbor_stride)):
        neighbor_ids = [i for i in range(max(0, f-neighbor_stride), min(video_length, f+neighbor_stride+1))]
        ref_ids = get_ref_index(f, neighbor_ids, video_length)
        selected_imgs = imgs[:1, neighbor_ids+ref_ids, :, :, :]
        selected_masks = masks[:1, neighbor_ids+ref_ids, :, :, :]
        with torch.no_grad():
            masked_imgs = selected_imgs*(1-selected_masks)
            pred_img, _ = model(masked_imgs, len(neighbor_ids))

            pred_img = (pred_img + 1) / 2
            pred_img = pred_img.cpu().permute(0, 2, 3, 1).numpy() * 255
            for i in range(len(neighbor_ids)):
                idx = neighbor_ids[i]
                img = np.array(pred_img[i]).astype(
                    np.uint8)*binary_masks[idx] + frames[idx] * (1-binary_masks[idx])
                if comp_frames[idx] is None:
                    comp_frames[idx] = img
                else:
                    comp_frames[idx] = comp_frames[idx].astype(
                        np.float32)*0.5 + img.astype(np.float32)*0.5

    # saving videos
    ext_name = '_results.mp4'
    save_base_name = args.video.split('/')[-1]
    save_name = save_base_name.replace('.mp4', ext_name) if args.use_mp4 else save_base_name + ext_name
    save_dir_name = 'results'
    if not os.path.exists(save_dir_name):
        os.makedirs(save_dir_name)
    save_path = os.path.join(save_dir_name, save_name)
    
    writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), default_fps, (args.outw, args.outh))
    for f in range(video_length):
        comp = comp_frames[f].astype(np.uint8)
        if w != args.outw:
            comp = cv2.resize(comp, (args.outw, args.outh), interpolation=cv2.INTER_LINEAR)
        writer.write(cv2.cvtColor(comp, cv2.COLOR_BGR2RGB))
    writer.release()
    print(f'Finish test! The result video is saved in: {save_path}.')

    # show results
    print('Let us enjoy the result!')
    fig = plt.figure('Let us enjoy the result')
    ax1 = fig.add_subplot(1,2,1)
    ax1.axis('off'); ax1.set_title('Original Video')
    ax2 = fig.add_subplot(1,2,2)
    ax2.axis('off'); ax2.set_title('Our Result')
    imdata1 = ax1.imshow(frames[0])
    imdata2 = ax2.imshow(comp_frames[0].astype(np.uint8))

    def update(idx):
        imdata1.set_data(frames[idx])
        imdata2.set_data(comp_frames[idx].astype(np.uint8))
    
    fig.tight_layout()
    anim = animation.FuncAnimation(fig, update, frames=len(frames), interval=50)
    plt.show()

if __name__ == '__main__':
    main_worker()
