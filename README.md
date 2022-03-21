# E<sup>2</sup>FGVI (CVPR 2022)

![Python 3.7](https://img.shields.io/badge/python-3.7-green.svg?style=plastic)
![pytorch 1.6.0](https://img.shields.io/badge/pytorch-1.5.1-green.svg?style=plastic)

This repository contains the official implementation of the following paper:
> **Towards An End-to-End Framework for Flow-Guided Video Inpainting**<br>
> Zhen Li<sup>#</sup>, Cheng-Ze Lu<sup>#</sup>, Jianhua Qin, Chun-Le Guo<sup>*</sup>, Ming-Ming Cheng<br>
> IEEE/CVF Conference on Computer Vision and Pattern Recognition (**CVPR**), 2022<br>

[Paper]
[Project Page (TBD)]
[Poster (TBD)]
[Video (TBD)]

You can try our colab demo here: [Open in Colab] (TBD)

## Demo

![teaser](./figs/teaser.gif)

### More examples (click for details):

<table>
<tr>
   <td> 
      <details> 
      <summary> 
      <strong>Coco (click me)</strong>
      </summary> 
      <img src="https://user-images.githubusercontent.com/21050959/159160822-8ed5947c-e91d-4597-8e20-4b443a2244ed.gif">
      </details>
   </td>
   <td> 
      <details> 
      <summary> 
      <strong>Tennis </strong>
      </summary> 
      <img src="https://user-images.githubusercontent.com/21050959/159160843-4b167115-e338-4e0b-9ca4-b564233c2c7a.gif">
      </details>
   </td>
</tr>
<tr>
   <td> 
      <details> 
      <summary> 
      <strong>Space </strong>
      </summary> 
      <img src="https://user-images.githubusercontent.com/21050959/159171328-1222c70e-9bb9-47e3-b765-4b1baaf631f5.gif">
      </details>
   </td>
   <td> 
      <details> 
      <summary> 
      <strong>Motocross </strong>
      </summary> 
      <img src="https://user-images.githubusercontent.com/21050959/159163010-ed78b4bd-c8dd-472c-ad3e-82bc8baca43a.gif">
      </details>
   </td>
</tr>
</table>

## Overview
![overall_structure](./figs/framework.png)

### :rocket: Highlights:
- **SOTA performance**: The proposed E<sup>2</sup>FGVI achieves significant improvements on all quantitative metrics in comparison with SOTA methods.
- **Highly effiency**: Our method processes 432 × 240 videos at 0.12 seconds per frame on a Titan XP GPU, which is nearly 15× faster than previous flow-based methods. Besides, our method has the lowest FLOPs among all compared SOTA
methods.

## Work in Progress
- [ ] Colab demo
- [ ] Update arXiv link & website page
- [ ] High-resolution version
- [ ] Update Youtube / Bilibili link

## Dependencies and Installation

1. Clone Repo

   ```bash
   git clone https://github.com/Paper99/E2FGVI.git
   ```

2. Create Conda Environment and Install Dependencies

   ```bash
   conda env create -f environment.yml
   conda activate e2fgvi
   ```
   - Python >= 3.7
   - PyTorch >= 1.5
   - CUDA >= 9.2
   - [mmcv-full](https://github.com/open-mmlab/mmcv#installation) (following the pipeline to install)

## Get Started
### Prepare pretrained models
Before performing the following steps, please download our pretrained model first.

:link: **Download Links:** [[Google Drive](https://drive.google.com/file/d/1tNJMTJ2gmWdIXJoHVi5-H504uImUiJW9/view?usp=sharing)] [[Baidu Disk](https://pan.baidu.com/s/1qXAErbilY_n_Fh9KB8UF7w?pwd=lsjw)]

Then, unzip the file and place the models to `release_model` directory.

The directory structure will be arranged as:
```
release_model
   |- E2FGVI-CVPR22.pth
   |- i3d_rgb_imagenet.pt (for evaluating VFID metric)
   |- README.md
```

### Quick test
We provide two examples in the [`examples`](./examples) directory.

Run the following command to enjoy them:
```shell
# The first example (using split video frames)
python test.py --video examples/tennis --mask examples/tennis_mask  --ckpt release_model/E2FGVI-CVPR22.pth
# The second example (using mp4 format video)
python test.py --video examples/schoolgirls.mp4 --mask examples/schoolgirls_mask  --ckpt release_model/E2FGVI-CVPR22.pth
```
The inpainting video will be saved in the `results` directory.

Please prepare your own **mp4 video** (or **split frames**) and **frame-wise masks** if you want to test more cases.
### Prepare dataset for training and evaluation
<table>
<thead>
  <tr>
    <th>Dataset</th>
    <th>YouTube-VOS</th>
    <th>DAVIS</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>Details</td>
    <td>For training (3,471) and evaluation (508)</td>
    <td>For evaluation (50 in 90)</td>
  <tr>
    <td>Images</td>
    <td> [<a href="https://competitions.codalab.org/competitions/19544#participate-get-data">Official Link</a>] (Download train and test all frames) </td>
    <td> [<a href="https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip">Official Link</a>] (2017, 480p, TrainVal) </td>
  </tr>
  <tr>
    <td>Masks</td>
    <td colspan="2"> [<a href="https://drive.google.com/file/d/1dFTneS_zaJAHjglxU10gYzr1-xALgHa4/view?usp=sharing">Google Drive</a>] [<a href="https://pan.baidu.com/s/1JC-UKmlQfjhVtD81196cxA?pwd=87e3">Baidu Disk</a>] (For reproducing paper results) </td>
  </tr>
</tbody>
</table>

The training and test split files are provided in `datasets/<dataset_name>`.

For each dataset, you should place `JPEGImages` to `datasets/<dataset_name>`.

Then, run `sh datasets/zip_dir.sh` (**Note**: please edit the folder path accordingly) for compressing each video in `datasets/<dataset_name>/JPEGImages`.

Unzip downloaded mask files to `datasets`.

The `datasets` directory structure will be arranged as: (**Note**: please check it carefully)
```
datasets
   |- davis
      |- JPEGImages
         |- <video_name>.zip
         |- <video_name>.zip
      |- test_masks
         |- <video_name>
            |- 00000.png
            |- 00001.png   
      |- train.json
      |- test.json
   |- youtube-vos
      |- JPEGImages
         |- <video_id>.zip
         |- <video_id>.zip
      |- test_masks
         |- <video_id>
            |- 00000.png
            |- 00001.png
      |- train.json
      |- test.json   
   |- zip_file.sh
```
### Evaluation
Run the following command for evaluation:
```shell
 python evaluate.py --dataset <dataset_name> --data_root datasets/ --ckpt release_model/E2FGVI-CVPR22.pth
```
You will get scores as paper reported.

The scores will also be saved in the `results/<dataset_name>` directory.

Please `--save_results` for further [evaluating temporal warping error](https://github.com/phoenix104104/fast_blind_video_consistency#evaluation).

### Training
Our training configures are provided in [`train_e2fgvi.json`](./configs/train_e2fgvi.json)

Run the following command for training:
```shell
 python train.py -c configs/train_e2fgvi.json
```
You could run the same command if you want to resume your training.

The training loss can be monitored by running:
```shell
tensorboard --logdir release_model                                                   
```

You could follow [this pipeline](https://github.com/Paper99/E2FGVI#evaluation) to evaluate your model.
## Results  

### Quantitative results
![quantitative_results](./figs/quantitative_results.png)
## Citation

   If you find our repo useful for your research, please consider citing our paper:

   ```bibtex
   @inproceedings{li2022e2fgvi,
     author = {Zhen Li, Cheng-Ze Lu, Jianhua Qin, Chun-Le Guo, Ming-Ming Cheng},
     title = {Towards An End-to-End Framework for Flow-Guided Video Inpainting},
     booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
     year={2022}
   }
   ```
## Contact

If you have any question, please feel free to contact us via `zhenli1031ATgmail.com` or `czlu919AToutlook.com`.

## Acknowledgement

The codebase is maintained by [Zhen Li](https://github.com/Paper99) and [Cheng-Ze Lu](https://github.com/LGYoung).

This code is based on [STTN](https://github.com/researchmm/STTN), [FuseFormer](https://github.com/ruiliu-ai/FuseFormer), [Focal-Transformer](https://github.com/microsoft/Focal-Transformer), and [MMEditing](https://github.com/open-mmlab/mmediting).
