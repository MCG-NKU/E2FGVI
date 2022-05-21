# E<sup>2</sup>FGVI (CVPR 2022)-简体中文
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/towards-an-end-to-end-framework-for-flow/video-inpainting-on-davis)](https://paperswithcode.com/sota/video-inpainting-on-davis?p=towards-an-end-to-end-framework-for-flow)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/towards-an-end-to-end-framework-for-flow/video-inpainting-on-youtube-vos)](https://paperswithcode.com/sota/video-inpainting-on-youtube-vos?p=towards-an-end-to-end-framework-for-flow)

![Python 3.7](https://img.shields.io/badge/python-3.7-green.svg?style=plastic)
![pytorch 1.6.0](https://img.shields.io/badge/pytorch-1.5.1-green.svg?style=plastic)

[English](README.md) | 简体中文

本项目包含了以下论文的官方实现：
> **Towards An End-to-End Framework for Flow-Guided Video Inpainting**<br>
> Zhen Li<sup>#</sup>, Cheng-Ze Lu<sup>#</sup>, Jianhua Qin, Chun-Le Guo<sup>*</sup>, Ming-Ming Cheng<br>
> IEEE/CVF Conference on Computer Vision and Pattern Recognition (**CVPR**), 2022<br>

[[论文](https://arxiv.org/abs/2204.02663)]
[[Demo Video (Youtube)](https://www.youtube.com/watch?v=N--qC3T2wc4)]
[[演示视频 (B站)](https://www.bilibili.com/video/BV1Ta411n7eH?spm_id_from=333.999.0.0)]
[项目主页 (待定)]
[海报 (待定)]

Colab实例：[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/12rwY2gtG8jVWlNx9pjmmM8uGmh5ue18G?usp=sharing)

## :star: 最新进展
- *2022.05.15:* 可适配**任意分辨率**的E<sup>2</sup>FGVI-HQ已发布.该模型仅需要在 432x240 的分辨率下进行训练, 即可适配更高分辨率下的推理任务.并且, 该模型比原先模型能够取得**更好**的PSNR/SSIM指标.
:link: 下载链接: [[Google Drive](https://drive.google.com/file/d/10wGdKSUOie0XmCr8SQ2A2FeDe-mfn5w3/view?usp=sharing)] [[Baidu Disk](https://pan.baidu.com/s/1jfm1oFU1eIy-IRfuHP8YXw?pwd=ssb3)] :movie_camera: 演示视频: [[Youtube](https://www.youtube.com/watch?v=N--qC3T2wc4)] [[B站](https://www.bilibili.com/video/BV1Ta411n7eH?spm_id_from=333.999.0.0)]

- *2022.04.06:* 代码公开发布.
## 演示视频

![teaser](./figs/teaser.gif)

### 更多示例 (点击查看详情):

<table>
<tr>
   <td> 
      <details> 
      <summary> 
      <strong>Coco </strong>
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

## 概述
![overall_structure](./figs/framework.png)

### :rocket: 特性：
- **更好的性能**: 本文提出的E<sup>2</sup>FGVI模型相较于现有工作在所有量化指标上取得了显著提升.
- **更快的速度**: 本文的方法在一张Titan XP GPU上, 处理分辨率为 432 × 240 的视频大约需要0.12秒/帧, 大约是前有的基于光流的方法的15倍.除此以外, 本文的方法相较于之前最优的方法具有最低的FLOPs计算量.

## 正在进行中的工作
- [ ] 更新项目主页
- [ ] Hugging Face 演示
- [ ] 更高效的推理过程

## 安装

1. 克隆仓库

   ```bash
   git clone https://github.com/MCG-NKU/E2FGVI.git
   ```

2. 创建Conda环境并且安装依赖

   ```bash
   conda env create -f environment.yml
   conda activate e2fgvi
   ```
   - Python >= 3.7
   - PyTorch >= 1.5
   - CUDA >= 9.2
   - [mmcv-full](https://github.com/open-mmlab/mmcv#installation) (following the pipeline to install)

   若无法使用`environment.yml`安装依赖, 请参照[此处](https://github.com/MCG-NKU/E2FGVI/issues/3).

## 快速入门
### 准备预训练模型
首先请下载预训练模型

<table>
<thead>
  <tr>
    <th>模型</th>
    <th>:link: 下载链接 </th>
    <th>支持任意分辨率 ?</th>
    <th> PSNR / SSIM / VFID (DAVIS) </th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>E<sup>2</sup>FGVI</td>
    <th>
       [<a href="https://drive.google.com/file/d/1tNJMTJ2gmWdIXJoHVi5-H504uImUiJW9/view?usp=sharing">谷歌网盘</a>] 
       [<a href="https://pan.baidu.com/s/1qXAErbilY_n_Fh9KB8UF7w?pwd=lsjw">百度网盘</a>]
    </th>
    <th>:x:</th>
    <th>33.01 / 0.9721 / 0.116</th>
  </tr>
  <tr>
    <td>E<sup>2</sup>FGVI-HQ</td>
    <th>
       [<a href="https://drive.google.com/file/d/10wGdKSUOie0XmCr8SQ2A2FeDe-mfn5w3/view?usp=sharing">谷歌网盘</a>] 
       [<a href="https://pan.baidu.com/s/1jfm1oFU1eIy-IRfuHP8YXw?pwd=ssb3">百度网盘</a>]
    </th>
    <th>:o:</th>
    <th>33.06 / 0.9722 / 0.117</th>
  </tr>
</tbody>
</table>

然后, 解压文件并且将模型放入`release_model`文件夹下. 

文件夹目录结构如下：
```
release_model
   |- E2FGVI-CVPR22.pth
   |- E2FGVI-HQ-CVPR22.pth
   |- i3d_rgb_imagenet.pt (for evaluating VFID metric)
   |- README.md
```

### 测试
我们提供了两个测试[`示例`](./examples)

使用如下命令运行：
```shell
# 第一个示例 （使用视频帧）
python test.py --model e2fgvi (or e2fgvi_hq) --video examples/tennis --mask examples/tennis_mask  --ckpt release_model/E2FGVI-CVPR22.pth (or release_model/E2FGVI-HQ-CVPR22.pth)
# 第二个示例 （使用mp4格式的视频）
python test.py --model e2fgvi (or e2fgvi_hq) --video examples/schoolgirls.mp4 --mask examples/schoolgirls_mask  --ckpt release_model/E2FGVI-CVPR22.pth (or release_model/E2FGVI-HQ-CVPR22.pth)
```
视频补全的结果会被保存在`results`路径下.若果想要测试更多样例, 请准备**mp4视频**（或**视频帧**）以及**每一帧的mask**.

*注意：* E<sup>2</sup>FGVI会将输入视频放缩到固定的分辨率（432x240）, 然而E<sup>2</sup>FGVI-HQ不会改变输入视频的分辨率.如果需要自定义输出的分辨率, 请设置`--set_size`参数以及设置输出分辨率的`--width`和`--height`值.

例:
```shell
# 使用该命令输入720p视频
python test.py --model e2fgvi_hq --video <video_path> --mask <mask_path>  --ckpt release_model/E2FGVI-HQ-CVPR22.pth --set_size --width 1280 --height 720
```


### 准备训练与验证集
<table>
<thead>
  <tr>
    <th>数据集</th>
    <th>YouTube-VOS</th>
    <th>DAVIS</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>详情</td>
    <td>训练: 3,471, 验证: 508</td>
    <td>验证: 50 (共90)</td>
  <tr>
    <td>Images</td>
    <td> [<a href="https://competitions.codalab.org/competitions/19544#participate-get-data">官方链接</a>] (下载全部训练测试集) </td>
    <td> [<a href="https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip">官方链接</a>] (2017, 480p, TrainVal) </td>
  </tr>
  <tr>
    <td>Masks</td>
    <td colspan="2"> [<a href="https://drive.google.com/file/d/1dFTneS_zaJAHjglxU10gYzr1-xALgHa4/view?usp=sharing">谷歌网盘</a>] [<a href="https://pan.baidu.com/s/1JC-UKmlQfjhVtD81196cxA?pwd=87e3">百度网盘</a>] (复现论文结果) </td>
  </tr>
</tbody>
</table>

训练与测试集分割文件位于 `datasets/<dataset_name>`.

对于每一个数据集, 需要将 `JPEGImages` 放入 `datasets/<dataset_name>`目录下.

然后, 运行 `sh datasets/zip_dir.sh` (**注意**: 请编辑对应的目录路径) 来压缩位于`datasets/<dataset_name>/JPEGImages`的每一个视频.

将下载的mask解压缩至 `datasets`.

`datasets`目录结构如下: (**注意**: 请仔细核验)
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
运行如下的一个命令进行验证:
```shell
 # 验证E2FGVI模型
 python evaluate.py --model e2fgvi --dataset <dataset_name> --data_root datasets/ --ckpt release_model/E2FGVI-CVPR22.pth
 # 验证E2FGVI-HQ模型
 python evaluate.py --model e2fgvi_hq --dataset <dataset_name> --data_root datasets/ --ckpt release_model/E2FGVI-HQ-CVPR22.pth

```
若你验证 E<sup>2</sup>FGVI 模型, 那么将会得到论文中的验证结果.
E<sup>2</sup>FGVI-HQ 的验证结果请参考 [[此处](https://github.com/MCG-NKU/E2FGVI#prepare-pretrained-models)].

验证结果将被保存在 `results/<model_name>_<dataset_name>` 目录下.

若需[验证temporal warping error](https://github.com/phoenix104104/fast_blind_video_consistency#evaluation), 请添加 `--save_results` 参数.

### 训练
Our training configures are provided in [`train_e2fgvi.json`](./configs/train_e2fgvi.json) (for E<sup>2</sup>FGVI) and [`train_e2fgvi_hq.json`](./configs/train_e2fgvi_hq.json) (for E<sup>2</sup>FGVI-HQ).

本文的训练配置如 [`train_e2fgvi.json`](./configs/train_e2fgvi.json) (对于 E<sup>2</sup>FGVI) 与 [`train_e2fgvi_hq.json`](./configs/train_e2fgvi_hq.json) (对于 E<sup>2</sup>FGVI-HQ) 所示.

运行如下的一条命令进行训练：
```shell
 # 训练 E2FGVI
 python train.py -c configs/train_e2fgvi.json
 # 训练 E2FGVI-HQ
 python train.py -c configs/train_e2fgvi_hq.json
```
如果需要恢复训练, 请运行相同的指令.

训练损失能够使用如下命令可视化：
```shell
tensorboard --logdir release_model                                                   
```

请使用上述[步骤](https://github.com/MCG-NKU/E2FGVI#evaluation)来验证训练的模型.

## 结果  

### 定量结果
![quantitative_results](./figs/quantitative_results.png)
## 引用

   若我们的仓库对你的研究内容有帮助, 请参考如下 bibtex 引用本文：

   ```bibtex
   @inproceedings{liCvpr22vInpainting,
      title={Towards An End-to-End Framework for Flow-Guided Video Inpainting},
      author={Li, Zhen and Lu, Cheng-Ze and Qin, Jianhua and Guo, Chun-Le and Cheng, Ming-Ming},
      booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
      year={2022}
   }
   ```
## 联系方式

若有任何疑问, 请通过`zhenli1031ATgmail.com` 或 `czlu919AToutlook.com`联系.


## 致谢

该仓库由 [Zhen Li](https://paper99.github.io) 与 [Cheng-Ze Lu](https://github.com/LGYoung) 维护.

代码基于 [STTN](https://github.com/researchmm/STTN), [FuseFormer](https://github.com/ruiliu-ai/FuseFormer), [Focal-Transformer](https://github.com/microsoft/Focal-Transformer), 与 [MMEditing](https://github.com/open-mmlab/mmediting).
