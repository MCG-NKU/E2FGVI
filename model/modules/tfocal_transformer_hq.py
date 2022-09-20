"""
    This code is based on:
    [1] FuseFormer: Fusing Fine-Grained Information in Transformers for Video Inpainting, ICCV 2021
        https://github.com/ruiliu-ai/FuseFormer
    [2] Tokens-to-Token ViT: Training Vision Transformers from Scratch on ImageNet, ICCV 2021
        https://github.com/yitu-opensource/T2T-ViT
    [3] Focal Self-attention for Local-Global Interactions in Vision Transformers, NeurIPS 2021
        https://github.com/microsoft/Focal-Transformer
    [4] Self-slimmed Vision Transformer, ECCV 2022
        https://github.com/Sense-X/SiT
"""

import math
from functools import reduce

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.modules.feat_prop import flow_warp


class SoftSplit(nn.Module):
    def __init__(self, channel, hidden, kernel_size, stride, padding,
                 t2t_param):
        super(SoftSplit, self).__init__()
        self.kernel_size = kernel_size
        self.t2t = nn.Unfold(kernel_size=kernel_size,
                             stride=stride,
                             padding=padding)
        c_in = reduce((lambda x, y: x * y), kernel_size) * channel
        self.embedding = nn.Linear(c_in, hidden)

        self.t2t_param = t2t_param

    def forward(self, x, b, output_size):
        f_h = int((output_size[0] + 2 * self.t2t_param['padding'][0] -
                   (self.t2t_param['kernel_size'][0] - 1) - 1) /
                  self.t2t_param['stride'][0] + 1)      # token在竖直方向的个数
        f_w = int((output_size[1] + 2 * self.t2t_param['padding'][1] -
                   (self.t2t_param['kernel_size'][1] - 1) - 1) /
                  self.t2t_param['stride'][1] + 1)      # token在水平方向的个数

        feat = self.t2t(x)      # 把特征图划分为token(不含有可学习参数)，[B*t, C*token_h*token_w, f_h*f_w]
        feat = feat.permute(0, 2, 1)    # [B*t, Num_token, Length_token]
        # feat shape [b*t, num_vec, ks*ks*c]
        feat = self.embedding(feat)     # [B*t, Num_token, hidden] 含参数
        # feat shape after embedding [b, t*num_vec, hidden]
        feat = feat.view(b, -1, f_h, f_w, feat.size(2))     # [B, t, f_h, f_w, hidden]
        return feat


class SoftSplit_FlowGuide(nn.Module):
    """
    Using forward and backward flow to guide LOCAL trans feat embedding.
    Using same embedding func for local and non local frames now.
    """
    def __init__(self, channel, hidden, kernel_size, stride, padding,
                 t2t_param):
        super(SoftSplit_FlowGuide, self).__init__()
        self.kernel_size = kernel_size
        self.t2t = nn.Unfold(kernel_size=kernel_size,
                             stride=stride,
                             padding=padding)
        c_in = reduce((lambda x, y: x * y), kernel_size) * channel
        self.embedding = nn.Linear(c_in, hidden)
        self.embedding_with_flow = nn.Linear(c_in * 2 + c_in // channel * 4, hidden)  # addition channel of flow
        # self.embedding_with_flow = nn.Linear(c_in, hidden)

        self.t2t_param = t2t_param

    def forward(self, x, b, c, output_size, flow_forward, flow_backward, local_t):
        f_h = int((output_size[0] + 2 * self.t2t_param['padding'][0] -
                   (self.t2t_param['kernel_size'][0] - 1) - 1) /
                  self.t2t_param['stride'][0] + 1)      # token在竖直方向的个数
        f_w = int((output_size[1] + 2 * self.t2t_param['padding'][1] -
                   (self.t2t_param['kernel_size'][1] - 1) - 1) /
                  self.t2t_param['stride'][1] + 1)      # token在水平方向的个数
        h, w = output_size

        x_non_local = torch.cat((x[:, local_t:, :, :, :], x[:, local_t // 2, :, :, :].unsqueeze(dim=1)), dim=1)   # local frame 与光流对齐后t少了一帧，因此non_local_frame多取1帧中间帧
        local_feat = x[:, :local_t, :, :, :]
        forward_local_feat = local_feat[:, :-1, :, :, :]
        backward_local_feat = local_feat[:, 1:, :, :, :]
        x_local = torch.cat((forward_local_feat, flow_forward,
                             backward_local_feat, flow_backward), dim=2).view(-1, c*2+4, h, w)  # 2*channel + 2*flow
        feat_local = self.t2t(x_local)
        feat_local = feat_local.permute(0, 2, 1)
        feat_non_local = self.t2t(x_non_local.view(-1, c, h, w))
        feat_non_local = feat_non_local.permute(0, 2, 1)

        feat_local = self.embedding_with_flow(feat_local)
        feat_local = feat_local.view(b, -1, f_h, f_w, feat_local.size(2))
        feat_non_local = self.embedding(feat_non_local)
        feat_non_local = feat_non_local.view(b, -1, f_h, f_w, feat_non_local.size(2))

        return torch.cat((feat_local, feat_non_local), dim=1)


class TokenSlimmingModule(nn.Module):
    r"""Token Slim Module from SiT.
        Revised by Hao:
        Add token fusion for video inpainting support.
        Slightly change input and output dim, and ratio."""
    def __init__(self, dim, keeped_patches, ratio=0.5625):
        super().__init__()
        hidden_dim = int(dim * ratio)
        self.weight = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, keeped_patches))
        self.scale = nn.Parameter(torch.ones(1, 1, 1))

    def forward(self, x):
        b, t, num_h, num_w, hidden = x.shape
        x = x.view(b*t, -1, hidden)     # B*T, Num_Token, C

        weight = self.weight(x)
        weight = F.softmax(weight * self.scale, dim=1).transpose(2, 1)
        x = torch.bmm(weight, x)

        x = x.view(b, t, int(num_h * 0.75), int(num_w * 0.75), hidden)  # 两个方向上各自缩减25%的token
        return x


class ReverseTSM(nn.Module):
    r"""Reverse Token Slim Module from SiT.
        Revised by Hao:
        Add token fusion for video inpainting support.
        Slightly change input and output dim, and ratio."""
    def __init__(self, dim, keeped_patches, recovered_patches, ratio=4.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.mlp1 = Mlp(keeped_patches, int(recovered_patches*ratio), recovered_patches)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp2 = Mlp(dim, int(dim*ratio), dim)
    def forward(self, x):
        b, t, num_h, num_w, hidden = x.shape
        x = x.view(b*t, -1, hidden)     # B*T, Num_Token, C

        x = self.norm1(x)
        x = self.mlp1(x.transpose(2, 1))
        x = x.transpose(2, 1)
        x = x + self.mlp2(self.norm2(x))

        x = x.view(b, t, int(num_h * 4 // 3), int(num_w * 4 // 3), hidden)  # 两个方向上各自缩减25%的token
        return x


class ReverseTSM_v2(nn.Module):
    r"""Reverse Token Slim Module inspired by SiT.
        Revised by Hao:
        Add token fusion for video inpainting support.
        Change MLP structure.
        Add trans feat skip connection and mlp0 support."""
    def __init__(self, dim, keeped_patches, recovered_patches, ratio=4.):
        super().__init__()
        self.norm0 = nn.LayerNorm(dim)
        self.mlp0 = Mlp(keeped_patches + recovered_patches,
                        hidden_features=int((keeped_patches + recovered_patches)*ratio),
                        out_features=recovered_patches)
        self.norm1 = nn.LayerNorm(dim)
        self.mlp1 = Mlp(recovered_patches,
                        hidden_features=int(recovered_patches*ratio),
                        out_features=recovered_patches)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp2 = Mlp(dim,
                        hidden_features=int(dim*ratio),
                        out_features=dim)
    def forward(self, x, trans_feat):
        b, t, num_h, num_w, hidden = x.shape
        x = x.view(b*t, -1, hidden)     # B*T, Num_Token_Red, C
        trans_feat = trans_feat.view(b * t, -1, hidden)  # B*T, Num_Token_Ori, C
        x = torch.cat((x, trans_feat), dim=1)   # B*T, Num_Token_Ori + Num_Token_Red, C

        x = self.norm0(x)
        x = self.mlp0(x.transpose(2, 1))
        x = x.transpose(2, 1)
        x = self.norm1(x)
        x = self.mlp1(x.transpose(2, 1))
        x = x.transpose(2, 1)
        x = x + self.mlp2(self.norm2(x))

        x = x.view(b, t, int(num_h * 4 // 3), int(num_w * 4 // 3), hidden)  # 两个方向上各自缩减25%的token
        return x


class Mlp(nn.Module):
    '''
    MLP with support to use group linear operator
    '''
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.act = act_layer()
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class FlowHead(nn.Module):
    "Flow head for compute token flow"
    def __init__(self, input_dim, hidden_factor=2):
        super(FlowHead, self).__init__()

        hidden_dim = input_dim * hidden_factor
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 2, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


class SoftComp(nn.Module):
    r"""Revised by Hao:
        Add token fusion for video inpainting support.
        Transfer x to contiguous before view operation"""
    def __init__(self, channel, hidden, kernel_size, stride, padding):
        super(SoftComp, self).__init__()
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        c_out = reduce((lambda x, y: x * y), kernel_size) * channel
        self.embedding = nn.Linear(hidden, c_out)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias_conv = nn.Conv2d(channel,
                                   channel,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1)
        # TODO upsample conv
        # self.bias_conv = nn.Conv2d()
        # self.bias = nn.Parameter(torch.zeros((channel, h, w), dtype=torch.float32), requires_grad=True)

    def forward(self, x, t, output_size):
        b_, _, _, _, c_ = x.shape
        x = x.contiguous().view(b_, -1, c_)
        feat = self.embedding(x)
        b, _, c = feat.size()
        feat = feat.view(b * t, -1, c).permute(0, 2, 1)
        feat = F.fold(feat,
                      output_size=output_size,
                      kernel_size=self.kernel_size,
                      stride=self.stride,
                      padding=self.padding)
        feat = self.bias_conv(feat)
        return feat


class FusionFeedForward(nn.Module):
    def __init__(self, d_model, n_vecs=None, t2t_params=None):
        super(FusionFeedForward, self).__init__()
        # We set d_ff as a default to 1960
        hd = 1960
        self.conv1 = nn.Sequential(nn.Linear(d_model, hd))
        self.conv2 = nn.Sequential(nn.GELU(), nn.Linear(hd, d_model))
        assert t2t_params is not None and n_vecs is not None
        self.t2t_params = t2t_params

    def forward(self, x, output_size):
        n_vecs = 1
        for i, d in enumerate(self.t2t_params['kernel_size']):
            n_vecs *= int((output_size[i] + 2 * self.t2t_params['padding'][i] -
                           (d - 1) - 1) / self.t2t_params['stride'][i] + 1)

        x = self.conv1(x)
        b, n, c = x.size()
        normalizer = x.new_ones(b, n, 49).view(-1, n_vecs, 49).permute(0, 2, 1)
        normalizer = F.fold(normalizer,
                            output_size=output_size,
                            kernel_size=self.t2t_params['kernel_size'],
                            padding=self.t2t_params['padding'],
                            stride=self.t2t_params['stride'])

        x = F.fold(x.view(-1, n_vecs, c).permute(0, 2, 1),
                   output_size=output_size,
                   kernel_size=self.t2t_params['kernel_size'],
                   padding=self.t2t_params['padding'],
                   stride=self.t2t_params['stride'])

        x = F.unfold(x / normalizer,
                     kernel_size=self.t2t_params['kernel_size'],
                     padding=self.t2t_params['padding'],
                     stride=self.t2t_params['stride']).permute(
                         0, 2, 1).contiguous().view(b, n, c)
        x = self.conv2(x)
        return x


class MixConv2d(nn.Module):
    """MixConv2d from HRViT."""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.conv_3 = nn.Conv2d(
            in_channels // 2,
            out_channels // 2,
            kernel_size,
            stride=1,
            padding=padding,
            dilation=dilation,
            groups=groups // 2,
            bias=bias,
        )
        self.conv_5 = nn.Conv2d(
            in_channels - in_channels // 2,
            out_channels - out_channels // 2,
            kernel_size + 2,
            stride=stride,
            padding=padding + 1,
            dilation=dilation,
            groups=groups - groups // 2,
            bias=bias,
        )

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        x1 = self.conv_3(x1)
        x2 = self.conv_5(x2)
        x = torch.cat([x1, x2], dim=1)
        return x


class MixFusionFeedForward(nn.Module):
    """Mix F3N for transformer, by Hao."""
    def __init__(self, d_model, n_vecs=None, t2t_params=None):
        super(MixFusionFeedForward, self).__init__()
        # We set d_ff as a default to 1960
        # TODO: 研究这里的hidden dim和输入dim挂钩会怎么样
        hd = 1960   # hidden dim
        self.conv1 = nn.Sequential(nn.Linear(d_model, hd))

        # MixConv
        self.mix_conv = MixConv2d(
            in_channels=hd,
            out_channels=hd,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hd,
            dilation=1,
            bias=True,
        )

        self.conv2 = nn.Sequential(nn.GELU(), nn.Linear(hd, d_model))
        assert t2t_params is not None and n_vecs is not None
        self.t2t_params = t2t_params

    def forward(self, x, output_size, T, H, W):
        n_vecs = 1
        for i, d in enumerate(self.t2t_params['kernel_size']):
            n_vecs *= int((output_size[i] + 2 * self.t2t_params['padding'][i] -
                           (d - 1) - 1) / self.t2t_params['stride'][i] + 1)

        x = self.conv1(x)
        b, n, c = x.size()
        normalizer = x.new_ones(b, n, 49).view(-1, n_vecs, 49).permute(0, 2, 1)
        normalizer = F.fold(normalizer,
                            output_size=output_size,
                            kernel_size=self.t2t_params['kernel_size'],
                            padding=self.t2t_params['padding'],
                            stride=self.t2t_params['stride'])

        x = F.fold(x.view(-1, n_vecs, c).permute(0, 2, 1),
                   output_size=output_size,
                   kernel_size=self.t2t_params['kernel_size'],
                   padding=self.t2t_params['padding'],
                   stride=self.t2t_params['stride'])

        x = F.unfold(x / normalizer,
                     kernel_size=self.t2t_params['kernel_size'],
                     padding=self.t2t_params['padding'],
                     stride=self.t2t_params['stride']).permute(
                         0, 2, 1).contiguous().view(b, n, c)

        x = x.reshape(b*T, H, W, c).permute(0, 3, 1, 2).contiguous()    # B*T, C, H, W
        x = self.mix_conv(x).permute(0, 2, 3, 1).contiguous().reshape(b, n, c)  # B, T*H*W, C

        x = self.conv2(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: shape is (B, T, H, W, C)
        window_size (tuple[int]): window size
    Returns:
        windows: (B*num_windows, T*window_size*window_size, C)
    """
    B, T, H, W, C = x.shape
    x = x.view(B, T, H // window_size[0], window_size[0], W // window_size[1],
               window_size[1], C)
    windows = x.permute(0, 2, 4, 1, 3, 5, 6).contiguous().view(
        -1, T * window_size[0] * window_size[1], C)
    return windows


def window_partition_noreshape(x, window_size):
    """
    Args:
        x: shape is (B, T, H, W, C)
        window_size (tuple[int]): window size
    Returns:
        windows: (B, num_windows_h, num_windows_w, T, window_size, window_size, C)
    """
    B, T, H, W, C = x.shape
    x = x.view(B, T, H // window_size[0], window_size[0], W // window_size[1],
               window_size[1], C)
    windows = x.permute(0, 2, 4, 1, 3, 5, 6).contiguous()
    return windows


def window_reverse(windows, window_size, T, H, W):
    """
    Args:
        windows: shape is (num_windows*B, T, window_size, window_size, C)
        window_size (tuple[int]): Window size
        T (int): Temporal length of video
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, T, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size[0] / window_size[1]))
    x = windows.view(B, H // window_size[0], W // window_size[1], T,
                     window_size[0], window_size[1], -1)
    x = x.permute(0, 3, 1, 4, 2, 5, 6).contiguous().view(B, T, H, W, -1)
    return x


class TemporalLePEAttention(nn.Module):
    """CSWin attention.
        Revised by Hao:
            1. Able to compute attention with non-square input.
            2. Extend CSWin to Temporal-CSWin.
            3. Enhance CSWin with global window attention with pooling and focal.
            temporal (bool): It True, extend CSWin to Temporal CSWin
            cs_focal (bool): It True, extend CSWin with global window attention with pooling and focal
            cs_sw (bool): If True, extend CSWin with sliding window logic."""
    def __init__(self, dim, resolution, idx, split_size=7, dim_out=None, num_heads=8, attn_drop=0., proj_drop=0.,
                 qk_scale=None, temporal=False, cs_focal=False, cs_focal_v2=False, cs_sw=False,
                 pool_strip=False, pool_sw=2):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out or dim
        self.resolution = resolution
        self.split_size = split_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        if idx == -1:
            H_sp, W_sp = self.resolution[0], self.resolution[1]
        elif idx == 0:
            H_sp, W_sp = self.resolution[0], self.split_size
        elif idx == 1:
            W_sp, H_sp = self.resolution[1], self.split_size
        else:
            print("ERROR MODE", idx)
            exit(0)
        self.H_sp = H_sp
        self.W_sp = W_sp
        stride = 1
        self.get_v = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)
        self.idx = idx

        self.attn_drop = nn.Dropout(attn_drop)
        self.temporal = temporal
        self.cs_focal = cs_focal
        self.cs_focal_v2 = cs_focal_v2      # if true, the sliding window will has same direction of pooled feat
        self.cs_sw = cs_sw
        self.pool_strip = pool_strip
        self.pool_sw = pool_sw

        # 如果使用不同宽度的条带来增强1的窗口，需要池化层
        # 也可以池化/反池化到宽度不为1的窗口，和主窗口的宽度(split_size)、数量一致即可
        if self.pool_strip:
            self.strip_pooling = nn.ModuleList()
            # 对于横向和纵向当然需要不同的pooling layer了
            self.strip_pooling.append(
                nn.Linear(pool_sw, self.split_size))
            self.strip_pooling[-1].weight.data.fill_(
                self.split_size / pool_sw)
            self.strip_pooling[-1].bias.data.fill_(0)

        # 暂时不给滑窗做mask了
        if self.cs_sw:
            # 滑窗需要用到mask挡住原本不是这个区域的特征(循环过来的特征)
            # 横向条带和纵向其实需要不同的滑窗方向以及mask
            # 如果条带宽度为1，滑窗是没有意义的，报错
            if self.split_size == 1:
                raise Exception('Slide Window ONLY work with strip width > 1')
            # 在两个方向上扩展的长度其实是一样的，而且每次只扩展一条直线上的特征
            self.expand_size = [H_sp // 2, W_sp // 2]  # 窗口大小除以2是拓展大小

        #     # H_sp等于纵向分辨率时，这个时候需要横向的滑窗，和横向的mask
        #     # get mask for rolled k and rolled v
        #     mask_l = torch.ones(H_sp,  W_sp)
        #     mask_l[:, :-self.expand_size[1]] = 0
        #     mask_u = torch.ones(H_sp,  W_sp)
        #     mask_u[:-self.expand_size[0], :] = 0
        #
        #     # mask_rolled = torch.stack((mask_l, mask_u), 0).flatten(0)
        #     # self.register_buffer("valid_ind_rolled",
        #     #                      mask_rolled.nonzero(as_tuple=False).view(-1))
        #     mask_rolled = torch.stack((mask_l, mask_u), 0).nonzero(as_tuple=False)
        #     self.register_buffer("valid_ind_rolled", mask_rolled)

        if self.cs_focal:
            # 用于池化记忆kv的层
            # 使用线性层池化
            if not self.cs_sw:
                # 池化到宽度为1
                self.pool_layers = nn.ModuleList()
                window_size_glo = [self.H_sp, self.W_sp]
                self.pool_layers.append(
                    nn.Linear(window_size_glo[0] * window_size_glo[1], 1))
                self.pool_layers[-1].weight.data.fill_(
                    1. / (window_size_glo[0] * window_size_glo[1]))
                self.pool_layers[-1].bias.data.fill_(0)
            else:
                # 池化到宽度为条带的宽度
                self.pool_layers = nn.ModuleList()
                window_size_glo = [self.H_sp, self.W_sp]
                self.pool_layers.append(
                    nn.Linear(window_size_glo[0] * window_size_glo[1], self.split_size))
                self.pool_layers[-1].weight.data.fill_(
                    self.split_size / (window_size_glo[0] * window_size_glo[1]))
                self.pool_layers[-1].bias.data.fill_(0)

            # 展开函数
            self.unfolds = nn.ModuleList()

            if not self.cs_focal_v2:
                # 使用与池化完特征垂直的滑窗，感受野局限于内部
                # build relative position bias between local patch and pooled windows
                if idx == 0:
                    # H_sp等于纵向分辨率时，纵向的步幅要+1防止翻倍计算全局attention
                    stride = [2, 1]
                elif idx == 1:
                    # 反之当横向的窗口大小等于横向分辨率时，横向的步幅要+1防止翻倍计算全局attention
                    stride = [1, 2]
                self.focal_window = [self.H_sp, self.W_sp]
                kernel_size = self.focal_window
                # define unfolding operations
                # 保证unfold后的kv尺寸和原来一样 变向等于是展开了kv
                self.unfolds += [
                    nn.Unfold(kernel_size=kernel_size,
                              stride=stride,
                              padding=tuple(i // 2 for i in kernel_size))
                ]
            else:
                # 第二个版本的focal cs win
                # 使用与池化完特征方向相同的滑窗，感受野扩展到非局部
                if self.split_size == 1:
                    # 条形窗口宽度为1
                    stride = 1
                    self.focal_window = [self.W_sp, self.H_sp]      # 刚好和原来的窗口相反
                    kernel_size = self.focal_window
                    if idx == 0:
                        # H_sp等于纵向分辨率时，考虑最后一个窗口需要pad H_sp-1, 注意padding是两边的
                        padding = [0, self.H_sp//2]
                    elif idx == 1:
                        # 反之当横向的窗口大小等于横向分辨率时，考虑最后一个窗口需要pad W_sp-1, 注意padding是两边的
                        padding = [self.W_sp//2, 0]
                else:
                    # 条形窗口宽度不为1
                    if not self.cs_sw:
                        # 此时池化完的特征宽度上也要padding，并且步幅必须为2
                        self.focal_window = [self.W_sp, self.H_sp]      # 刚好和原来的窗口相反
                        kernel_size = self.focal_window
                        if idx == 0:
                            # H_sp等于纵向分辨率时，考虑最后一个窗口需要pad H_sp-1, 注意padding是两边的
                            padding = [self.W_sp//2, self.H_sp//2]
                            stride = [2, 1]
                        elif idx == 1:
                            # 反之当横向的窗口大小等于横向分辨率时，考虑最后一个窗口需要pad W_sp-1, 注意padding是两边的
                            padding = [self.W_sp//2, self.H_sp//2]
                            stride = [1, 2]
                    else:
                        # 宽度上不需要padding因为会池化到和条带宽度相同
                        # 似乎和宽度为1的逻辑是一样的？
                        self.focal_window = [self.W_sp, self.H_sp]      # 刚好和原来的窗口相反
                        kernel_size = self.focal_window
                        if idx == 0:
                            # H_sp等于纵向分辨率时，考虑最后一个窗口需要pad H_sp-1, 注意padding是两边的
                            padding = [0, self.H_sp//2]
                            stride = [1, 1]
                        elif idx == 1:
                            # 反之当横向的窗口大小等于横向分辨率时，考虑最后一个窗口需要pad W_sp-1, 注意padding是两边的
                            padding = [self.W_sp//2, 0]
                            stride = [1, 1]

                # define unfolding operations
                # 保证unfold后的kv尺寸和原来一样 变向等于是展开了kv
                self.unfolds += [
                    nn.Unfold(kernel_size=kernel_size,
                              stride=stride,
                              padding=padding)
                ]

    def im2cswin(self, x, H_sp=None, W_sp=None):
        """
        H_sp: height of strip window size.
        W_sp: Width of strip window size.
        """
        B, N, C = x.shape

        # H = W = int(np.sqrt(N))
        H = self.resolution[0]
        W = self.resolution[1]

        if H_sp is None:
            # default manner
            H_sp = self.H_sp
            W_sp = self.W_sp

        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
        x = self.img2windows(x, H_sp, W_sp)
        x = x.reshape(-1, H_sp * W_sp, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        # x: [-1, head, H_sp*W_sp, C/head]
        return x

    def img2windows(self, img, H_sp, W_sp):
        """
        img: B C H W
        """
        B, C, H, W = img.shape
        img_reshape = img.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
        img_perm = img_reshape.permute(0, 2, 4, 3, 5, 1).contiguous().reshape(-1, H_sp * W_sp, C)
        return img_perm

    def get_lepe(self, x, func):
        B, N, C = x.shape

        # H = W = int(np.sqrt(N))
        H = self.resolution[0]
        W = self.resolution[1]

        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)

        H_sp, W_sp = self.H_sp, self.W_sp
        x = x.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous().reshape(-1, C, H_sp, W_sp)  ### B', C, H', W'

        lepe = func(x)  ### B', C, H', W'
        lepe = lepe.reshape(-1, self.num_heads, C // self.num_heads, H_sp * W_sp).permute(0, 1, 3, 2).contiguous()

        x = x.reshape(-1, self.num_heads, C // self.num_heads, self.H_sp * self.W_sp).permute(0, 1, 3, 2).contiguous()
        return x, lepe

    def windows2img(self, img_splits_hw, H_sp, W_sp, H, W):
        """
        img_splits_hw: B' H W C
        """
        B = int(img_splits_hw.shape[0] / (H * W / H_sp / W_sp))

        img = img_splits_hw.view(B, H // H_sp, W // W_sp, H_sp, W_sp, -1)
        img = img.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return img

    def im2cswin_temporal(self, x):
        B, T, H, W, C = x.shape
        x = x.permute(0, 4, 1, 2, 3).contiguous()   # B C T H W
        x = self.img2windows_temporal(x, self.H_sp, self.W_sp)  # B*H/H_sp*W/W_sp T*H_sp*W_sp C
        x = x.reshape(-1, T * self.H_sp * self.W_sp, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        return x

    def img2windows_temporal(self, img, H_sp, W_sp):
        """
        img: B C T H W
        """
        B, C, T, H, W = img.shape
        img_reshape = img.view(B, C, T, H // H_sp, H_sp, W // W_sp, W_sp)
        img_perm = img_reshape.permute(0, 3, 5, 2, 4, 6, 1).contiguous().reshape(-1, T * H_sp * W_sp, C)
        return img_perm

    def get_lepe_temporal(self, x, func):
        B, T, H, W, C = x.shape
        x = x.permute(0, 1, 4, 2, 3).contiguous().reshape(B * T, C, H, W)   # B*T C H W

        H_sp, W_sp = self.H_sp, self.W_sp
        x = x.view(B, T, C, H // H_sp, H_sp, W // W_sp, W_sp)
        x = x.permute(0, 3, 5, 1, 2, 4, 6).contiguous().reshape(-1, C, H_sp, W_sp)  # B'*T, C, H', W'

        lepe = func(x)  ### B'*T, C, H', W'
        lepe = lepe.reshape(B * H // H_sp * W // W_sp, T, self.num_heads, C // self.num_heads, H_sp * W_sp)\
            .permute(0, 2, 1, 4, 3).contiguous().reshape(-1, self.num_heads, T * H_sp * W_sp, C // self.num_heads)
        # lepe: B' head T*H_sp*W_sp C/head

        x = x.reshape(B * H // H_sp * W // W_sp, T, self.num_heads, C // self.num_heads, self.H_sp * self.W_sp)\
            .permute(0, 2, 1, 4, 3).contiguous().reshape(-1, self.num_heads, T * H_sp * W_sp, C // self.num_heads)
        # x: B' head T*Hsp*Wsp C/head
        return x, lepe

    def windows2img_temporal(self, img_splits_hw, T, H_sp, W_sp, H, W):
        """
        img_splits_hw: B' THW C
        img: B T H W C
        """
        B = int(img_splits_hw.shape[0] / (H * W / H_sp / W_sp))

        img = img_splits_hw.view(B, H // H_sp, W // W_sp, T, H_sp, W_sp, -1)
        img = img.permute(0, 3, 1, 4, 2, 5, 6).contiguous().view(B, T, H, W, -1)
        return img

    def forward(self, qkv):
        """
        q,k,v: B, T, H, W, C
        x: B L C
        """
        q, k, v = qkv[0], qkv[1], qkv[2]

        B, T, H, W, C = q.shape

        # 池化kv用于获得global att
        if self.cs_focal:
            nWh = H // self.H_sp    # 窗口数量
            nWw = W // self.W_sp

            # 改变kv形状->B, nWh, nWw, T, C, window_size_h*window_size_w
            k = k.reshape(B, T, nWh, self.H_sp, nWw, self.W_sp, C).permute(0, 2, 4, 1, 6, 3, 5).contiguous() \
                .reshape(B, nWh, nWw, T, C, self.H_sp * self.W_sp)
            v = v.reshape(B, T, nWh, self.H_sp, nWw, self.W_sp, C).permute(0, 2, 4, 1, 6, 3, 5).contiguous() \
                .reshape(B, nWh, nWw, T, C, self.H_sp * self.W_sp)

            # 池化kv
            k_pooled = self.pool_layers[0](k).flatten(-2)  # B, nWh, nWw, T, C
            v_pooled = self.pool_layers[0](v).flatten(-2)  # B, nWh, nWw, T, C

            if self.split_size == 1 or not self.cs_sw:
                # 转化池化后的kv到需要的shape
                k_pooled = k_pooled.permute(0, 3, 4, 1, 2).contiguous().reshape(B * T, C, nWh, nWw)  # B*T, C, nWh, nWw
                v_pooled = v_pooled.permute(0, 3, 4, 1, 2).contiguous().reshape(B * T, C, nWh, nWw)  # B*T, C, nWh, nWw
            else:
                # 条带宽度不为1时池化的形状也不同
                if self.idx == 0:
                    # 纵向条纹，池化完是横向的
                    k_pooled = k_pooled.permute(0, 3, 4, 1, 2).contiguous().reshape(B * T, C, nWh * self.split_size, nWw)  # B*T, C, nWh, nWw
                    v_pooled = v_pooled.permute(0, 3, 4, 1, 2).contiguous().reshape(B * T, C, nWh * self.split_size, nWw)  # B*T, C, nWh, nWw
                else:
                    # 横向条纹，池化完是纵向的
                    k_pooled = k_pooled.permute(0, 3, 4, 1, 2).contiguous().reshape(B * T, C, nWh, nWw * self.split_size)  # B*T, C, nWh, nWw
                    v_pooled = v_pooled.permute(0, 3, 4, 1, 2).contiguous().reshape(B * T, C, nWh, nWw * self.split_size)  # B*T, C, nWh, nWw

            # 恢复kv形状->B, T, H, W, C
            k = k.reshape(B, nWh, nWw, T, C, self.H_sp, self.W_sp).permute(0, 3, 1, 5, 2, 6, 4)\
                .contiguous().reshape(B, T, H, W, C)
            v = v.reshape(B, nWh, nWw, T, C, self.H_sp, self.W_sp).permute(0, 3, 1, 5, 2, 6, 4)\
                .contiguous().reshape(B, T, H, W, C)

        ### Img2Window
        if self.temporal:
            # 3D temporal cs win att
            q = self.im2cswin_temporal(q)
            k = self.im2cswin_temporal(k)
            v, lepe = self.get_lepe_temporal(v, self.get_v)
        else:
            # 2D cs win att
            # reshape qkv to [B*T H*W C]
            q = q.reshape(B * T, H * W, C)
            k = k.reshape(B * T, H * W, C)
            v = v.reshape(B * T, H * W, C)

            # 利用不同宽度的kv池化到当前宽度来增强kv，先获得不同宽度的滑窗捏
            if self.pool_strip:
                # 获得不同宽度的条带
                # 水平和竖直不同捏
                if self.idx == 0:
                    # H_sp等于纵向分辨率时, W相当于是条带宽度
                    k_large_strip = self.im2cswin(k, H_sp=self.H_sp, W_sp=self.pool_sw)
                    v_large_strip = self.im2cswin(v, H_sp=self.H_sp, W_sp=self.pool_sw)

                    # 池化大宽度kv
                    # 改变kv形状->-1, head, C/head, H_sp, Pool_W
                    k_large_strip = k_large_strip.permute(0, 1, 3, 2).contiguous()\
                        .reshape(-1, self.num_heads, C//self.num_heads, self.H_sp, self.pool_sw)
                    v_large_strip = v_large_strip.permute(0, 1, 3, 2).contiguous()\
                        .reshape(-1, self.num_heads, C // self.num_heads, self.H_sp, self.pool_sw)

                    # 池化->-1, head, C/head, H_sp, 1
                    k_large_strip = self.strip_pooling[0](k_large_strip)
                    v_large_strip = self.strip_pooling[0](v_large_strip)

                    # 池化后的kv恢复到原来的形状->-1, head, H_sp * self.split_size, C/head
                    k_large_strip = k_large_strip.view(-1, self.num_heads, C // self.num_heads, self.H_sp * self.split_size)\
                        .permute(0, 1, 3, 2).contiguous()
                    v_large_strip = v_large_strip.view(-1, self.num_heads, C // self.num_heads, self.H_sp * self.split_size) \
                        .permute(0, 1, 3, 2).contiguous()

                    # 对于副窗口宽度为1的情况，把反池化的窗口信息放到窗口大小维度(亚像素特征)
                    if self.pool_sw == 1:
                        k_large_strip = k_large_strip.reshape(B * T * (W // self.pool_sw) // self.split_size,
                                                              self.split_size,
                                                              H // self.H_sp, self.num_heads,
                                                              self.H_sp * self.split_size,
                                                              C // self.num_heads).permute(0, 2, 3, 1, 4,
                                                                                           5).contiguous() \
                            .view(-1, self.num_heads, self.H_sp * self.split_size * self.split_size,
                                  C // self.num_heads)
                        v_large_strip = v_large_strip.reshape(B * T * (W // self.pool_sw) // self.split_size,
                                                              self.split_size,
                                                              H // self.H_sp, self.num_heads,
                                                              self.H_sp * self.split_size,
                                                              C // self.num_heads).permute(0, 2, 3, 1, 4,
                                                                                           5).contiguous() \
                            .view(-1, self.num_heads, self.H_sp * self.split_size * self.split_size,
                                  C // self.num_heads)

                    # 获得滑窗的大宽度kv，保证数量一样
                    # 只有副窗口宽度为2时适用以下的滑窗逻辑，副窗口宽度为1时不需要滑窗，只有主窗口需要滑窗保证数量一样
                    if self.pool_sw == 2:
                        # 先把token都移动了，然后再划分窗口，正值：向下/右移动，负值：向上/左移动
                        (k_l, v_l) = map(
                            lambda t: torch.roll(t,
                                                 shifts=(0, -self.pool_sw // 2),
                                                 dims=(1, 2)), (k, v))

                        # 划分一下窗口捏
                        # k_l: [-1, head, H_sp*W_sp, C/head]
                        k_l = self.im2cswin(k_l.reshape(B * T, H * W, C), H_sp=self.H_sp, W_sp=self.pool_sw)
                        v_l = self.im2cswin(v_l.reshape(B * T, H * W, C), H_sp=self.H_sp, W_sp=self.pool_sw)

                        k_l = k_l.permute(0, 1, 3, 2).contiguous() \
                            .reshape(-1, self.num_heads, C // self.num_heads, self.H_sp, self.pool_sw)
                        v_l = v_l.permute(0, 1, 3, 2).contiguous() \
                            .reshape(-1, self.num_heads, C // self.num_heads, self.H_sp, self.pool_sw)

                        k_l = self.strip_pooling[0](k_l)
                        v_l = self.strip_pooling[0](v_l)

                        k_l = k_l.view(-1, self.num_heads, C // self.num_heads, self.H_sp * self.split_size) \
                            .permute(0, 1, 3, 2).contiguous()
                        v_l = v_l.view(-1, self.num_heads, C // self.num_heads, self.H_sp * self.split_size) \
                            .permute(0, 1, 3, 2).contiguous()

                        # 汇总原本的大窗口和滑窗的窗口
                        k_large_strip = torch.cat((k_large_strip, k_l), dim=0)
                        v_large_strip = torch.cat((v_large_strip, v_l), dim=0)

                    # 只有副窗口宽度为4时适用以下的滑窗逻辑
                    if self.pool_sw == 4:
                        # 先把token都移动了，然后再划分窗口，正值：向下/右移动，负值：向上/左移动

                        for sw_idx in range(0, self.pool_sw - 1):
                            (k_l, v_l) = map(
                                lambda t: torch.roll(t,
                                                     shifts=(0, -(sw_idx + 1)),
                                                     dims=(1, 2)), (k, v))

                            # 划分一下窗口捏
                            # k_l: [-1, head, H_sp*W_sp, C/head]
                            k_l = self.im2cswin(k_l.reshape(B * T, H * W, C), H_sp=self.H_sp, W_sp=self.pool_sw)
                            v_l = self.im2cswin(v_l.reshape(B * T, H * W, C), H_sp=self.H_sp, W_sp=self.pool_sw)

                            k_l = k_l.permute(0, 1, 3, 2).contiguous() \
                                .reshape(-1, self.num_heads, C // self.num_heads, self.H_sp, self.pool_sw)
                            v_l = v_l.permute(0, 1, 3, 2).contiguous() \
                                .reshape(-1, self.num_heads, C // self.num_heads, self.H_sp, self.pool_sw)

                            k_l = self.strip_pooling[0](k_l)
                            v_l = self.strip_pooling[0](v_l)

                            k_l = k_l.view(-1, self.num_heads, C // self.num_heads, self.H_sp * self.split_size) \
                                .permute(0, 1, 3, 2).contiguous()
                            v_l = v_l.view(-1, self.num_heads, C // self.num_heads, self.H_sp * self.split_size) \
                                .permute(0, 1, 3, 2).contiguous()

                            # 汇总原本的大窗口和滑窗的窗口
                            k_large_strip = torch.cat((k_large_strip, k_l), dim=0)
                            v_large_strip = torch.cat((v_large_strip, v_l), dim=0)

                        # 把信息放到窗口大小维度, 因为也有滑窗所以不需要除以副窗口的宽度
                        k_large_strip = k_large_strip.reshape(B * T * W // self.split_size,
                                                              self.split_size,
                                                              H // self.H_sp, self.num_heads,
                                                              self.H_sp * self.split_size,
                                                              C // self.num_heads).permute(0, 2, 3, 1, 4,
                                                                                           5).contiguous() \
                            .view(-1, self.num_heads, self.H_sp * self.split_size * self.split_size,
                                  C // self.num_heads)
                        v_large_strip = v_large_strip.reshape(B * T * W // self.split_size,
                                                              self.split_size,
                                                              H // self.H_sp, self.num_heads,
                                                              self.H_sp * self.split_size,
                                                              C // self.num_heads).permute(0, 2, 3, 1, 4,
                                                                                           5).contiguous() \
                            .view(-1, self.num_heads, self.H_sp * self.split_size * self.split_size,
                                  C // self.num_heads)

                elif self.idx == 1:
                    # W_sp等于纵向分辨率时, H相当于是条带宽度
                    k_large_strip = self.im2cswin(k, H_sp=self.pool_sw, W_sp=self.W_sp)
                    v_large_strip = self.im2cswin(v, H_sp=self.pool_sw, W_sp=self.W_sp)

                    # 池化大宽度kv
                    # 改变kv形状->-1, head, C/head, W_sp, Pool_H
                    k_large_strip = k_large_strip.permute(0, 1, 3, 2).contiguous() \
                        .reshape(-1, self.num_heads, C // self.num_heads, self.W_sp, self.pool_sw)
                    v_large_strip = v_large_strip.permute(0, 1, 3, 2).contiguous() \
                        .reshape(-1, self.num_heads, C // self.num_heads, self.W_sp, self.pool_sw)

                    # 池化->-1, head, C/head, W_sp, 1
                    k_large_strip = self.strip_pooling[0](k_large_strip)
                    v_large_strip = self.strip_pooling[0](v_large_strip)

                    # 池化后的kv恢复到原来的形状->-1, head, self.split_size * W_sp, C/head
                    k_large_strip = k_large_strip.view(-1, self.num_heads, C // self.num_heads, self.split_size * self.W_sp) \
                        .permute(0, 1, 3, 2).contiguous()
                    v_large_strip = v_large_strip.view(-1, self.num_heads, C // self.num_heads, self.split_size * self.W_sp) \
                        .permute(0, 1, 3, 2).contiguous()

                    # 对于副窗口宽度为1的情况，把反池化的窗口信息放到窗口大小维度(亚像素特征)
                    if self.pool_sw == 1:
                        k_large_strip = k_large_strip.reshape(B * T * (W // self.W_sp) * (H // self.pool_sw) // self.split_size,
                                                              self.split_size, self.num_heads,
                                                              self.split_size * self.W_sp,
                                                              C // self.num_heads).permute(0, 2, 3, 1, 4).contiguous() \
                            .view(-1, self.num_heads, self.split_size * self.W_sp * self.split_size,
                                  C // self.num_heads)
                        v_large_strip = v_large_strip.reshape(B * T * (W // self.W_sp) * (H // self.pool_sw) // self.split_size,
                                                              self.split_size, self.num_heads,
                                                              self.split_size * self.W_sp,
                                                              C // self.num_heads).permute(0, 2, 3, 1, 4).contiguous() \
                            .view(-1, self.num_heads, self.split_size * self.W_sp * self.split_size,
                                  C // self.num_heads)

                    # 获得滑窗的大宽度kv，保证数量一样
                    # 只有副窗口宽度为2时适用以下的滑窗逻辑，副窗口宽度为1时不需要滑窗，只有主窗口需要滑窗保证数量一样
                    if self.pool_sw == 2:
                        # 先把token都移动了，然后再划分窗口，正值：向下/右移动，负值：向上/左移动
                        (k_u, v_u) = map(
                            lambda t: torch.roll(t,
                                                 shifts=(-self.pool_sw // 2, 0),
                                                 dims=(1, 2)), (k, v))

                        # 划分一下窗口捏
                        # k_u: [-1, head, H_sp*W_sp, C/head]
                        k_u = self.im2cswin(k_u.reshape(B * T, H * W, C), H_sp=self.pool_sw, W_sp=self.W_sp)
                        v_u = self.im2cswin(v_u.reshape(B * T, H * W, C), H_sp=self.pool_sw, W_sp=self.W_sp)

                        k_u = k_u.permute(0, 1, 3, 2).contiguous() \
                            .reshape(-1, self.num_heads, C // self.num_heads, self.W_sp, self.pool_sw)
                        v_u = v_u.permute(0, 1, 3, 2).contiguous() \
                            .reshape(-1, self.num_heads, C // self.num_heads, self.W_sp, self.pool_sw)

                        k_u = self.strip_pooling[0](k_u)
                        v_u = self.strip_pooling[0](v_u)

                        k_u = k_u.view(-1, self.num_heads, C // self.num_heads, self.split_size * self.W_sp) \
                            .permute(0, 1, 3, 2).contiguous()
                        v_u = v_u.view(-1, self.num_heads, C // self.num_heads, self.split_size * self.W_sp) \
                            .permute(0, 1, 3, 2).contiguous()

                        # 汇总原本的大窗口和滑窗的窗口 窗口数量汇总所以在0维度
                        k_large_strip = torch.cat((k_large_strip, k_u), dim=0)
                        v_large_strip = torch.cat((v_large_strip, v_u), dim=0)

                    # 只有副窗口宽度为4时适用以下的滑窗逻辑
                    if self.pool_sw == 4:
                        # 先把token都移动了，然后再划分窗口，正值：向下/右移动，负值：向上/左移动

                        for sw_idx in range(0, self.pool_sw - 1):
                            (k_u, v_u) = map(
                                lambda t: torch.roll(t,
                                                     shifts=(-self.pool_sw // 2, 0),
                                                     dims=(1, 2)), (k, v))

                            # 划分一下窗口捏
                            # k_u: [-1, head, H_sp*W_sp, C/head]
                            k_u = self.im2cswin(k_u.reshape(B * T, H * W, C), H_sp=self.pool_sw, W_sp=self.W_sp)
                            v_u = self.im2cswin(v_u.reshape(B * T, H * W, C), H_sp=self.pool_sw, W_sp=self.W_sp)

                            k_u = k_u.permute(0, 1, 3, 2).contiguous() \
                                .reshape(-1, self.num_heads, C // self.num_heads, self.W_sp, self.pool_sw)
                            v_u = v_u.permute(0, 1, 3, 2).contiguous() \
                                .reshape(-1, self.num_heads, C // self.num_heads, self.W_sp, self.pool_sw)

                            k_u = self.strip_pooling[0](k_u)
                            v_u = self.strip_pooling[0](v_u)

                            k_u = k_u.view(-1, self.num_heads, C // self.num_heads, self.split_size * self.W_sp) \
                                .permute(0, 1, 3, 2).contiguous()
                            v_u = v_u.view(-1, self.num_heads, C // self.num_heads, self.split_size * self.W_sp) \
                                .permute(0, 1, 3, 2).contiguous()

                            # 汇总原本的大窗口和滑窗的窗口 窗口数量汇总所以在0维度
                            k_large_strip = torch.cat((k_large_strip, k_u), dim=0)
                            v_large_strip = torch.cat((v_large_strip, v_u), dim=0)

                        # 把信息放到窗口大小维度
                        k_large_strip = k_large_strip.reshape(
                            B * T * (W // self.W_sp) * H // self.split_size,
                            self.split_size, self.num_heads,
                            self.split_size * self.W_sp,
                            C // self.num_heads).permute(0, 2, 3, 1, 4).contiguous() \
                            .view(-1, self.num_heads, self.split_size * self.W_sp * self.split_size,
                                  C // self.num_heads)
                        v_large_strip = v_large_strip.reshape(
                            B * T * (W // self.W_sp) * H // self.split_size,
                            self.split_size, self.num_heads,
                            self.split_size * self.W_sp,
                            C // self.num_heads).permute(0, 2, 3, 1, 4).contiguous() \
                            .view(-1, self.num_heads, self.split_size * self.W_sp * self.split_size,
                                  C // self.num_heads)

            q = self.im2cswin(q)
            k = self.im2cswin(k)
            v, lepe = self.get_lepe(v, self.get_v)

        # 滑窗逻辑
        if self.cs_sw:
            if self.idx == 0:
                # H_sp等于纵向分辨率时，这个时候需要横向的滑窗
                # 直接向左移动，在条带宽度为2时，向左或者向右滑动是一样的, 也就是在宽度方向上移动

                # 先恢复回token
                # B*T*H/H_sp*W/W_sp head H_sp*W_sp C/head -> B*T*H/H_sp*W/W_sp H_sp*W_sp C
                k_l = k.transpose(1, 2).reshape(-1, self.H_sp * self.W_sp, C)
                k_l = self.windows2img(k_l, self.H_sp, self.W_sp, H, W)  # B*T H W C
                v_l = v.transpose(1, 2).reshape(-1, self.H_sp * self.W_sp, C)
                v_l = self.windows2img(v_l, self.H_sp, self.W_sp, H, W)  # B*T H W C

                # 先把token都移动了，然后再划分窗口，正值：向下/右移动，负值：向上/左移动
                (k_l, v_l) = map(
                    lambda t: torch.roll(t,
                                         shifts=(0, -self.expand_size[1]),
                                         dims=(1, 2)), (k_l, v_l))

                # 划分一下窗口捏
                # k_l: [-1, head, H_sp*W_sp, C/head]
                k_l = self.im2cswin(k_l.reshape(B * T, H * W, C))
                v_l = self.im2cswin(v_l.reshape(B * T, H * W, C))

                # # 转换k_l, v_l到窗口个数的格式
                # k_l = k_l.reshape(B, T, H // self.H_sp, W // self.W_sp, self.num_heads, self.H_sp * self.W_sp, C // self.num_heads)\
                #     .permute(0, 5, 4, 1, 2, 3, 6).contiguous()\
                #     .reshape(B * self.H_sp * self.W_sp, self.num_heads, T, H // self.H_sp * W // self.W_sp, C // self.num_heads)
                # v_l = v_l.reshape(B, T, H // self.H_sp, W // self.W_sp, self.num_heads, self.H_sp * self.W_sp,
                #                   C // self.num_heads) \
                #     .permute(0, 5, 4, 1, 2, 3, 6).contiguous() \
                #     .reshape(B * self.H_sp * self.W_sp, self.num_heads, T, H // self.H_sp * W // self.W_sp, C // self.num_heads)

                # # mask掉同一个窗口内含有非局部特征的这些窗口
                # k_rolled = k_l[:, :, :, self.valid_ind_rolled[0]]
                # v_rolled = v_l[:, :, :, self.valid_ind_rolled[0]]
                k_rolled = k_l
                v_rolled = v_l
            elif self.idx == 1:
                # 当横向的窗口大小等于横向分辨率时，这个时候需要竖直方向的位移和滑窗
                # 直接向上移动，在条带宽度为2时，向上或者向下滑动是一样的

                # 先恢复回token
                # B*T*H/H_sp*W/W_sp head H_sp*W_sp C/head -> B*T*H/H_sp*W/W_sp H_sp*W_sp C
                k_u = k.transpose(1, 2).reshape(-1, self.H_sp * self.W_sp, C)
                k_u = self.windows2img(k_u, self.H_sp, self.W_sp, H, W)  # B*T H W C
                v_u = v.transpose(1, 2).reshape(-1, self.H_sp * self.W_sp, C)
                v_u = self.windows2img(v_u, self.H_sp, self.W_sp, H, W)  # B*T H W C

                # 先把token都移动了，然后再划分窗口，正值：向下/右移动，负值：向上/左移动
                (k_u, v_u) = map(
                    lambda t: torch.roll(t,
                                         shifts=(-self.expand_size[0], 0),
                                         dims=(1, 2)), (k_u, v_u))

                # 划分一下窗口捏
                # k_l: [-1, head, H_sp*W_sp, C/head]
                k_u = self.im2cswin(k_u.reshape(B * T, H * W, C))
                v_u = self.im2cswin(v_u.reshape(B * T, H * W, C))

                # # 挡住没用的窗口，因为有的元素是循环的。
                # k_rolled = k_u[:, :, self.valid_ind_rolled[1]]
                # v_rolled = v_u[:, :, self.valid_ind_rolled[1]]
                k_rolled = k_u
                v_rolled = v_u

        # 利用池化kv增强kv
        if self.cs_focal:
            if self.temporal:
                # 时间也展开(其实一样因为时间窗口是1)
                (k_pooled, v_pooled) = map(
                    lambda t: self.unfolds[0]
                    (t).view(B, T, C, self.unfolds[0].kernel_size[0], self.
                             unfolds[0].kernel_size[1], -1)
                        .permute(0, 5, 1, 3, 4, 2).contiguous().view(
                        -1, T * self.unfolds[0].kernel_size[0] * self.unfolds[
                            0].kernel_size[1], self.num_heads, C // self.
                               num_heads).permute(0, 2, 1, 3).contiguous(),
                    # (B x (nWh*nWw)) x nHeads x (T x unfold_wsize x unfold_wsize) x C/head
                    (k_pooled, v_pooled))

                if self.cs_focal_v2:
                    # 因为两侧对称的padding会导致unfold多一个滑窗
                    if self.idx == 0:
                        # 丢掉竖直方向上最后一个
                        k_pooled = k_pooled.view(
                            B, nWh, -1, self.num_heads, T * self.unfolds[0].kernel_size[0]
                            * self.unfolds[0].kernel_size[1], C // self.num_heads)[:, :, :-1, :, :, :, :] \
                            .reshape(-1, self.num_heads, T * self.unfolds[0].kernel_size[0] * self.unfolds[0]
                                     .kernel_size[1], C // self.num_heads)      # drop last
                        v_pooled = v_pooled.view(
                            B, nWh, -1, self.num_heads, T * self.unfolds[0].kernel_size[0]
                            * self.unfolds[0].kernel_size[1], C // self.num_heads)[:, :, :-1, :, :, :, :] \
                            .reshape(-1, self.num_heads, T * self.unfolds[0].kernel_size[0] * self.unfolds[0]
                                     .kernel_size[1], C // self.num_heads)      # drop last
                    elif self.idx == 1:
                        # 丢掉水平方向上最后一个
                        k_pooled = k_pooled.view(
                            B, -1, nWw, self.num_heads, T * self.unfolds[0].kernel_size[0]
                            * self.unfolds[0].kernel_size[1], C // self.num_heads)[:, :-1, :, :, :, :, :] \
                            .reshape(-1, self.num_heads, T * self.unfolds[0].kernel_size[0] * self.unfolds[0]
                                     .kernel_size[1], C // self.num_heads)     # drop last
                        v_pooled = v_pooled.view(
                            B, -1, nWw, self.num_heads, T * self.unfolds[0].kernel_size[0]
                            * self.unfolds[0].kernel_size[1], C // self.num_heads)[:, :-1, :, :, :, :, :] \
                            .reshape(-1, self.num_heads, T *  self.unfolds[0].kernel_size[0] * self.unfolds[0]
                                     .kernel_size[1], C // self.num_heads)  # drop last

            else:
                # 空间展开
                (k_pooled, v_pooled) = map(
                    lambda t: self.unfolds[0]
                    (t).view(B * T, C, self.unfolds[0].kernel_size[0], self.
                             unfolds[0].kernel_size[1], -1)
                        .permute(0, 4, 2, 3, 1).contiguous().view(
                        -1, self.unfolds[0].kernel_size[0] * self.unfolds[
                            0].kernel_size[1], self.num_heads, C // self.
                            num_heads).permute(0, 2, 1, 3).contiguous(),
                    # (B x T x (nWh*nWw)) x nHeads x (unfold_wsize x unfold_wsize) x C/head
                    (k_pooled, v_pooled))

                if self.cs_focal_v2:
                    # 因为两侧对称的padding会导致unfold多一个滑窗
                    if self.idx == 0:
                        # 丢掉竖直方向上最后一个
                        k_pooled = k_pooled.view(
                            B, T, nWh, -1, self.num_heads, self.unfolds[0].kernel_size[0]
                            * self.unfolds[0].kernel_size[1], C // self.num_heads)[:, :, :, :-1, :, :, :] \
                            .reshape(-1, self.num_heads, self.unfolds[0].kernel_size[0] * self.unfolds[0]
                                     .kernel_size[1], C // self.num_heads)      # drop last
                        v_pooled = v_pooled.view(
                            B, T, nWh, -1, self.num_heads, self.unfolds[0].kernel_size[0]
                            * self.unfolds[0].kernel_size[1], C // self.num_heads)[:, :, :, :-1, :, :, :] \
                            .reshape(-1, self.num_heads, self.unfolds[0].kernel_size[0] * self.unfolds[0]
                                     .kernel_size[1], C // self.num_heads)      # drop last
                    elif self.idx == 1:
                        # 丢掉水平方向上最后一个
                        k_pooled = k_pooled.view(
                            B, T, -1, nWw, self.num_heads, self.unfolds[0].kernel_size[0]
                            * self.unfolds[0].kernel_size[1], C // self.num_heads)[:, :, :-1, :, :, :, :] \
                            .reshape(-1, self.num_heads, self.unfolds[0].kernel_size[0] * self.unfolds[0]
                                     .kernel_size[1], C // self.num_heads)      # drop last
                        v_pooled = v_pooled.view(
                            B, T, -1, nWw, self.num_heads, self.unfolds[0].kernel_size[0]
                            * self.unfolds[0].kernel_size[1], C // self.num_heads)[:, :, :-1, :, :, :, :] \
                            .reshape(-1, self.num_heads, self.unfolds[0].kernel_size[0] * self.unfolds[0]
                                     .kernel_size[1], C // self.num_heads)      # drop last

            # 增强kv
            k = torch.cat((k, k_pooled), dim=2)
            v = torch.cat((v, v_pooled), dim=2)

        # 利用滑窗kv增强kv
        if self.cs_sw:
            k = torch.cat((k, k_rolled), dim=2)
            v = torch.cat((v, v_rolled), dim=2)

        # 利用不同宽度的kv池化到当前宽度来增强kv
        if self.pool_strip:
            k = torch.cat((k, k_large_strip), dim=2)
            v = torch.cat((v, v_large_strip), dim=2)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # B head N C @ B head C N --> B head N N
        attn = nn.functional.softmax(attn, dim=-1, dtype=attn.dtype)
        attn = self.attn_drop(attn)

        x = (attn @ v) + lepe

        ### Window2Img
        if self.temporal:
            # 3D temporal cs win att
            x = x.transpose(1, 2).reshape(-1, T * self.H_sp * self.W_sp, C)
            x = self.windows2img_temporal(x, T, self.H_sp, self.W_sp, H, W).view(B, -1, C)  # B T*H*W C
        else:
            # 2D cs win att
            # B*T*H/H_sp*W/W_sp head H_sp*W_sp C/head -> B*T*H/H_sp*W/W_sp H_sp*W_sp C
            x = x.transpose(1, 2).reshape(-1, self.H_sp * self.W_sp, C)
            x = self.windows2img(x, self.H_sp, self.W_sp, H, W).view(B, -1, C)  # B T*H*W C

            # recover qkv to [B, T, H, W, C] -> NOT NEED
            # reshape x to [B, T, H, W, C] -> NOT NEED cm_x在和x融合前会reshape的

        return x


class CrossFocalAttention(nn.Module):
    """Cross Temporal focal window attention based on t-focal window attention of e2fgvi."""
    def __init__(self, dim, expand_size, window_size, focal_window,
                 focal_level, num_heads, qkv_bias, pool_method):

        super().__init__()
        self.dim = dim
        self.expand_size = expand_size
        self.window_size = window_size  # Wh, Ww
        self.pool_method = pool_method
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.focal_level = focal_level
        self.focal_window = focal_window

        if any(i > 0 for i in self.expand_size) and focal_level > 0:
            # get mask for rolled k and rolled v
            mask_tl = torch.ones(self.window_size[0], self.window_size[1])
            mask_tl[:-self.expand_size[0], :-self.expand_size[1]] = 0
            mask_tr = torch.ones(self.window_size[0], self.window_size[1])
            mask_tr[:-self.expand_size[0], self.expand_size[1]:] = 0
            mask_bl = torch.ones(self.window_size[0], self.window_size[1])
            mask_bl[self.expand_size[0]:, :-self.expand_size[1]] = 0
            mask_br = torch.ones(self.window_size[0], self.window_size[1])
            mask_br[self.expand_size[0]:, self.expand_size[1]:] = 0
            mask_rolled = torch.stack((mask_tl, mask_tr, mask_bl, mask_br),
                                      0).flatten(0)
            self.register_buffer("valid_ind_rolled",
                                 mask_rolled.nonzero(as_tuple=False).view(-1))

        if pool_method != "none" and focal_level > 1:
            self.unfolds = nn.ModuleList()

            # build relative position bias between local patch and pooled windows
            for k in range(focal_level - 1):
                stride = 2**k
                kernel_size = tuple(2 * (i // 2) + 2**k + (2**k - 1)
                                    for i in self.focal_window)
                # define unfolding operations
                self.unfolds += [
                    nn.Unfold(kernel_size=kernel_size,
                              stride=stride,
                              padding=tuple(i // 2 for i in kernel_size))
                ]

                # define unfolding index for focal_level > 0
                if k > 0:
                    mask = torch.zeros(kernel_size)
                    mask[(2**k) - 1:, (2**k) - 1:] = 1
                    self.register_buffer(
                        "valid_ind_unfold_{}".format(k),
                        mask.flatten(0).nonzero(as_tuple=False).view(-1))

        # qkv是现成的因此不需要重新编码
        # self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.proj = nn.Linear(dim, dim)

        self.softmax = nn.Softmax(dim=-1)

        # 用于池化记忆kv的层
        self.pool_layers = nn.ModuleList()
        if self.pool_method != "none":
            # for k in range(self.focal_level - 1):
            # focal level=2, k=0
            window_size_glo = tuple(
                math.floor(i) for i in self.window_size)
            self.pool_layers.append(
                nn.Linear(window_size_glo[0] * window_size_glo[1], 1))
            self.pool_layers[-1].weight.data.fill_(
                1. / (window_size_glo[0] * window_size_glo[1]))
            self.pool_layers[-1].bias.data.fill_(0)

    def forward(self, qkv, mask_all=None):
        """
        Args:
            x: input qkv with shape of (3, B, T, Wh, Ww, C) from different modality
            mask: (0/-inf) mask with shape of (num_windows, T*Wh*Ww, T*Wh*Ww) or None

            output: (nW*B, Wh*Ww, C)
        """
        q, k, v = qkv[0], qkv[1], qkv[2]  # B, T, nH, nW, C
        B, T, nH, nW, C = q.shape   # nH/W是多少个token；nWh/w是多少个window
        nWh = nH // self.window_size[0]
        nWw = nW // self.window_size[1]

        # 生成池化的qkv代替原来从池化的特征x_all[1]中编码全局(window内池化)qkv的操作, 注意q的形状没有改变过
        # 首先实现直接池化获得池化qkv的方法
        # x_windows_noreshape = x_windows_noreshape.view(
        #     B, nWh, nWw, T, window_size_glo[0] * window_size_glo[1],
        #     C).transpose(4, 5)  # B, nWh, nWw, T, C, window_size_h*window_size_w
        # x_windows_pooled = self.pool_layers[k](
        #     x_windows_noreshape).flatten(-2)  # B, nWh, nWw, T, C | window被池化聚合

        # 改变kv形状->B, nWh, nWw, T, C, window_size_h*window_size_w
        k = k.reshape(B, T, nWh, self.window_size[0], nWw, self.window_size[1], C).permute(0, 2, 4, 1, 6, 3, 5) \
            .reshape(B, nWh, nWw, T, C, self.window_size[0] * self.window_size[1]).contiguous()
        v = v.reshape(B, T, nWh, self.window_size[0], nWw, self.window_size[1], C).permute(0, 2, 4, 1, 6, 3, 5) \
            .reshape(B, nWh, nWw, T, C, self.window_size[0] * self.window_size[1]).contiguous()

        # 池化kv
        k_pooled_k = self.pool_layers[0](k).flatten(-2)  # B, nWh, nWw, T, C
        v_pooled_k = self.pool_layers[0](v).flatten(-2)  # B, nWh, nWw, T, C

        # 转化池化后的kv到需要的shape
        k_pooled_k = k_pooled_k.permute(0, 3, 4, 1, 2).reshape(B * T, C, nWh, nWw).contiguous()  # B*T, C, nWh, nWw
        v_pooled_k = v_pooled_k.permute(0, 3, 4, 1, 2).reshape(B * T, C, nWh, nWw).contiguous()  # B*T, C, nWh, nWw

        # 恢复kv形状->B, T, nH, nW, C
        k = k.reshape(B, nWh, nWw, T, C, self.window_size[0], self.window_size[1]).permute(0, 3, 1, 5, 2, 6, 4) \
            .reshape(B, T, nH, nW, C).contiguous()
        v = v.reshape(B, nWh, nWw, T, C, self.window_size[0], self.window_size[1]).permute(0, 3, 1, 5, 2, 6, 4) \
            .reshape(B, T, nH, nW, C).contiguous()

        # partition q map
        (q_windows, k_windows, v_windows) = map(
            lambda t: window_partition(t, self.window_size).view(
                -1, T, self.window_size[0] * self.window_size[1], self.
                num_heads, C // self.num_heads).permute(0, 3, 1, 2, 4).
            contiguous().view(-1, self.num_heads, T * self.window_size[
                0] * self.window_size[1], C // self.num_heads), (q, k, v))
        # q(k/v)_windows shape : [16, 4, 225, 128] i.e. [B*nWh*nWw, head, T*H/nWh*H/nWw, C/head]

        if any(i > 0 for i in self.expand_size) and self.focal_level > 0:
            (k_tl, v_tl) = map(
                lambda t: torch.roll(t,
                                     shifts=(-self.expand_size[0], -self.
                                             expand_size[1]),
                                     dims=(2, 3)), (k, v))
            (k_tr, v_tr) = map(
                lambda t: torch.roll(t,
                                     shifts=(-self.expand_size[0], self.
                                             expand_size[1]),
                                     dims=(2, 3)), (k, v))
            (k_bl, v_bl) = map(
                lambda t: torch.roll(t,
                                     shifts=(self.expand_size[0], -self.
                                             expand_size[1]),
                                     dims=(2, 3)), (k, v))
            (k_br, v_br) = map(
                lambda t: torch.roll(t,
                                     shifts=(self.expand_size[0], self.
                                             expand_size[1]),
                                     dims=(2, 3)), (k, v))
            # k_tl.shape=k.shape k_tl_windows=[B*nWh*nWw, T, H/nWh*H/nWw, head, C/head]
            (k_tl_windows, k_tr_windows, k_bl_windows, k_br_windows) = map(
                lambda t: window_partition(t, self.window_size).view(
                    -1, T, self.window_size[0] * self.window_size[1], self.
                    num_heads, C // self.num_heads), (k_tl, k_tr, k_bl, k_br))
            (v_tl_windows, v_tr_windows, v_bl_windows, v_br_windows) = map(
                lambda t: window_partition(t, self.window_size).view(
                    -1, T, self.window_size[0] * self.window_size[1], self.
                    num_heads, C // self.num_heads), (v_tl, v_tr, v_bl, v_br))
            k_rolled = torch.cat(
                (k_tl_windows, k_tr_windows, k_bl_windows, k_br_windows),
                2).permute(0, 3, 1, 2, 4).contiguous()      # k_rolled=[B*nWh*nWw, head, T, 4*H/nWh*H/nWw, C/head]
            v_rolled = torch.cat(
                (v_tl_windows, v_tr_windows, v_bl_windows, v_br_windows),
                2).permute(0, 3, 1, 2, 4).contiguous()

            # mask out tokens in current window
            k_rolled = k_rolled[:, :, :, self.valid_ind_rolled]     # [B*nWh*nWw, head, T, 4*H/nWh*H/nWw-60, C/head]
            v_rolled = v_rolled[:, :, :, self.valid_ind_rolled]
            temp_N = k_rolled.shape[3]
            k_rolled = k_rolled.view(-1, self.num_heads, T * temp_N,
                                     C // self.num_heads)
            v_rolled = v_rolled.view(-1, self.num_heads, T * temp_N,
                                     C // self.num_heads)
            k_rolled = torch.cat((k_windows, k_rolled), 2)
            v_rolled = torch.cat((v_windows, v_rolled), 2)
        else:
            k_rolled = k_windows
            v_rolled = v_windows

        # q(k/v)_windows shape : [16, 4, 225, 128]
        # k_rolled.shape : [16, 4, 5, 165, 128]
        # ideal expanded window size 153 ((5+2*2)*(9+2*4))
        # k_windows=45 expand_window=108 overlap_window=12 (since expand_size < window_size / 2)

        if self.pool_method != "none" and self.focal_level > 1:
            k_pooled = []
            v_pooled = []
            for k in range(self.focal_level - 1):
                stride = 2**k

                # # B, T, nWh, nWw, C
                # x_window_pooled = x_all[k + 1].permute(0, 3, 1, 2,
                #                                        4).contiguous()

                # nWh, nWw = x_window_pooled.shape[2:4]

                # generate mask for pooled windows, 这里的.new创建的内容和原来的无关, 只要个形状罢了
                # mask = x_window_pooled.new(T, nWh, nWw).fill_(1)
                mask = k_pooled_k.reshape(B, T, C, nWh, nWw).permute(0, 1, 3, 4, 2).new(T, nWh, nWw).fill_(1)

                # unfold mask: [nWh*nWw//s//s, k*k, 1]
                unfolded_mask = self.unfolds[k](mask.unsqueeze(1)).view(
                    1, T, self.unfolds[k].kernel_size[0], self.unfolds[k].kernel_size[1], -1).permute(4, 1, 2, 3, 0).contiguous().\
                    view(nWh*nWw // stride // stride, -1, 1)

                if k > 0:
                    valid_ind_unfold_k = getattr(
                        self, "valid_ind_unfold_{}".format(k))
                    unfolded_mask = unfolded_mask[:, valid_ind_unfold_k]

                x_window_masks = unfolded_mask.flatten(1).unsqueeze(0)
                x_window_masks = x_window_masks.masked_fill(
                    x_window_masks == 0,
                    float(-100.0)).masked_fill(x_window_masks > 0, float(0.0))
                mask_all[k + 1] = x_window_masks

                # # qkv是现成的，不需要重新编码
                # # generate k and v for pooled windows
                # qkv_pooled = self.qkv(x_window_pooled).reshape(
                #     B, T, nWh, nWw, 3, C).permute(4, 0, 1, 5, 2,
                #                                   3).view(3, -1, C, nWh,
                #                                           nWw).contiguous()
                # # B*T, C, nWh, nWw
                # k_pooled_k, v_pooled_k = qkv_pooled[1], qkv_pooled[2]

                # k_pooled_k shape: [5, 512, 4, 4]
                # self.unfolds[k](k_pooled_k) shape: [5, 23040 (512 * 5 * 9 ), 16]

                (k_pooled_k, v_pooled_k) = map(
                    lambda t: self.unfolds[k]
                    (t).view(B, T, C, self.unfolds[k].kernel_size[0], self.
                             unfolds[k].kernel_size[1], -1)
                    .permute(0, 5, 1, 3, 4, 2).contiguous().view(
                        -1, T, self.unfolds[k].kernel_size[0] * self.unfolds[
                            k].kernel_size[1], self.num_heads, C // self.
                        num_heads).permute(0, 3, 1, 2, 4).contiguous(),
                    # (B x (nH*nW)) x nHeads x T x (unfold_wsize x unfold_wsize) x head_dim
                    (k_pooled_k, v_pooled_k))
                # k_pooled_k shape : [16, 4, 5, 45, 128]

                # select valid unfolding index
                if k > 0:
                    (k_pooled_k, v_pooled_k) = map(
                        lambda t: t[:, :, :, valid_ind_unfold_k],
                        (k_pooled_k, v_pooled_k))

                k_pooled_k = k_pooled_k.view(
                    -1, self.num_heads, T * self.unfolds[k].kernel_size[0] *
                    self.unfolds[k].kernel_size[1], C // self.num_heads)
                v_pooled_k = v_pooled_k.view(
                    -1, self.num_heads, T * self.unfolds[k].kernel_size[0] *
                    self.unfolds[k].kernel_size[1], C // self.num_heads)

                k_pooled += [k_pooled_k]
                v_pooled += [v_pooled_k]

            # k_all (v_all) shape : [16, 4, 5 * 210, 128]
            k_all = torch.cat([k_rolled] + k_pooled, 2)
            v_all = torch.cat([v_rolled] + v_pooled, 2)
        else:
            k_all = k_rolled
            v_all = v_rolled

        N = k_all.shape[-2]
        q_windows = q_windows * self.scale
        # B*nW, nHead, T*window_size*window_size, T*focal_window_size*focal_window_size
        attn = (q_windows @ k_all.transpose(-2, -1))
        # T * 45
        window_area = T * self.window_size[0] * self.window_size[1]
        # T * 165
        window_area_rolled = k_rolled.shape[2]

        if self.pool_method != "none" and self.focal_level > 1:
            offset = window_area_rolled
            for k in range(self.focal_level - 1):
                # add attentional mask
                # mask_all[1] shape [1, 16, T * 45]

                bias = tuple((i + 2**k - 1) for i in self.focal_window)

                if mask_all[k + 1] is not None:
                    attn[:, :, :window_area, offset:(offset + (T*bias[0]*bias[1]))] = \
                        attn[:, :, :window_area, offset:(offset + (T*bias[0]*bias[1]))] + \
                        mask_all[k+1][:, :, None, None, :].repeat(
                            attn.shape[0] // mask_all[k+1].shape[1], 1, 1, 1, 1).view(-1, 1, 1, mask_all[k+1].shape[-1])

                offset += T * bias[0] * bias[1]

        if mask_all[0] is not None:
            nW = mask_all[0].shape[0]
            attn = attn.view(attn.shape[0] // nW, nW, self.num_heads,
                             window_area, N)
            attn[:, :, :, :, :
                 window_area] = attn[:, :, :, :, :window_area] + mask_all[0][
                     None, :, None, :, :]
            attn = attn.view(-1, self.num_heads, window_area, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        x = (attn @ v_all).transpose(1, 2).reshape(attn.shape[0], window_area,
                                                   C)
        x = self.proj(x)
        return x


class WindowAttention(nn.Module):
    """Temporal focal window attention
    """
    def __init__(self, dim, expand_size, window_size, focal_window,
                 focal_level, num_heads, qkv_bias, pool_method):

        super().__init__()
        self.dim = dim
        self.expand_size = expand_size
        self.window_size = window_size  # Wh, Ww
        self.pool_method = pool_method
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.focal_level = focal_level
        self.focal_window = focal_window

        if any(i > 0 for i in self.expand_size) and focal_level > 0:
            # get mask for rolled k and rolled v
            mask_tl = torch.ones(self.window_size[0], self.window_size[1])
            mask_tl[:-self.expand_size[0], :-self.expand_size[1]] = 0
            mask_tr = torch.ones(self.window_size[0], self.window_size[1])
            mask_tr[:-self.expand_size[0], self.expand_size[1]:] = 0
            mask_bl = torch.ones(self.window_size[0], self.window_size[1])
            mask_bl[self.expand_size[0]:, :-self.expand_size[1]] = 0
            mask_br = torch.ones(self.window_size[0], self.window_size[1])
            mask_br[self.expand_size[0]:, self.expand_size[1]:] = 0
            mask_rolled = torch.stack((mask_tl, mask_tr, mask_bl, mask_br),
                                      0).flatten(0)
            self.register_buffer("valid_ind_rolled",
                                 mask_rolled.nonzero(as_tuple=False).view(-1))

        if pool_method != "none" and focal_level > 1:
            self.unfolds = nn.ModuleList()

            # build relative position bias between local patch and pooled windows
            for k in range(focal_level - 1):
                stride = 2**k
                kernel_size = tuple(2 * (i // 2) + 2**k + (2**k - 1)
                                    for i in self.focal_window)
                # define unfolding operations
                self.unfolds += [
                    nn.Unfold(kernel_size=kernel_size,
                              stride=stride,
                              padding=tuple(i // 2 for i in kernel_size))
                ]

                # define unfolding index for focal_level > 0
                if k > 0:
                    mask = torch.zeros(kernel_size)
                    mask[(2**k) - 1:, (2**k) - 1:] = 1
                    self.register_buffer(
                        "valid_ind_unfold_{}".format(k),
                        mask.flatten(0).nonzero(as_tuple=False).view(-1))

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x_all, mask_all=None):
        """
        Args:
            x: input features with shape of (B, T, Wh, Ww, C)
            mask: (0/-inf) mask with shape of (num_windows, T*Wh*Ww, T*Wh*Ww) or None

            output: (nW*B, Wh*Ww, C)
        """
        x = x_all[0]

        B, T, nH, nW, C = x.shape
        qkv = self.qkv(x).reshape(B, T, nH, nW, 3,
                                  C).permute(4, 0, 1, 2, 3, 5).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]  # B, T, nH, nW, C

        # partition q map
        (q_windows, k_windows, v_windows) = map(
            lambda t: window_partition(t, self.window_size).view(
                -1, T, self.window_size[0] * self.window_size[1], self.
                num_heads, C // self.num_heads).permute(0, 3, 1, 2, 4).
            contiguous().view(-1, self.num_heads, T * self.window_size[
                0] * self.window_size[1], C // self.num_heads), (q, k, v))
        # q(k/v)_windows shape : [16, 4, 225, 128]

        if any(i > 0 for i in self.expand_size) and self.focal_level > 0:
            (k_tl, v_tl) = map(
                lambda t: torch.roll(t,
                                     shifts=(-self.expand_size[0], -self.
                                             expand_size[1]),
                                     dims=(2, 3)), (k, v))
            (k_tr, v_tr) = map(
                lambda t: torch.roll(t,
                                     shifts=(-self.expand_size[0], self.
                                             expand_size[1]),
                                     dims=(2, 3)), (k, v))
            (k_bl, v_bl) = map(
                lambda t: torch.roll(t,
                                     shifts=(self.expand_size[0], -self.
                                             expand_size[1]),
                                     dims=(2, 3)), (k, v))
            (k_br, v_br) = map(
                lambda t: torch.roll(t,
                                     shifts=(self.expand_size[0], self.
                                             expand_size[1]),
                                     dims=(2, 3)), (k, v))

            (k_tl_windows, k_tr_windows, k_bl_windows, k_br_windows) = map(
                lambda t: window_partition(t, self.window_size).view(
                    -1, T, self.window_size[0] * self.window_size[1], self.
                    num_heads, C // self.num_heads), (k_tl, k_tr, k_bl, k_br))
            (v_tl_windows, v_tr_windows, v_bl_windows, v_br_windows) = map(
                lambda t: window_partition(t, self.window_size).view(
                    -1, T, self.window_size[0] * self.window_size[1], self.
                    num_heads, C // self.num_heads), (v_tl, v_tr, v_bl, v_br))
            k_rolled = torch.cat(
                (k_tl_windows, k_tr_windows, k_bl_windows, k_br_windows),
                2).permute(0, 3, 1, 2, 4).contiguous()
            v_rolled = torch.cat(
                (v_tl_windows, v_tr_windows, v_bl_windows, v_br_windows),
                2).permute(0, 3, 1, 2, 4).contiguous()

            # mask out tokens in current window
            k_rolled = k_rolled[:, :, :, self.valid_ind_rolled]
            v_rolled = v_rolled[:, :, :, self.valid_ind_rolled]
            temp_N = k_rolled.shape[3]
            k_rolled = k_rolled.view(-1, self.num_heads, T * temp_N,
                                     C // self.num_heads)
            v_rolled = v_rolled.view(-1, self.num_heads, T * temp_N,
                                     C // self.num_heads)
            k_rolled = torch.cat((k_windows, k_rolled), 2)
            v_rolled = torch.cat((v_windows, v_rolled), 2)
        else:
            k_rolled = k_windows
            v_rolled = v_windows

        # q(k/v)_windows shape : [16, 4, 225, 128]
        # k_rolled.shape : [16, 4, 5, 165, 128]
        # ideal expanded window size 153 ((5+2*2)*(9+2*4))
        # k_windows=45 expand_window=108 overlap_window=12 (since expand_size < window_size / 2)

        if self.pool_method != "none" and self.focal_level > 1:
            k_pooled = []
            v_pooled = []
            for k in range(self.focal_level - 1):
                stride = 2**k
                # B, T, nWh, nWw, C
                x_window_pooled = x_all[k + 1].permute(0, 3, 1, 2,
                                                       4).contiguous()

                nWh, nWw = x_window_pooled.shape[2:4]

                # generate mask for pooled windows
                mask = x_window_pooled.new(T, nWh, nWw).fill_(1)
                # unfold mask: [nWh*nWw//s//s, k*k, 1]
                unfolded_mask = self.unfolds[k](mask.unsqueeze(1)).view(
                    1, T, self.unfolds[k].kernel_size[0], self.unfolds[k].kernel_size[1], -1).permute(4, 1, 2, 3, 0).contiguous().\
                    view(nWh*nWw // stride // stride, -1, 1)

                if k > 0:
                    valid_ind_unfold_k = getattr(
                        self, "valid_ind_unfold_{}".format(k))
                    unfolded_mask = unfolded_mask[:, valid_ind_unfold_k]

                x_window_masks = unfolded_mask.flatten(1).unsqueeze(0)
                x_window_masks = x_window_masks.masked_fill(
                    x_window_masks == 0,
                    float(-100.0)).masked_fill(x_window_masks > 0, float(0.0))
                mask_all[k + 1] = x_window_masks

                # generate k and v for pooled windows
                qkv_pooled = self.qkv(x_window_pooled).reshape(
                    B, T, nWh, nWw, 3, C).permute(4, 0, 1, 5, 2,
                                                  3).view(3, -1, C, nWh,
                                                          nWw).contiguous()
                # B*T, C, nWh, nWw
                k_pooled_k, v_pooled_k = qkv_pooled[1], qkv_pooled[2]
                # k_pooled_k shape: [5, 512, 4, 4], i.e. [B*T, C, nWh, nWw] 空间池化后的window, 最后两个通道是window数量
                # self.unfolds[k](k_pooled_k) shape: [5, 23040 (512 * 5 * 9 ), 16]

                (k_pooled_k, v_pooled_k) = map(
                    lambda t: self.unfolds[k]
                    (t).view(B, T, C, self.unfolds[k].kernel_size[0], self.
                             unfolds[k].kernel_size[1], -1)
                    .permute(0, 5, 1, 3, 4, 2).contiguous().view(
                        -1, T, self.unfolds[k].kernel_size[0] * self.unfolds[
                            k].kernel_size[1], self.num_heads, C // self.
                        num_heads).permute(0, 3, 1, 2, 4).contiguous(),
                    # (B x (nH*nW)) x nHeads x T x (unfold_wsize x unfold_wsize) x head_dim
                    (k_pooled_k, v_pooled_k))
                # k_pooled_k shape : [16, 4, 5, 45, 128],
                # i.e. [B * nWh * nWw, head, T, sh * sw, C // head], sh和sw是window的尺寸(5*9)

                # select valid unfolding index
                if k > 0:
                    (k_pooled_k, v_pooled_k) = map(
                        lambda t: t[:, :, :, valid_ind_unfold_k],
                        (k_pooled_k, v_pooled_k))

                k_pooled_k = k_pooled_k.view(
                    -1, self.num_heads, T * self.unfolds[k].kernel_size[0] *
                    self.unfolds[k].kernel_size[1], C // self.num_heads)
                v_pooled_k = v_pooled_k.view(
                    -1, self.num_heads, T * self.unfolds[k].kernel_size[0] *
                    self.unfolds[k].kernel_size[1], C // self.num_heads)

                k_pooled += [k_pooled_k]
                v_pooled += [v_pooled_k]

            # k_all (v_all) shape : [16, 4, 5 * 210, 128], i.e. [B * nWh * nWw, head, k_rolled + k_pooled, C // head]
            # k_pooled : [B * nWh * nWw, head, T * sh * sw, C // head]
            k_all = torch.cat([k_rolled] + k_pooled, 2)
            v_all = torch.cat([v_rolled] + v_pooled, 2)
        else:
            k_all = k_rolled
            v_all = v_rolled

        N = k_all.shape[-2]
        q_windows = q_windows * self.scale
        # B*nW, nHead, T*window_size*window_size, T*focal_window_size*focal_window_size
        attn = (q_windows @ k_all.transpose(-2, -1))
        # T * 45
        window_area = T * self.window_size[0] * self.window_size[1]
        # T * 165
        window_area_rolled = k_rolled.shape[2]

        if self.pool_method != "none" and self.focal_level > 1:
            offset = window_area_rolled
            for k in range(self.focal_level - 1):
                # add attentional mask
                # mask_all[1] shape [1, 16, T * 45]

                bias = tuple((i + 2**k - 1) for i in self.focal_window)

                if mask_all[k + 1] is not None:
                    attn[:, :, :window_area, offset:(offset + (T*bias[0]*bias[1]))] = \
                        attn[:, :, :window_area, offset:(offset + (T*bias[0]*bias[1]))] + \
                        mask_all[k+1][:, :, None, None, :].repeat(
                            attn.shape[0] // mask_all[k+1].shape[1], 1, 1, 1, 1).view(-1, 1, 1, mask_all[k+1].shape[-1])

                offset += T * bias[0] * bias[1]

        if mask_all[0] is not None:
            nW = mask_all[0].shape[0]
            attn = attn.view(attn.shape[0] // nW, nW, self.num_heads,
                             window_area, N)
            attn[:, :, :, :, :
                 window_area] = attn[:, :, :, :, :window_area] + mask_all[0][
                     None, :, None, :, :]
            attn = attn.view(-1, self.num_heads, window_area, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        x = (attn @ v_all).transpose(1, 2).reshape(attn.shape[0], window_area,
                                                   C)
        x = self.proj(x)
        return x


class WindowAttentionMem(nn.Module):
    """Temporal focal window attention with memory built in."""
    def __init__(self, dim, expand_size, window_size, focal_window,
                 focal_level, num_heads, qkv_bias, pool_method,
                 memory, max_mem_len, compression_factor, mem_pool, store_lf, align_cache, sub_token_align, sub_factor,
                 cross_att, time_att, time_deco, temp_focal, cs_win, mem_att, cs_focal, cs_focal_v2, cs_win_strip):

        super().__init__()
        self.dim = dim
        self.expand_size = expand_size
        self.window_size = window_size  # Wh, Ww
        self.pool_method = pool_method
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.focal_level = focal_level
        self.focal_window = focal_window

        self.memory = memory
        self.mem_pool = mem_pool
        self.store_lf = store_lf
        self.align_cache = align_cache
        self.sub_token_align = sub_token_align
        self.cross_att = cross_att
        self.time_att = time_att
        self.time_deco = time_deco
        self.temp_focal = temp_focal
        self.cs_win = cs_win
        self.mem_att = mem_att
        self.cs_focal = cs_focal
        self.cs_focal_v2 = cs_focal_v2

        if any(i > 0 for i in self.expand_size) and focal_level > 0:
            # get mask for rolled k and rolled v
            mask_tl = torch.ones(self.window_size[0], self.window_size[1])
            mask_tl[:-self.expand_size[0], :-self.expand_size[1]] = 0
            mask_tr = torch.ones(self.window_size[0], self.window_size[1])
            mask_tr[:-self.expand_size[0], self.expand_size[1]:] = 0
            mask_bl = torch.ones(self.window_size[0], self.window_size[1])
            mask_bl[self.expand_size[0]:, :-self.expand_size[1]] = 0
            mask_br = torch.ones(self.window_size[0], self.window_size[1])
            mask_br[self.expand_size[0]:, self.expand_size[1]:] = 0
            mask_rolled = torch.stack((mask_tl, mask_tr, mask_bl, mask_br),
                                      0).flatten(0)
            self.register_buffer("valid_ind_rolled",
                                 mask_rolled.nonzero(as_tuple=False).view(-1))

        if pool_method != "none" and focal_level > 1:
            self.unfolds = nn.ModuleList()

            # build relative position bias between local patch and pooled windows
            for k in range(focal_level - 1):
                stride = 2**k
                kernel_size = tuple(2 * (i // 2) + 2**k + (2**k - 1)
                                    for i in self.focal_window)
                # define unfolding operations
                self.unfolds += [
                    nn.Unfold(kernel_size=kernel_size,
                              stride=stride,
                              padding=tuple(i // 2 for i in kernel_size))
                ]

                # define unfolding index for focal_level > 0
                if k > 0:
                    mask = torch.zeros(kernel_size)
                    mask[(2**k) - 1:, (2**k) - 1:] = 1
                    self.register_buffer(
                        "valid_ind_unfold_{}".format(k),
                        mask.flatten(0).nonzero(as_tuple=False).view(-1))

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.softmax = nn.Softmax(dim=-1)

        if self.memory:
            self.m_k = []  # 缓存的memory keys
            self.m_v = []  # 缓存的memory values
            self.max_len = max_mem_len  # 缓存memory的最大记忆长度
            self.compression_factor = compression_factor  # 缓存memory的通道压缩因子

            if not self.mem_pool:
                # memory机制的含参数运算层-[基于通道的压缩]
                # 兼容局部非局部都存储和只存储局部帧的行为
                self.f_k = nn.Linear(dim, dim // self.compression_factor, bias=True)  # 用于更新上一时刻的k并压缩之前的记忆张量
                self.f_v = nn.Linear(dim, dim // self.compression_factor, bias=True)  # 用于更新上一时刻的v并压缩之前的记忆张量

                if not self.cross_att:
                    # 使用线性层聚合时需要这些层
                    self.lin_q = nn.Linear(dim, dim, bias=True)  # 用于把当前的q转换为适合查找记忆力的q
                    self.lin_k = nn.Linear(
                        dim + dim // self.compression_factor * self.max_len,
                        dim, bias=True)  # 用于把记忆里的k和当前的k进行融合
                    self.lin_v = nn.Linear(
                        dim + dim // self.compression_factor * self.max_len,
                        dim, bias=True)  # 用于把记忆里的v和当前的v进行融合

                if self.cross_att:
                    # 使用cross attention对齐记忆缓存和当前帧

                    # 当记忆时间大于1时，需要先将缓存里的记忆压缩到和当前迭代同样尺度，才能做attention.
                    if not self.mem_att:
                        # 使用线性层聚合不同时间的记忆，然后和当前做cross att
                        if self.max_len > 1:
                            self.lin_k = nn.Linear(
                                dim // self.compression_factor * self.max_len,
                                dim, bias=qkv_bias)  # 用于把记忆里的k和当前的k进行融合
                            self.lin_v = nn.Linear(
                                dim // self.compression_factor * self.max_len,
                                dim, bias=qkv_bias)  # 用于把记忆里的v和当前的v进行融合
                    else:
                        # 使用cross att聚合不同时间的记忆和当前特征
                        pass

                    # 将记忆查询输出和当前帧的输出融合
                    self.fusion_proj = nn.Linear(2 * dim, dim)
                    if not (self.temp_focal or self.cs_win):
                        # 使用标准的attention
                        self.cm_proj = nn.Linear(dim, dim)

                        if self.time_deco:
                            # 解耦时间和空间注意力
                            self.cm_proj_t = nn.Linear(dim, dim)

                    elif self.temp_focal:
                        # 使用temp focal cross attention
                        self.cf_att = CrossFocalAttention(dim,
                                                          expand_size=self.expand_size,
                                                          window_size=self.window_size,
                                                          focal_window=focal_window,
                                                          focal_level=focal_level,
                                                          num_heads=num_heads,
                                                          qkv_bias=qkv_bias,
                                                          pool_method=pool_method)
                    elif self.cs_win:
                        # 使用cs win attention
                        window_stride = 4  # 每个window在两个方向上占用了多少个token
                        split_size = cs_win_strip  # 条形窗口的宽度
                        num_heads_cs = num_heads//2

                        patches_resolution = [self.window_size[0] * window_stride,
                                              self.window_size[1] * window_stride]     # token的纵向和横向的个数
                        if not self.time_deco:
                            # 把时间和空间窗口合并进行3D cross attention
                            self.cs_att = nn.ModuleList([
                                TemporalLePEAttention(dim//2, resolution=patches_resolution, idx=i,
                                                      split_size=split_size, num_heads=num_heads_cs, dim_out=dim//2,
                                                      qk_scale=None, attn_drop=0., proj_drop=0.,
                                                      temporal=True, cs_focal=cs_focal, cs_focal_v2=cs_focal_v2)
                                for i in range(0, 2)])      # 两个，一个横向一个纵向
                        elif self.time_deco:
                            # 解耦时间和空间注意力
                            self.cs_att = nn.ModuleList([
                                TemporalLePEAttention(dim//2, resolution=patches_resolution, idx=i,
                                                      split_size=split_size, num_heads=num_heads_cs, dim_out=dim//2,
                                                      qk_scale=None, attn_drop=0., proj_drop=0.,
                                                      temporal=False, cs_focal=cs_focal, cs_focal_v2=cs_focal_v2)
                                for i in range(0, 2)])      # 两个，一个横向一个纵向
                            # 时间attention的线性层
                            self.cm_proj_t = nn.Linear(dim, dim)
                        # cs win 的线性层
                        self.cs_proj = nn.Linear(dim, dim)

            else:
                # memory机制的含参数运算层-[基于池化的压缩]
                # 全连接池化层 如果使用token缩减，将4替换为缩减比例*4
                self.pool_q = nn.Linear(self.focal_window[0] * 4 * self.focal_window[1] * 4,
                                        math.floor(self.focal_window[0] * 4 / self.compression_factor) *
                                        math.floor(self.focal_window[1] * 4 / self.compression_factor), bias=True)
                self.pool_q.weight.data.fill_(
                    1. / (self.focal_window[0] * 4 * self.focal_window[1] * 4))
                self.pool_q.bias.data.fill_(0)
                self.pool_k = nn.Linear(self.focal_window[0] * 4 * self.focal_window[1] * 4,
                                        math.floor(self.focal_window[0] * 4 / self.compression_factor) *
                                        math.floor(self.focal_window[1] * 4 / self.compression_factor), bias=True)
                self.pool_k.weight.data.fill_(
                    1. / (self.focal_window[0] * 4 * self.focal_window[1] * 4))
                self.pool_k.bias.data.fill_(0)
                self.pool_v = nn.Linear(self.focal_window[0] * 4 * self.focal_window[1] * 4,
                                        math.floor(self.focal_window[0] * 4 / self.compression_factor) *
                                        math.floor(self.focal_window[1] * 4 / self.compression_factor), bias=True)
                self.pool_v.weight.data.fill_(
                    1. / (self.focal_window[0] * 4 * self.focal_window[1] * 4))
                self.pool_v.bias.data.fill_(0)

                self.f_k = nn.Linear(dim, dim // self.compression_factor, bias=True)  # 用于更新上一时刻的k并压缩之前的记忆张量
                self.f_v = nn.Linear(dim, dim // self.compression_factor, bias=True)  # 用于更新上一时刻的v并压缩之前的记忆张量

                # self.f_k = nn.Linear(dim, dim, bias=True)  # 用于更新上一时刻的k
                # self.f_v = nn.Linear(dim, dim, bias=True)  # 用于更新上一时刻的v

                self.lin_q = nn.Linear(dim, dim, bias=True)  # 用于把当前的q转换为适合查找记忆力的q
                self.lin_k = nn.Linear(
                    int(dim + dim // self.compression_factor * self.max_len),
                    dim, bias=True)  # 用于把记忆里的k和当前的k进行融合
                self.lin_v = nn.Linear(
                    int(dim + dim // self.compression_factor * self.max_len),
                    dim, bias=True)  # 用于把记忆里的v和当前的v进行融合

                self.unpool_q = nn.Linear(self.focal_window[0] * 4 * self.focal_window[1] * 4 +
                                          math.floor(self.focal_window[0] * 4 / self.compression_factor) *
                                          math.floor(self.focal_window[1] * 4 / self.compression_factor),
                                          self.focal_window[0] * 4 * self.focal_window[1] * 4, bias=True)
                self.unpool_k = nn.Linear(self.focal_window[0] * 4 * self.focal_window[1] * 4 +
                                          math.floor(self.focal_window[0] * 4 / self.compression_factor) *
                                          math.floor(self.focal_window[1] * 4 / self.compression_factor),
                                          self.focal_window[0] * 4 * self.focal_window[1] * 4, bias=True)
                self.unpool_v = nn.Linear(self.focal_window[0] * 4 * self.focal_window[1] * 4 +
                                          math.floor(self.focal_window[0] * 4 / self.compression_factor) *
                                          math.floor(self.focal_window[1] * 4 / self.compression_factor),
                                          self.focal_window[0] * 4 * self.focal_window[1] * 4, bias=True)

            if self.align_cache:
                # 使用光流对齐缓存
                if not self.sub_token_align:
                    self.flow_head = FlowHead(input_dim=(dim + dim // self.compression_factor), hidden_factor=2)
                else:
                    self.sub_factor = sub_factor
                    self.flow_head = FlowHead(
                        input_dim=(dim // self.sub_factor + (dim // self.compression_factor) // self.sub_factor),
                        hidden_factor=2)

                # 缓存对齐的记忆张量防止两次backward错误
                self.m_k_aligned = []  # 缓存的已对齐memory keys
                self.m_v_aligned = []  # 缓存的已对齐memory keys

    def forward(self, x_all, mask_all=None, l_t=5):
        """
        Args:
            x_all: input features with shape of (B, T, Wh, Ww, C)
            mask_all: (0/-inf) mask with shape of (num_windows, T*Wh*Ww, T*Wh*Ww) or None
            l_t: local frame nums

            output: (nW*B, Wh*Ww, C)
        """
        x = x_all[0]    # x_all[1]是用来生成池化的kv来做self attention focal的

        B, T, nH, nW, C = x.shape
        qkv = self.qkv(x).reshape(B, T, nH, nW, 3,
                                  C).permute(4, 0, 1, 2, 3, 5).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]  # B, T, nH, nW, C

        # memory ability
        if self.memory:

            if not self.mem_pool:
                # 通道压缩时序的记忆力

                if not self.store_lf:
                    # 局部和随机的非局部帧都会被存储
                    # 压缩上一个记忆缓存
                    if len(self.m_k) != 0:
                        cm_k = self.f_k(self.m_k[-1])
                        cm_v = self.f_v(self.m_v[-1])
                    else:
                        # 第一帧时没有记忆张量，使用当前帧的k，v
                        cm_k = self.f_k(k)
                        cm_v = self.f_v(v)

                    # 增强qkv
                    if not self.cross_att:
                        # 使用线性层聚合缓存和当前迭代
                        q = self.lin_q(q)
                        if len(self.m_k) == self.max_len:
                            # 因为缓存里的最后一帧还没被压缩的替换，所以不取最后一帧
                            k = self.lin_k(torch.cat((
                                torch.cat(self.m_k[:-1], dim=4), cm_k, k), dim=4))
                            v = self.lin_v(torch.cat((
                                torch.cat(self.m_v[:-1], dim=4), cm_v, v), dim=4))
                            # 后面的索引用到了k所以保存一下
                            k_temp = k
                        else:
                            repeat_k = self.max_len - len(self.m_k)
                            repeat_v = self.max_len - len(self.m_v)
                            # 尽量使用缓存中的帧，不够的使用当前帧提取的代替
                            if len(self.m_k) == 0:
                                k = self.lin_k(torch.cat((cm_k.repeat(1, 1, 1, 1, repeat_k), k), dim=4))
                                v = self.lin_v(torch.cat((cm_v.repeat(1, 1, 1, 1, repeat_v), v), dim=4))
                                k_temp = k  # debug
                            else:
                                k_rep_feat = self.f_k(k)
                                k_rep_feat = k_rep_feat.repeat(1, 1, 1, 1, repeat_k)
                                v_rep_feat = self.f_v(v)
                                v_rep_feat = v_rep_feat.repeat(1, 1, 1, 1, repeat_v)
                                k = self.lin_k(torch.cat((
                                    torch.cat(self.m_k[:-1], dim=4), cm_k, k_rep_feat, k), dim=4))
                                v = self.lin_v(torch.cat((
                                    torch.cat(self.m_v[:-1], dim=4), cm_v, v_rep_feat, v), dim=4))
                                k_temp = k  # debug

                    else:
                        # 使用cross attention聚合记忆

                        # 聚合记忆缓存，用于后续和当前特征进行cross attention
                        if self.max_len == 1:
                            # 只记忆1次迭代时，不需要聚合
                            att_num = 1
                            mem_k = cm_k
                            mem_v = cm_v
                        else:
                            # 记忆缓存时间大于1，需要聚合记忆再做attention
                            if not self.mem_att:
                                # 使用线性层聚合记忆
                                # 只需要最后聚合的记忆和当前特征做一次cross att
                                att_num = 1
                                if len(self.m_k) == self.max_len:
                                    # 记忆缓存满了，直接用线性层聚合
                                    # 因为缓存里的最后一帧还没被压缩的替换，所以不取最后一帧
                                    mem_k = self.lin_k(torch.cat((
                                        torch.cat(self.m_k[:-1], dim=4), cm_k), dim=4))
                                    mem_v = self.lin_v(torch.cat((
                                        torch.cat(self.m_v[:-1], dim=4), cm_v), dim=4))
                                else:
                                    # 记忆缓存没满，复制一下
                                    repeat_k = self.max_len - len(self.m_k)
                                    repeat_v = self.max_len - len(self.m_v)
                                    if len(self.m_k) == 0:
                                        # 缓存里面啥也没有，当前帧多复制几次
                                        mem_k = self.lin_k(cm_k.repeat(1, 1, 1, 1, repeat_k))
                                        mem_v = self.lin_v(cm_v.repeat(1, 1, 1, 1, repeat_k))
                                    else:
                                        # 尽量使用缓存中的帧
                                        mem_k = self.lin_k(torch.cat((
                                            torch.cat(self.m_k[:-1], dim=4), cm_k.repeat(1, 1, 1, 1, repeat_k + 1)), dim=4))
                                        mem_v = self.lin_v(torch.cat((
                                            torch.cat(self.m_v[:-1], dim=4), cm_v.repeat(1, 1, 1, 1, repeat_v + 1)), dim=4))
                            else:
                                # 直接把不同时间的记忆和当前迭代分别做attention就好了
                                # attention的次数等于记忆的长度
                                if len(self.m_k) == 0:
                                    # 当缓存里是空的时，直接和当前特征自己做1次attention
                                    att_num = 1
                                else:
                                    # 当缓存里不是空的时，和缓存里的做attention
                                    att_num = len(self.m_k)

                        for att_idx in range(0, att_num):

                            # 记忆时间超过1并且不用线性层聚合才需要这些判断逻辑
                            # 也就是说，只有需要做多次cross attention才需要这些逻辑
                            if self.mem_att:
                                # 每次迭代是对不同时间的记忆做cross attention
                                if len(self.m_k) == 0:
                                    # 缓存里是空的，和更新过的自己做self attention
                                    mem_k = cm_k
                                    mem_v = cm_v
                                    pass
                                else:
                                    # 缓存里不是空的，和缓存里更新过的记忆以及当前更新的记忆做attention
                                    # 先取缓存里的记忆
                                    if att_idx != (att_num - 1):
                                        mem_k = self.m_k[:-1][att_idx]
                                        mem_v = self.m_v[:-1][att_idx]
                                    else:
                                        # 最后1次取当前压缩过的kv
                                        mem_k = cm_k
                                        mem_v = cm_v

                            # 各种不同的cross attention选择
                            if not self.time_att:
                                # 信息只在Nh Nw维度流动(空间维度)
                                q = q.reshape(B * T, self.num_heads, nH * nW, C // self.num_heads).contiguous()
                                mem_k = mem_k.reshape(B * T, self.num_heads, nH * nW, C // self.num_heads).contiguous()
                                mem_v = mem_v.reshape(B * T, self.num_heads, nH * nW, C // self.num_heads).contiguous()
                                cm_attn = (q @ mem_k.transpose(-2, -1)) * self.scale
                                cm_attn = cm_attn.softmax(dim=-1)
                                cm_x = (cm_attn @ mem_v).transpose(1, 2).reshape(B * T, nH * nW, C)
                                cm_x = self.cm_proj(cm_x)
                                # 恢复原来的shape
                                q = q.reshape(B, T, nH, nW, C).contiguous()
                                # mem_k = mem_k.reshape(B, T, nH, nW, C).contiguous()
                                # mem_v = mem_v.reshape(B, T, nH, nW, C).contiguous()

                            else:
                                # 信息将额外在T维度流动
                                if self.time_deco and not self.cs_win:
                                    # 解耦时间和空间注意力的vanilla attention
                                    # 时间注意力
                                    q = q.permute(0, 2, 3, 1, 4).reshape(B * nH * nW, T, C // self.num_heads, self.num_heads)\
                                        .permute(0, 3, 1, 2).contiguous()   # B*N, head, T, C//head
                                    mem_k = mem_k.permute(0, 2, 3, 1, 4).reshape(B * nH * nW, T, C // self.num_heads, self.num_heads)\
                                        .permute(0, 3, 1, 2).contiguous()
                                    mem_v = mem_v.permute(0, 2, 3, 1, 4).reshape(B * nH * nW, T, C // self.num_heads, self.num_heads)\
                                        .permute(0, 3, 1, 2).contiguous()
                                    cm_attn_t = (q @ mem_k.transpose(-2, -1)) * self.scale
                                    cm_attn_t = cm_attn_t.softmax(dim=-1)
                                    cm_x_t = (cm_attn_t @ mem_v).permute(0, 2, 3, 1).reshape(B, nH * nW, T, C)\
                                        .permute(0, 2, 1, 3).reshape(B * T, nH * nW, C)
                                    cm_x_t = self.cm_proj_t(cm_x_t)
                                    # 恢复qkv的shape
                                    q = q.permute(0, 2, 3, 1).reshape(B, nH, nW, T, C).permute(0, 3, 1, 2, 4).contiguous()
                                    mem_k = mem_k.permute(0, 2, 3, 1).reshape(B, nH, nW, T, C).permute(0, 3, 1, 2, 4).contiguous()
                                    mem_v = mem_v.permute(0, 2, 3, 1).reshape(B, nH, nW, T, C).permute(0, 3, 1, 2, 4).contiguous()

                                    # 空间注意力
                                    q = q.reshape(B * T, nH * nW, C // self.num_heads, self.num_heads).permute(0, 3, 1, 2)\
                                        .contiguous()   # BT, head, N, C//head
                                    mem_k = mem_k.reshape(B * T, nH * nW, C // self.num_heads, self.num_heads).permute(0, 3, 1, 2)\
                                        .contiguous()
                                    mem_v = mem_v.reshape(B * T, nH * nW, C // self.num_heads, self.num_heads).permute(0, 3, 1, 2)\
                                        .contiguous()
                                    cm_attn = (q @ mem_k.transpose(-2, -1)) * self.scale
                                    cm_attn = cm_attn.softmax(dim=-1)
                                    cm_x = (cm_attn @ mem_v).transpose(1, 2).reshape(B * T, nH * nW, C)
                                    cm_x = self.cm_proj(cm_x)
                                    # 恢复qkv的shape
                                    q = q.permute(0, 2, 3, 1).reshape(B, T, nH, nW, C).contiguous()
                                    # mem_k = mem_k.permute(0, 2, 3, 1).reshape(B, T, nH, nW, C).contiguous()
                                    # mem_v = mem_v.permute(0, 2, 3, 1).reshape(B, T, nH, nW, C).contiguous()

                                    # 暂时只使用相加融合两次查询
                                    cm_x += cm_x_t

                                elif self.temp_focal:
                                    # 基于temporal focal attention实现时空记忆聚合
                                    cm_x = self.cf_att(qkv=[q, mem_k, mem_v], mask_all=mask_all)

                                elif self.cs_win:
                                    # 基于cswin attention聚合时空记忆和当前迭代

                                    if self.time_deco:
                                        # 解耦时间和空间聚合，时间聚合使用vanilla attention
                                        q = q.permute(0, 2, 3, 1, 4).reshape(B * nH * nW, T, C // self.num_heads,
                                                                             self.num_heads) \
                                            .permute(0, 3, 1, 2).contiguous()  # B*N, head, T, C//head
                                        mem_k = mem_k.permute(0, 2, 3, 1, 4).reshape(B * nH * nW, T, C // self.num_heads,
                                                                                     self.num_heads) \
                                            .permute(0, 3, 1, 2).contiguous()
                                        mem_v = mem_v.permute(0, 2, 3, 1, 4).reshape(B * nH * nW, T, C // self.num_heads,
                                                                                     self.num_heads) \
                                            .permute(0, 3, 1, 2).contiguous()
                                        cm_attn_t = (q @ mem_k.transpose(-2, -1)) * self.scale
                                        cm_attn_t = cm_attn_t.softmax(dim=-1)
                                        cm_x_t = (cm_attn_t @ mem_v).permute(0, 2, 3, 1).reshape(B, nH * nW, T, C) \
                                            .permute(0, 2, 1, 3).reshape(B * T, nH * nW, C)
                                        cm_x_t = self.cm_proj_t(cm_x_t)
                                        # 恢复qkv的shape
                                        q = q.permute(0, 2, 3, 1).reshape(B, nH, nW, T, C).permute(0, 3, 1, 2,
                                                                                                   4).contiguous()
                                        mem_k = mem_k.permute(0, 2, 3, 1).reshape(B, nH, nW, T, C).permute(0, 3, 1, 2,
                                                                                                           4).contiguous()
                                        mem_v = mem_v.permute(0, 2, 3, 1).reshape(B, nH, nW, T, C).permute(0, 3, 1, 2,
                                                                                                           4).contiguous()

                                    cm_x1 = self.cs_att[0](qkv=[q[:, :, :, :, :C // 2],
                                                                mem_k[:, :, :, :, :C // 2],
                                                                mem_v[:, :, :, :, :C // 2]])
                                    cm_x2 = self.cs_att[1](qkv=[q[:, :, :, :, C // 2:],
                                                                mem_k[:, :, :, :, C // 2:],
                                                                mem_v[:, :, :, :, C // 2:]])
                                    cm_x = torch.cat([cm_x1, cm_x2], dim=2)
                                    cm_x = self.cs_proj(cm_x)

                                    if self.time_deco:
                                        # 暂时只使用相加融合两次查询
                                        cm_x += cm_x_t.reshape(B, T * nH * nW, C)

                                else:
                                    # 不解耦时间和空间的vanilla attention
                                    q = q.reshape(B, self.num_heads, T * nH * nW, C // self.num_heads).contiguous()
                                    mem_k = mem_k.reshape(B, self.num_heads, T * nH * nW, C // self.num_heads).contiguous()
                                    mem_v = mem_v.reshape(B, self.num_heads, T * nH * nW, C // self.num_heads).contiguous()
                                    cm_attn = (q @ mem_k.transpose(-2, -1)) * self.scale
                                    cm_attn = cm_attn.softmax(dim=-1)
                                    cm_x = (cm_attn @ mem_v).transpose(1, 2).reshape(B * T, nH * nW, C)
                                    cm_x = self.cm_proj(cm_x)

                                    # 恢复原来的shape
                                    q = q.reshape(B, T, nH, nW, C).contiguous()
                                    # mem_k = mem_k.reshape(B, T, nH, nW, C).contiguous()
                                    # mem_v = mem_v.reshape(B, T, nH, nW, C).contiguous()

                                    # 除了解耦的，前面两个版本的qkv shape变换都需要检查

                            # cm_x_final用来存储不同时间记忆attention的结果
                            if att_idx == 0:
                                # 第一次直接初始化cm_x_final
                                cm_x_final = cm_x
                            else:
                                cm_x_final += cm_x

                        k_temp = k  # debug

                else:
                    # 仅存储局部帧的记忆
                    q_lf = q[:, :l_t, ...]
                    k_lf = k[:, :l_t, ...]
                    v_lf = v[:, :l_t, ...]

                    # 压缩上一个记忆缓存
                    if len(self.m_k) != 0:
                        cm_k = self.f_k(self.m_k[-1])
                        cm_v = self.f_v(self.m_v[-1])
                    else:
                        # 第一帧时没有记忆张量，使用当前的局部帧k，v
                        cm_k = self.f_k(k_lf)
                        cm_v = self.f_v(v_lf)

                    if self.align_cache:
                        # 在增强前将缓存里面所有的记忆与当前迭代的k v对齐
                        # 在记忆缓存的最后一帧被压缩后进行对齐
                        cm_k = cm_k.reshape(B * l_t, C // self.compression_factor, nH, nW)  # B*Lt, C_compress, nH, nW
                        cm_v = cm_v.reshape(B * l_t, C // self.compression_factor, nH, nW)  # B*Lt, C_compress, nH, nW
                        k_lf = k_lf.reshape(B * l_t, C, nH, nW)                             # B*Lt, C, nH, nW
                        v_lf = v_lf.reshape(B * l_t, C, nH, nW)                             # B*Lt, C, nH, nW

                        if not self.sub_token_align:
                            # 在token尺度估计光流完成对齐
                            token_flow_k = self.flow_head(torch.cat((cm_k, k_lf), dim=1)).reshape(B * l_t, nH, nW, 2)
                            token_flow_v = self.flow_head(torch.cat((cm_v, v_lf), dim=1)).reshape(B * l_t, nH, nW, 2)

                            cm_k = flow_warp(cm_k, token_flow_k)                             # B*Lt, C_compress, nH, nW
                            cm_v = flow_warp(cm_v, token_flow_v)                             # B*Lt, C_compress, nH, nW

                            cm_k = cm_k.reshape(B, l_t, nH, nW, C // self.compression_factor)
                            cm_v = cm_v.reshape(B, l_t, nH, nW, C // self.compression_factor)
                        else:
                            # 在sub-token尺度估计光流完成对齐
                            group_stride = C // self.sub_factor
                            group_stride_compressed = group_stride // self.compression_factor
                            cm_kk_list = []  # 防止两次梯度反串报错
                            cm_vv_list = []
                            # kk_lf_list = []  # 存储sub-token的kk_lf加快速度
                            for group_idx in range(0, self.sub_factor):
                                # 取出当前group的sub-token
                                cm_kk = cm_k[:,
                                        group_stride_compressed * group_idx:group_stride_compressed * (group_idx + 1),
                                        :, :]
                                cm_vv = cm_v[:,
                                        group_stride_compressed * group_idx:group_stride_compressed * (group_idx + 1),
                                        :, :]
                                kk_lf = k_lf[:, group_stride * group_idx:group_stride * (group_idx + 1), :, :]
                                vv_lf = v_lf[:, group_stride * group_idx:group_stride * (group_idx + 1), :, :]

                                # sub-token尺度的光流估计
                                token_flow_kk = self.flow_head(torch.cat((cm_kk, kk_lf), dim=1))\
                                    .reshape(B * l_t, nH, nW, 2)
                                token_flow_vv = self.flow_head(torch.cat((cm_vv, vv_lf), dim=1))\
                                    .reshape(B * l_t, nH, nW, 2)

                                # sub-token尺度的光流warp对齐
                                cm_kk = flow_warp(cm_kk, token_flow_kk)     # B*Lt, C_compress/sub_factor, nH, nW
                                cm_vv = flow_warp(cm_vv, token_flow_vv)     # B*Lt, C_compress/sub_factor, nH, nW
                                cm_kk_list.append(cm_kk)
                                cm_vv_list.append(cm_vv)

                            # 重组回完整的cm_kk_align, 作用相当于cm_k
                            cm_kk_align = torch.cat(cm_kk_list, dim=1).reshape(B, l_t, nH, nW, C // self.compression_factor)
                            cm_vv_align = torch.cat(cm_vv_list, dim=1).reshape(B, l_t, nH, nW, C // self.compression_factor)

                        # 对齐缓存里的其他帧，注意因为缓存里的最后一次迭代还没被压缩，所以不需要对齐，上面的就是对齐最后一次迭代的流程
                        self.m_k_aligned = []   # 对齐的长度会比不对齐的list长度少1
                        self.m_v_aligned = []   # 之所以新创建list是为了防止2次梯度反传报错，如果使用retain graph会导致显存消耗增加

                        if not self.sub_token_align:
                            # 在token尺度对齐缓存里的所有帧
                            for cache_index in range(0, len(self.m_k)-1):
                                k_mem = self.m_k[cache_index]
                                v_mem = self.m_v[cache_index]
                                k_mem = k_mem.reshape(B * l_t, C // self.compression_factor, nH,
                                                      nW)  # B*Lt, C_compress, nH, nW
                                v_mem = v_mem.reshape(B * l_t, C // self.compression_factor, nH,
                                                      nW)  # B*Lt, C_compress, nH, nW

                                # calc token flow
                                token_flow_k = self.flow_head(torch.cat((k_mem, k_lf), dim=1)).reshape(B * l_t, nH, nW, 2)
                                token_flow_v = self.flow_head(torch.cat((v_mem, v_lf), dim=1)).reshape(B * l_t, nH, nW, 2)

                                # warp tokens
                                k_mem = flow_warp(k_mem, token_flow_k)  # B*Lt, C, nH, nW
                                v_mem = flow_warp(v_mem, token_flow_v)  # B*Lt, C, nH, nW

                                # retrieve
                                k_mem = k_mem.reshape(B, l_t, nH, nW, C // self.compression_factor)
                                v_mem = v_mem.reshape(B, l_t, nH, nW, C // self.compression_factor)
                                self.m_k_aligned.append(k_mem)
                                self.m_v_aligned.append(v_mem)
                        else:
                            # 在sub-token尺度对齐缓存里的所有帧
                            for cache_index in range(0, len(self.m_k) - 1):
                                k_mem = self.m_k[cache_index]
                                v_mem = self.m_v[cache_index]
                                k_mem = k_mem.reshape(B * l_t, C // self.compression_factor, nH,
                                                      nW)  # B*Lt, C_compress, nH, nW
                                v_mem = v_mem.reshape(B * l_t, C // self.compression_factor, nH,
                                                      nW)  # B*Lt, C_compress, nH, nW

                                kk_mem_list = []  # 防止两次梯度反传报错
                                vv_mem_list = []
                                for group_idx in range(0, self.sub_factor):
                                    # 取出当前group的sub-token
                                    kk_mem = k_mem[:,
                                             group_stride_compressed * group_idx:group_stride_compressed * (
                                                     group_idx + 1),
                                             :, :]
                                    vv_mem = v_mem[:,
                                             group_stride_compressed * group_idx:group_stride_compressed * (
                                                     group_idx + 1),
                                             :, :]
                                    kk_lf = k_lf[:, group_stride * group_idx:group_stride * (group_idx + 1), :, :]
                                    vv_lf = v_lf[:, group_stride * group_idx:group_stride * (group_idx + 1), :, :]

                                    # sub-token尺度的光流估计
                                    token_flow_kk = self.flow_head(torch.cat((kk_mem, kk_lf), dim=1)) \
                                        .reshape(B * l_t, nH, nW, 2)
                                    token_flow_vv = self.flow_head(torch.cat((vv_mem, vv_lf), dim=1)) \
                                        .reshape(B * l_t, nH, nW, 2)

                                    # sub-token尺度的光流warp对齐
                                    kk_mem = flow_warp(kk_mem, token_flow_kk)  # B*Lt, C_compress/sub_factor, nH, nW
                                    vv_mem = flow_warp(vv_mem, token_flow_vv)  # B*Lt, C_compress/sub_factor, nH, nW
                                    kk_mem_list.append(kk_mem)
                                    vv_mem_list.append(vv_mem)

                                # 重组回完整的k_mem, 作用相当于k_mem
                                k_mem = torch.cat(kk_mem_list, dim=1).reshape(B, l_t, nH, nW,
                                                                                   C // self.compression_factor)
                                v_mem = torch.cat(vv_mem_list, dim=1).reshape(B, l_t, nH, nW,
                                                                                   C // self.compression_factor)
                                self.m_k_aligned.append(k_mem)
                                self.m_v_aligned.append(v_mem)

                        # 恢复当前k v的shape
                        k_lf = k_lf.reshape(B, l_t, nH, nW, C)
                        v_lf = v_lf.reshape(B, l_t, nH, nW, C)

                    # 增强局部帧的qkv
                    q_lf = self.lin_q(q_lf)
                    if len(self.m_k) == self.max_len:
                        # 缓存满了的情况，不需要补充临时的记忆张量
                        # 因为缓存里的最后一帧还没被压缩的替换，所以不取最后一帧的缓存，直接拿压缩完的cm就可
                        if not self.align_cache:
                            # 把没对齐的和当前iter的k v融合
                            if self.max_len > 1:    # 缓存长度大于1时可以cat
                                k_lf = self.lin_k(torch.cat((
                                    torch.cat(self.m_k[:-1], dim=4), cm_k, k_lf), dim=4))
                                v_lf = self.lin_v(torch.cat((
                                    torch.cat(self.m_v[:-1], dim=4), cm_v, v_lf), dim=4))
                            else:
                                # 当最大记忆时长只有1次迭代时，压缩过后的最后一个记忆就是全部的记忆了
                                k_lf = self.lin_k(torch.cat((cm_k, k_lf), dim=4))
                                v_lf = self.lin_v(torch.cat((cm_v, v_lf), dim=4))
                        elif self.align_cache and not self.sub_token_align:
                            # 在token尺度把对齐的和当前iter的k v融合
                            if self.max_len > 1:  # 缓存长度大于1时可以cat
                                k_lf = self.lin_k(torch.cat((
                                    torch.cat(self.m_k_aligned[:], dim=4), cm_k, k_lf), dim=4))
                                v_lf = self.lin_v(torch.cat((
                                    torch.cat(self.m_v_aligned[:], dim=4), cm_v, v_lf), dim=4))
                            else:
                                # 当最大记忆时长只有1次迭代时，压缩过后的最后一个记忆就是全部的记忆了
                                k_lf = self.lin_k(torch.cat((cm_k, k_lf), dim=4))
                                v_lf = self.lin_v(torch.cat((cm_v, v_lf), dim=4))
                        else:
                            # 在sub-token尺度把对齐的和当前iter的k v融合
                            if self.max_len > 1:  # 缓存长度大于1时可以cat
                                k_lf = self.lin_k(torch.cat((
                                    torch.cat(self.m_k_aligned[:], dim=4), cm_kk_align, k_lf), dim=4))
                                v_lf = self.lin_v(torch.cat((
                                    torch.cat(self.m_v_aligned[:], dim=4), cm_vv_align, v_lf), dim=4))
                            else:
                                # 当最大记忆时长只有1次迭代时，压缩过后的最后一个记忆就是全部的记忆了
                                k_lf = self.lin_k(torch.cat((cm_kk_align, k_lf), dim=4))
                                v_lf = self.lin_v(torch.cat((cm_vv_align, v_lf), dim=4))
                    else:
                        # 缓存还没有存满，需要复制当前帧的张量
                        repeat_k = self.max_len - len(self.m_k)
                        repeat_v = self.max_len - len(self.m_v)
                        if len(self.m_k) == 0:
                            # 缓存里啥也没有，直接把当前的全复制了
                            if not self.sub_token_align:
                                # 使用未对齐的或者token尺度对齐的cm_k
                                k_lf = self.lin_k(torch.cat((cm_k.repeat(1, 1, 1, 1, repeat_k), k_lf), dim=4))
                                v_lf = self.lin_v(torch.cat((cm_v.repeat(1, 1, 1, 1, repeat_v), v_lf), dim=4))
                            else:
                                # 使用对齐的sub-token级别cm_kk_align
                                k_lf = self.lin_k(torch.cat((cm_kk_align.repeat(1, 1, 1, 1, repeat_k), k_lf), dim=4))
                                v_lf = self.lin_v(torch.cat((cm_vv_align.repeat(1, 1, 1, 1, repeat_v), v_lf), dim=4))
                        else:
                            # 尽量使用缓存中的帧，不够的使用当前帧提取的代替
                            k_rep_feat = self.f_k(k_lf)
                            k_rep_feat = k_rep_feat.repeat(1, 1, 1, 1, repeat_k)
                            v_rep_feat = self.f_v(v_lf)
                            v_rep_feat = v_rep_feat.repeat(1, 1, 1, 1, repeat_v)
                            if not self.align_cache:
                                # 把没对齐的和当前iter的k v融合
                                k_lf = self.lin_k(torch.cat((
                                    torch.cat(self.m_k[:-1], dim=4), cm_k, k_rep_feat, k_lf), dim=4))
                                v_lf = self.lin_v(torch.cat((
                                    torch.cat(self.m_v[:-1], dim=4), cm_v, v_rep_feat, v_lf), dim=4))
                            elif not self.sub_token_align:
                                # 把token级别对齐的和当前iter的k v融合
                                k_lf = self.lin_k(torch.cat((
                                    torch.cat(self.m_k_aligned[:], dim=4), cm_k, k_rep_feat, k_lf), dim=4))
                                v_lf = self.lin_v(torch.cat((
                                    torch.cat(self.m_v_aligned[:], dim=4), cm_v, v_rep_feat, v_lf), dim=4))
                            else:
                                # 把sub-token级别对齐的和当前iter的k v融合
                                k_lf = self.lin_k(torch.cat((
                                    torch.cat(self.m_k_aligned[:], dim=4), cm_kk_align, k_rep_feat, k_lf), dim=4))
                                v_lf = self.lin_v(torch.cat((
                                    torch.cat(self.m_v_aligned[:], dim=4), cm_vv_align, v_rep_feat, v_lf), dim=4))

                    # 把增强后的局部帧qkv还原到所有的qkv中
                    q[:, :l_t, ...] = q_lf
                    k[:, :l_t, ...] = k_lf
                    v[:, :l_t, ...] = v_lf

            else:
                # 空间压缩当前的q k v
                c_q = self.pool_q(q)
                c_k = self.pool_k(k)
                c_v = self.pool_v(v)

                # 通道压缩时序的记忆力
                cm_k = self.f_k(self.m_k[-1])
                cm_v = self.f_v(self.m_v[-1])

                # 增强qkv
                c_q = self.lin_q(c_q)
                c_k = self.lin_k(torch.cat((self.m_k[:], c_k), dim=4))
                c_v = self.lin_v(torch.cat((self.m_v[:], c_v), dim=4))

                # 恢复qkv的尺度，加跳跃连接
                q = self.unpool_q(torch.cat((q, c_q), dim=1))
                k = self.unpool_k(torch.cat((k, c_k), dim=1))
                v = self.unpool_v(torch.cat((v, c_v), dim=1))

            # 把q, k, v存储回qkv，后面算窗口attention会用到
            # qkv[0] = q_temp
            # qkv[1] = k_temp
            # qkv[2] = v_temp
            # qkv[0], qkv[1], qkv[2] = q_temp, k_temp, v_temp

        # partition q map
        (q_windows, k_windows, v_windows) = map(
            lambda t: window_partition(t, self.window_size).view(
                -1, T, self.window_size[0] * self.window_size[1], self.
                num_heads, C // self.num_heads).permute(0, 3, 1, 2, 4).
            contiguous().view(-1, self.num_heads, T * self.window_size[
                0] * self.window_size[1], C // self.num_heads), (q, k, v))
        # q(k/v)_windows shape : [16, 4, 225, 128]

        if any(i > 0 for i in self.expand_size) and self.focal_level > 0:
            (k_tl, v_tl) = map(
                lambda t: torch.roll(t,
                                     shifts=(-self.expand_size[0], -self.
                                             expand_size[1]),
                                     dims=(2, 3)), (k, v))
            (k_tr, v_tr) = map(
                lambda t: torch.roll(t,
                                     shifts=(-self.expand_size[0], self.
                                             expand_size[1]),
                                     dims=(2, 3)), (k, v))
            (k_bl, v_bl) = map(
                lambda t: torch.roll(t,
                                     shifts=(self.expand_size[0], -self.
                                             expand_size[1]),
                                     dims=(2, 3)), (k, v))
            (k_br, v_br) = map(
                lambda t: torch.roll(t,
                                     shifts=(self.expand_size[0], self.
                                             expand_size[1]),
                                     dims=(2, 3)), (k, v))

            (k_tl_windows, k_tr_windows, k_bl_windows, k_br_windows) = map(
                lambda t: window_partition(t, self.window_size).view(
                    -1, T, self.window_size[0] * self.window_size[1], self.
                    num_heads, C // self.num_heads), (k_tl, k_tr, k_bl, k_br))
            (v_tl_windows, v_tr_windows, v_bl_windows, v_br_windows) = map(
                lambda t: window_partition(t, self.window_size).view(
                    -1, T, self.window_size[0] * self.window_size[1], self.
                    num_heads, C // self.num_heads), (v_tl, v_tr, v_bl, v_br))
            k_rolled = torch.cat(
                (k_tl_windows, k_tr_windows, k_bl_windows, k_br_windows),
                2).permute(0, 3, 1, 2, 4).contiguous()
            v_rolled = torch.cat(
                (v_tl_windows, v_tr_windows, v_bl_windows, v_br_windows),
                2).permute(0, 3, 1, 2, 4).contiguous()

            # mask out tokens in current window
            k_rolled = k_rolled[:, :, :, self.valid_ind_rolled]
            v_rolled = v_rolled[:, :, :, self.valid_ind_rolled]
            temp_N = k_rolled.shape[3]
            k_rolled = k_rolled.view(-1, self.num_heads, T * temp_N,
                                     C // self.num_heads)
            v_rolled = v_rolled.view(-1, self.num_heads, T * temp_N,
                                     C // self.num_heads)
            k_rolled = torch.cat((k_windows, k_rolled), 2)
            v_rolled = torch.cat((v_windows, v_rolled), 2)
        else:
            k_rolled = k_windows
            v_rolled = v_windows

        # q(k/v)_windows shape : [16, 4, 225, 128]
        # k_rolled.shape : [16, 4, 5, 165, 128]
        # ideal expanded window size 153 ((5+2*2)*(9+2*4))
        # k_windows=45 expand_window=108 overlap_window=12 (since expand_size < window_size / 2)

        if self.pool_method != "none" and self.focal_level > 1:
            k_pooled = []
            v_pooled = []
            for k in range(self.focal_level - 1):
                stride = 2**k
                # B, T, nWh, nWw, C
                x_window_pooled = x_all[k + 1].permute(0, 3, 1, 2,
                                                       4).contiguous()

                nWh, nWw = x_window_pooled.shape[2:4]

                # generate mask for pooled windows
                mask = x_window_pooled.new(T, nWh, nWw).fill_(1)
                # unfold mask: [nWh*nWw//s//s, k*k, 1]
                unfolded_mask = self.unfolds[k](mask.unsqueeze(1)).view(
                    1, T, self.unfolds[k].kernel_size[0], self.unfolds[k].kernel_size[1], -1).permute(4, 1, 2, 3, 0).contiguous().\
                    view(nWh*nWw // stride // stride, -1, 1)

                if k > 0:
                    valid_ind_unfold_k = getattr(
                        self, "valid_ind_unfold_{}".format(k))
                    unfolded_mask = unfolded_mask[:, valid_ind_unfold_k]

                x_window_masks = unfolded_mask.flatten(1).unsqueeze(0)
                x_window_masks = x_window_masks.masked_fill(
                    x_window_masks == 0,
                    float(-100.0)).masked_fill(x_window_masks > 0, float(0.0))
                mask_all[k + 1] = x_window_masks

                # generate k and v for pooled windows
                qkv_pooled = self.qkv(x_window_pooled).reshape(
                    B, T, nWh, nWw, 3, C).permute(4, 0, 1, 5, 2,
                                                  3).view(3, -1, C, nWh,
                                                          nWw).contiguous()
                # B*T, C, nWh, nWw
                k_pooled_k, v_pooled_k = qkv_pooled[1], qkv_pooled[2]
                # k_pooled_k shape: [5, 512, 4, 4]
                # self.unfolds[k](k_pooled_k) shape: [5, 23040 (512 * 5 * 9 ), 16]

                (k_pooled_k, v_pooled_k) = map(
                    lambda t: self.unfolds[k]
                    (t).view(B, T, C, self.unfolds[k].kernel_size[0], self.
                             unfolds[k].kernel_size[1], -1)
                    .permute(0, 5, 1, 3, 4, 2).contiguous().view(
                        -1, T, self.unfolds[k].kernel_size[0] * self.unfolds[
                            k].kernel_size[1], self.num_heads, C // self.
                        num_heads).permute(0, 3, 1, 2, 4).contiguous(),
                    # (B x (nH*nW)) x nHeads x T x (unfold_wsize x unfold_wsize) x head_dim
                    (k_pooled_k, v_pooled_k))
                # k_pooled_k shape : [16, 4, 5, 45, 128]

                # select valid unfolding index
                if k > 0:
                    (k_pooled_k, v_pooled_k) = map(
                        lambda t: t[:, :, :, valid_ind_unfold_k],
                        (k_pooled_k, v_pooled_k))

                k_pooled_k = k_pooled_k.view(
                    -1, self.num_heads, T * self.unfolds[k].kernel_size[0] *
                    self.unfolds[k].kernel_size[1], C // self.num_heads)
                v_pooled_k = v_pooled_k.view(
                    -1, self.num_heads, T * self.unfolds[k].kernel_size[0] *
                    self.unfolds[k].kernel_size[1], C // self.num_heads)

                k_pooled += [k_pooled_k]
                v_pooled += [v_pooled_k]

            # k_all (v_all) shape : [16, 4, 5 * 210, 128]
            k_all = torch.cat([k_rolled] + k_pooled, 2)
            v_all = torch.cat([v_rolled] + v_pooled, 2)
        else:
            k_all = k_rolled
            v_all = v_rolled

        N = k_all.shape[-2]
        q_windows = q_windows * self.scale
        # B*nW, nHead, T*window_size*window_size, T*focal_window_size*focal_window_size
        attn = (q_windows @ k_all.transpose(-2, -1))
        # T * 45
        window_area = T * self.window_size[0] * self.window_size[1]
        # T * 165
        window_area_rolled = k_rolled.shape[2]

        if self.pool_method != "none" and self.focal_level > 1:
            offset = window_area_rolled
            for k in range(self.focal_level - 1):
                # add attentional mask
                # mask_all[1] shape [1, 16, T * 45]

                bias = tuple((i + 2**k - 1) for i in self.focal_window)

                if mask_all[k + 1] is not None:
                    attn[:, :, :window_area, offset:(offset + (T*bias[0]*bias[1]))] = \
                        attn[:, :, :window_area, offset:(offset + (T*bias[0]*bias[1]))] + \
                        mask_all[k+1][:, :, None, None, :].repeat(
                            attn.shape[0] // mask_all[k+1].shape[1], 1, 1, 1, 1).view(-1, 1, 1, mask_all[k+1].shape[-1])

                offset += T * bias[0] * bias[1]

        if mask_all[0] is not None:
            nW = mask_all[0].shape[0]
            attn = attn.view(attn.shape[0] // nW, nW, self.num_heads,
                             window_area, N)
            attn[:, :, :, :, :
                 window_area] = attn[:, :, :, :, :window_area] + mask_all[0][
                     None, :, None, :, :]
            attn = attn.view(-1, self.num_heads, window_area, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        x = (attn @ v_all).transpose(1, 2).reshape(attn.shape[0], window_area,
                                                   C)
        x = self.proj(x)

        # memory ability
        if self.memory:
            if self.cross_att:
                # 将从记忆中查询到的特征与当前特征融合
                # if (len(self.m_k) == 0) and (self.max_len != 1):
                #     # 没有记忆的时候不需要融合
                #     pass
                # else:
                #     # 有记忆的时候需要聚合，并且当最长记忆时长为1时，还会在没有记忆的时候与自己做self-attention增强

                res_x = self.fusion_proj(torch.cat((x, cm_x_final.reshape(attn.shape[0], window_area,
                                                   C)), dim=2))     # 这里cm_x_final的形状调整到和默认的x一致
                x = x + res_x

            # 缓存更新过的记忆张量
            if not self.sub_token_align:
                # 存储没对齐或者token级别对齐的记忆
                try:
                    self.m_k[-1] = cm_k.detach()
                    self.m_v[-1] = cm_v.detach()
                except:
                    # 第一帧的时候记忆张量list为空，需要保证list除了最后一个元素，其他元素都是压缩过的
                    self.m_k.append(cm_k.detach())
                    self.m_v.append(cm_v.detach())
            else:
                # 存储sub-token级别对齐的记忆
                try:
                    self.m_k[-1] = cm_kk_align.detach()
                    self.m_v[-1] = cm_vv_align.detach()
                except:
                    # 第一帧的时候记忆张量list为空，需要保证list除了最后一个元素，其他元素都是压缩过的
                    self.m_k.append(cm_kk_align.detach())
                    self.m_v.append(cm_vv_align.detach())

            # 缓存当前时刻还没被压缩过的记忆张量，会在下一个时刻被压缩
            if not self.store_lf:
                # 局部帧和非局部帧都会被缓存
                self.m_k.append(k_temp.detach())    # debug
                self.m_v.append(v.detach())
            else:
                # 只缓存局部帧
                self.m_k.append(k_lf.detach())
                self.m_v.append(v_lf.detach())

            # 保持记忆力的最大长度
            if len(self.m_k) > self.max_len:
                self.m_k.pop(0)
                self.m_v.pop(0)

            # # 清除缓存的梯度
            # for mem_k, mem_v in zip(self.m_k, self.m_v):
            #     mem_k.requires_grad = False
            #     mem_v.requires_grad = False

        return x


class TemporalFocalTransformerBlock(nn.Module):
    r""" Temporal Focal Transformer Block.
    Args:
        dim (int): Number of input channels. Equal to hidden dim.
        num_heads (int): Number of attention heads.
        window_size (tuple[int]): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        focal_level (int):  The number level of focal window.
        focal_window (int):  Window size of each focal window.
        n_vecs (int): Required for F3N.
        t2t_params (int): T2T parameters for F3N.
    Revised by Hao:
        Add token fusion support and memory ability.
        token_fusion (bool):  Required for Token Fusion Manner.
        memory (bool): Required for memory ability. Using WindowAttentionMem replace the original WindowAttention.
        max_mem_len (int):  Max memory length. Unit: Forward.
        compression_factor (int):  Memory compression factor on channel dimension.
        mem_pool (bool): Whether use pooling to reduce memory spatial size.
        store_lf (bool): If True, only local frames will be cached in the memory. Only work with mem_pool=False.
        align_cache (bool): If True, memory cache will be aligned to current frames before fusion.
                            Only work with mem_pool=False and store_lf=True.
        sub_token_align (bool): If True, memory cache will be aligned at sub-token resolution.
        sub_factor (int): How many groups of sub-token alignment.
        cross_att (bool): Whether use cross attention to align memory and current token.
        time_att (bool): If True, use cross attention to align memory and current token additionally on T dimension.
        time_deco (bool): If True, the Time and Space Cross Att. will be decoupled to reduce cost.
        temp_focal (bool): If True, use temporal focal att to cross att time and space.
        cs_win (bool): If True, use cswin att to cross att time and space.
        mem_att (bool): If True, use cross att to fuse different memory with current feat instead of linear and att.
        cs_focal (bool): If True, use focal mech to upgrade cs win att.
        cs_focal_v2 (bool): If True, upgrade cswin att with same direction sliding window of pooled feat,
                            Only work with cs_focal=True.
        cs_win_strip (int): cs win attention strip width. Default: 1.
    """
    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(5, 9),
                 mlp_ratio=4.,
                 qkv_bias=True,
                 pool_method="fc",
                 focal_level=2,
                 focal_window=(5, 9),
                 norm_layer=nn.LayerNorm,
                 n_vecs=None,
                 t2t_params=None,
                 token_fusion=False,
                 memory=False,
                 max_mem_len=4,
                 compression_factor=4,
                 mem_pool=False,
                 store_lf=False,
                 align_cache=False,
                 sub_token_align=False,
                 sub_factor=1,
                 cross_att=False,
                 time_att=False,
                 time_deco=False,
                 temp_focal=False,
                 cs_win=False,
                 mem_att=False,
                 cs_focal=False,
                 cs_focal_v2=False,
                 cs_win_strip=1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.expand_size = tuple(i // 2 for i in window_size)  # 窗口大小除以2是拓展大小
        self.mlp_ratio = mlp_ratio
        self.pool_method = pool_method
        self.focal_level = focal_level
        self.focal_window = focal_window

        self.token_fusion = token_fusion
        self.memory = memory

        self.window_size_glo = self.window_size

        self.pool_layers = nn.ModuleList()
        if self.pool_method != "none":
            for k in range(self.focal_level - 1):
                window_size_glo = tuple(
                    math.floor(i / (2**k)) for i in self.window_size_glo)
                self.pool_layers.append(
                    nn.Linear(window_size_glo[0] * window_size_glo[1], 1))
                self.pool_layers[-1].weight.data.fill_(
                    1. / (window_size_glo[0] * window_size_glo[1]))
                self.pool_layers[-1].bias.data.fill_(0)

        self.norm1 = norm_layer(dim)

        if not self.memory:
            # 使用默认的window attention
            self.attn = WindowAttention(dim,
                                        expand_size=self.expand_size,
                                        window_size=self.window_size,
                                        focal_window=focal_window,
                                        focal_level=focal_level,
                                        num_heads=num_heads,
                                        qkv_bias=qkv_bias,
                                        pool_method=pool_method)
        else:
            # 使用记忆增强的window attention
            self.attn = WindowAttentionMem(dim,
                                           expand_size=self.expand_size,
                                           window_size=self.window_size,
                                           focal_window=focal_window,
                                           focal_level=focal_level,
                                           num_heads=num_heads,
                                           qkv_bias=qkv_bias,
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
                                           cs_win_strip=cs_win_strip)

        self.norm2 = norm_layer(dim)
        self.mlp = FusionFeedForward(dim, n_vecs=n_vecs, t2t_params=t2t_params)

    def forward(self, x):

        # if self.memory:
        #     # 记忆力需要额外传入局部帧的时间长度
        l_t = x[2]

        output_size = x[1]
        x = x[0]

        B, T, H, W, C = x.shape

        shortcut = x
        x = self.norm1(x)

        shifted_x = x

        x_windows_all = [shifted_x]
        x_window_masks_all = [None]

        # partition windows tuple(i // 2 for i in window_size)
        if self.focal_level > 1 and self.pool_method != "none":
            # if we add coarser granularity and the pool method is not none
            for k in range(self.focal_level - 1):
                window_size_glo = tuple(
                    math.floor(i / (2**k)) for i in self.window_size_glo)
                pooled_h = math.ceil(H / window_size_glo[0]) * (2**k)
                pooled_w = math.ceil(W / window_size_glo[1]) * (2**k)
                H_pool = pooled_h * window_size_glo[0]
                W_pool = pooled_w * window_size_glo[1]

                x_level_k = shifted_x
                # trim or pad shifted_x depending on the required size
                if H > H_pool:
                    trim_t = (H - H_pool) // 2
                    trim_b = H - H_pool - trim_t
                    x_level_k = x_level_k[:, :, trim_t:-trim_b]
                elif H < H_pool:
                    pad_t = (H_pool - H) // 2
                    pad_b = H_pool - H - pad_t
                    x_level_k = F.pad(x_level_k, (0, 0, 0, 0, pad_t, pad_b))

                if W > W_pool:
                    trim_l = (W - W_pool) // 2
                    trim_r = W - W_pool - trim_l
                    x_level_k = x_level_k[:, :, :, trim_l:-trim_r]
                elif W < W_pool:
                    pad_l = (W_pool - W) // 2
                    pad_r = W_pool - W - pad_l
                    x_level_k = F.pad(x_level_k, (0, 0, pad_l, pad_r))

                x_windows_noreshape = window_partition_noreshape(
                    x_level_k.contiguous(), window_size_glo
                )  # B, nWh, nWw, T, window_size_h, window_size_w, C
                nWh, nWw = x_windows_noreshape.shape[1:3]
                x_windows_noreshape = x_windows_noreshape.view(
                    B, nWh, nWw, T, window_size_glo[0] * window_size_glo[1],
                    C).transpose(4, 5)  # B, nWh, nWw, T, C, window_size_h*window_size_w
                x_windows_pooled = self.pool_layers[k](
                    x_windows_noreshape).flatten(-2)  # B, nWh, nWw, T, C | window被池化聚合

                x_windows_all += [x_windows_pooled]
                x_window_masks_all += [None]

        # nW*B, T*window_size*window_size, C
        if not self.memory:
            # default
            attn_windows = self.attn(x_windows_all, mask_all=x_window_masks_all)
        else:
            # memory build in, with l_t as input
            attn_windows = self.attn(x_windows_all, mask_all=x_window_masks_all, l_t=l_t)

        # merge windows
        attn_windows = attn_windows.view(-1, T, self.window_size[0],
                                         self.window_size[1], C)    # _, T, nWh, nWw, C
        shifted_x = window_reverse(attn_windows, self.window_size, T, H,
                                   W)  # B T H' W' C, 从window格式变回token格式

        # FFN
        x = shortcut + shifted_x
        y = self.norm2(x)
        if not self.token_fusion:
            # default manner
            x = x + self.mlp(y.view(B, T * H * W, C), output_size).view(
                B, T, H, W, C)
        else:
            x = x + self.mlp(y.view(B, T * H * W, C), (H * 3, W * 3)).view(
                B, T, H, W, C)

        # if self.memory:
        #     # 记忆力需要额外传入局部帧的时间长度
        return x, output_size, l_t
        # else:
        #     # default
        #     return x, output_size


class Decoupled3DFocalAttentionMem(nn.Module):
    """Decoupled 3D Focal attention with memory built in, by Hao."""
    def __init__(self, dim, expand_size, window_size, focal_window,
                 focal_level, num_heads, qkv_bias,
                 memory, max_mem_len, compression_factor, mem_pool, store_lf, align_cache, sub_token_align, sub_factor,
                 cross_att, time_att, time_deco, temp_focal, cs_win, mem_att, cs_focal, cs_focal_v2, cs_win_strip,
                 conv_path, cs_sw, pool_strip, pool_sw):

        super().__init__()
        self.dim = dim
        self.expand_size = expand_size
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.focal_level = focal_level
        self.focal_window = focal_window

        self.memory = memory
        self.mem_pool = mem_pool
        self.store_lf = store_lf
        self.align_cache = align_cache
        self.sub_token_align = sub_token_align
        self.cross_att = cross_att
        self.time_att = time_att
        self.time_deco = time_deco
        self.temp_focal = temp_focal
        self.cs_win = cs_win
        self.mem_att = mem_att
        self.cs_focal = cs_focal
        self.cs_focal_v2 = cs_focal_v2
        self.conv_path = conv_path
        self.cs_sw = cs_sw
        self.pool_strip = pool_strip
        self.pool_sw = pool_sw

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        if self.memory:
            self.m_k = []  # 缓存的memory keys
            self.m_v = []  # 缓存的memory values
            self.max_len = max_mem_len  # 缓存memory的最大记忆长度
            self.compression_factor = compression_factor  # 缓存memory的通道压缩因子

            if not self.mem_pool:
                # memory机制的含参数运算层-[基于通道的压缩]
                # 兼容局部非局部都存储和只存储局部帧的行为
                self.f_k = nn.Linear(dim, dim // self.compression_factor, bias=True)  # 用于更新上一时刻的k并压缩之前的记忆张量
                self.f_v = nn.Linear(dim, dim // self.compression_factor, bias=True)  # 用于更新上一时刻的v并压缩之前的记忆张量

                if not self.cross_att:
                    # 使用线性层聚合时需要这些层
                    self.lin_q = nn.Linear(dim, dim, bias=True)  # 用于把当前的q转换为适合查找记忆力的q
                    self.lin_k = nn.Linear(
                        dim + dim // self.compression_factor * self.max_len,
                        dim, bias=True)  # 用于把记忆里的k和当前的k进行融合
                    self.lin_v = nn.Linear(
                        dim + dim // self.compression_factor * self.max_len,
                        dim, bias=True)  # 用于把记忆里的v和当前的v进行融合

                if self.cross_att:
                    # 使用cross attention对齐记忆缓存和当前帧

                    # 当记忆时间大于1时，需要先将缓存里的记忆压缩到和当前迭代同样尺度，才能做attention.
                    if not self.mem_att:
                        # 使用线性层聚合不同时间的记忆，然后和当前做cross att
                        if self.max_len > 1:
                            self.lin_k = nn.Linear(
                                dim // self.compression_factor * self.max_len,
                                dim, bias=qkv_bias)  # 用于把记忆里的k和当前的k进行融合
                            self.lin_v = nn.Linear(
                                dim // self.compression_factor * self.max_len,
                                dim, bias=qkv_bias)  # 用于把记忆里的v和当前的v进行融合
                    else:
                        # 使用cross att聚合不同时间的记忆和当前特征
                        pass

                    # 将记忆查询输出和当前帧的输出融合
                    self.fusion_proj = nn.Linear(2 * dim, dim)
                    if not (self.temp_focal or self.cs_win):
                        # 使用标准的attention
                        self.cm_proj = nn.Linear(dim, dim)

                        if self.time_deco:
                            # 解耦时间和空间注意力
                            self.cm_proj_t = nn.Linear(dim, dim)

                    elif self.temp_focal:
                        # 使用temp focal cross attention
                        self.cf_att = CrossFocalAttention(dim,
                                                          expand_size=self.expand_size,
                                                          window_size=self.window_size,
                                                          focal_window=focal_window,
                                                          focal_level=focal_level,
                                                          num_heads=num_heads,
                                                          qkv_bias=qkv_bias,
                                                          pool_method='fc')
                    elif self.cs_win:
                        # 使用cs win attention
                        window_stride = 4  # 每个window在两个方向上占用了多少个token
                        split_size = cs_win_strip  # 条形窗口的宽度
                        num_heads_cs = num_heads//2

                        patches_resolution = [self.window_size[0] * window_stride,
                                              self.window_size[1] * window_stride]     # token的纵向和横向的个数
                        if not self.time_deco:
                            # 把时间和空间窗口合并进行3D cross attention
                            self.cs_att = nn.ModuleList([
                                TemporalLePEAttention(dim//2, resolution=patches_resolution, idx=i,
                                                      split_size=split_size, num_heads=num_heads_cs, dim_out=dim//2,
                                                      qk_scale=None, attn_drop=0., proj_drop=0.,
                                                      temporal=True, cs_focal=cs_focal, cs_focal_v2=cs_focal_v2,
                                                      cs_sw=cs_sw, pool_strip=pool_strip, pool_sw=pool_sw)
                                for i in range(0, 2)])      # 两个，一个横向一个纵向
                        elif self.time_deco:
                            # 解耦时间和空间注意力
                            self.cs_att = nn.ModuleList([
                                TemporalLePEAttention(dim//2, resolution=patches_resolution, idx=i,
                                                      split_size=split_size, num_heads=num_heads_cs, dim_out=dim//2,
                                                      qk_scale=None, attn_drop=0., proj_drop=0.,
                                                      temporal=False, cs_focal=cs_focal, cs_focal_v2=cs_focal_v2,
                                                      cs_sw=cs_sw, pool_strip=pool_strip, pool_sw=pool_sw)
                                for i in range(0, 2)])      # 两个，一个横向一个纵向
                            # 时间attention的线性层
                            self.cm_proj_t = nn.Linear(dim, dim)
                        # cs win 的线性层
                        self.cs_proj = nn.Linear(dim, dim)

                        # 是否额外给cswin cross attention增加一个CONV Path
                        if self.conv_path:
                            self.parallel_conv_cross = nn.Sequential(
                                nn.Hardswish(inplace=False),
                                nn.Conv2d(
                                    dim,
                                    dim,
                                    kernel_size=3,
                                    padding=1,
                                    groups=dim,
                                ),
                            )

            else:
                # memory机制的含参数运算层-[基于池化的压缩]
                # 全连接池化层 如果使用token缩减，将4替换为缩减比例*4
                self.pool_q = nn.Linear(self.focal_window[0] * 4 * self.focal_window[1] * 4,
                                        math.floor(self.focal_window[0] * 4 / self.compression_factor) *
                                        math.floor(self.focal_window[1] * 4 / self.compression_factor), bias=True)
                self.pool_q.weight.data.fill_(
                    1. / (self.focal_window[0] * 4 * self.focal_window[1] * 4))
                self.pool_q.bias.data.fill_(0)
                self.pool_k = nn.Linear(self.focal_window[0] * 4 * self.focal_window[1] * 4,
                                        math.floor(self.focal_window[0] * 4 / self.compression_factor) *
                                        math.floor(self.focal_window[1] * 4 / self.compression_factor), bias=True)
                self.pool_k.weight.data.fill_(
                    1. / (self.focal_window[0] * 4 * self.focal_window[1] * 4))
                self.pool_k.bias.data.fill_(0)
                self.pool_v = nn.Linear(self.focal_window[0] * 4 * self.focal_window[1] * 4,
                                        math.floor(self.focal_window[0] * 4 / self.compression_factor) *
                                        math.floor(self.focal_window[1] * 4 / self.compression_factor), bias=True)
                self.pool_v.weight.data.fill_(
                    1. / (self.focal_window[0] * 4 * self.focal_window[1] * 4))
                self.pool_v.bias.data.fill_(0)

                self.f_k = nn.Linear(dim, dim // self.compression_factor, bias=True)  # 用于更新上一时刻的k并压缩之前的记忆张量
                self.f_v = nn.Linear(dim, dim // self.compression_factor, bias=True)  # 用于更新上一时刻的v并压缩之前的记忆张量

                # self.f_k = nn.Linear(dim, dim, bias=True)  # 用于更新上一时刻的k
                # self.f_v = nn.Linear(dim, dim, bias=True)  # 用于更新上一时刻的v

                self.lin_q = nn.Linear(dim, dim, bias=True)  # 用于把当前的q转换为适合查找记忆力的q
                self.lin_k = nn.Linear(
                    int(dim + dim // self.compression_factor * self.max_len),
                    dim, bias=True)  # 用于把记忆里的k和当前的k进行融合
                self.lin_v = nn.Linear(
                    int(dim + dim // self.compression_factor * self.max_len),
                    dim, bias=True)  # 用于把记忆里的v和当前的v进行融合

                self.unpool_q = nn.Linear(self.focal_window[0] * 4 * self.focal_window[1] * 4 +
                                          math.floor(self.focal_window[0] * 4 / self.compression_factor) *
                                          math.floor(self.focal_window[1] * 4 / self.compression_factor),
                                          self.focal_window[0] * 4 * self.focal_window[1] * 4, bias=True)
                self.unpool_k = nn.Linear(self.focal_window[0] * 4 * self.focal_window[1] * 4 +
                                          math.floor(self.focal_window[0] * 4 / self.compression_factor) *
                                          math.floor(self.focal_window[1] * 4 / self.compression_factor),
                                          self.focal_window[0] * 4 * self.focal_window[1] * 4, bias=True)
                self.unpool_v = nn.Linear(self.focal_window[0] * 4 * self.focal_window[1] * 4 +
                                          math.floor(self.focal_window[0] * 4 / self.compression_factor) *
                                          math.floor(self.focal_window[1] * 4 / self.compression_factor),
                                          self.focal_window[0] * 4 * self.focal_window[1] * 4, bias=True)

            if self.align_cache:
                # 使用光流对齐缓存
                if not self.sub_token_align:
                    self.flow_head = FlowHead(input_dim=(dim + dim // self.compression_factor), hidden_factor=2)
                else:
                    self.sub_factor = sub_factor
                    self.flow_head = FlowHead(
                        input_dim=(dim // self.sub_factor + (dim // self.compression_factor) // self.sub_factor),
                        hidden_factor=2)

                # 缓存对齐的记忆张量防止两次backward错误
                self.m_k_aligned = []  # 缓存的已对齐memory keys
                self.m_v_aligned = []  # 缓存的已对齐memory keys

        # ====使用 cs win attention 作为self attention====
        window_stride = 4  # 每个window在两个方向上占用了多少个token
        split_size = cs_win_strip  # 条形窗口的宽度
        num_heads_cs = num_heads // 2

        patches_resolution = [self.window_size[0] * window_stride,
                              self.window_size[1] * window_stride]  # token的纵向和横向的个数
        if not self.time_deco:
            # 把时间和空间窗口合并进行3D cross attention
            self.self_attn = nn.ModuleList([
                TemporalLePEAttention(dim // 2, resolution=patches_resolution, idx=i,
                                      split_size=split_size, num_heads=num_heads_cs, dim_out=dim // 2,
                                      qk_scale=None, attn_drop=0., proj_drop=0.,
                                      temporal=True, cs_focal=cs_focal, cs_focal_v2=cs_focal_v2,
                                      cs_sw=cs_sw, pool_strip=pool_strip, pool_sw=pool_sw)
                for i in range(0, 2)])  # 两个，一个横向一个纵向
        elif self.time_deco:
            # 解耦时间和空间注意力
            self.self_attn = nn.ModuleList([
                TemporalLePEAttention(dim // 2, resolution=patches_resolution, idx=i,
                                      split_size=split_size, num_heads=num_heads_cs, dim_out=dim // 2,
                                      qk_scale=None, attn_drop=0., proj_drop=0.,
                                      temporal=False, cs_focal=cs_focal, cs_focal_v2=cs_focal_v2,
                                      cs_sw=cs_sw, pool_strip=pool_strip, pool_sw=pool_sw)
                for i in range(0, 2)])  # 两个，一个横向一个纵向
            # 时间attention的线性层
            self.self_proj_t = nn.Linear(dim, dim)
        # cs win 的线性层
        self.self_proj = nn.Linear(dim, dim)

        # 是否额外给cswin self attention增加一个CONV Path
        if self.conv_path:
            self.parallel_conv_self = nn.Sequential(
                nn.Hardswish(inplace=False),
                nn.Conv2d(
                    dim,
                    dim,
                    kernel_size=3,
                    padding=1,
                    groups=dim,
                ),
            )

    def forward(self, x, mask_all=None, l_t=5):
        """
        Args:
            x_all: input features with shape of (B, T, Wh, Ww, C)
            mask_all: (0/-inf) mask with shape of (num_windows, T*Wh*Ww, T*Wh*Ww) or None
            l_t: local frame nums

            output: (nW*B, Wh*Ww, C)
        """
        B, T, nH, nW, C = x.shape
        qkv = self.qkv(x).reshape(B, T, nH, nW, 3,
                                  C).permute(4, 0, 1, 2, 3, 5).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]  # B, T, nH, nW, C

        # memory ability
        if self.memory:

            if not self.mem_pool:
                # 通道压缩时序的记忆力

                if not self.store_lf:
                    # 局部和随机的非局部帧都会被存储
                    # 压缩上一个记忆缓存
                    if len(self.m_k) != 0:
                        cm_k = self.f_k(self.m_k[-1])
                        cm_v = self.f_v(self.m_v[-1])
                    else:
                        # 第一帧时没有记忆张量，使用当前帧的k，v
                        cm_k = self.f_k(k)
                        cm_v = self.f_v(v)

                    # 增强qkv
                    if not self.cross_att:
                        # 使用线性层聚合缓存和当前迭代
                        q = self.lin_q(q)
                        if len(self.m_k) == self.max_len:
                            # 因为缓存里的最后一帧还没被压缩的替换，所以不取最后一帧
                            k = self.lin_k(torch.cat((
                                torch.cat(self.m_k[:-1], dim=4), cm_k, k), dim=4))
                            v = self.lin_v(torch.cat((
                                torch.cat(self.m_v[:-1], dim=4), cm_v, v), dim=4))
                            # 后面的索引用到了k所以保存一下
                            k_temp = k
                        else:
                            repeat_k = self.max_len - len(self.m_k)
                            repeat_v = self.max_len - len(self.m_v)
                            # 尽量使用缓存中的帧，不够的使用当前帧提取的代替
                            if len(self.m_k) == 0:
                                k = self.lin_k(torch.cat((cm_k.repeat(1, 1, 1, 1, repeat_k), k), dim=4))
                                v = self.lin_v(torch.cat((cm_v.repeat(1, 1, 1, 1, repeat_v), v), dim=4))
                                k_temp = k  # debug
                            else:
                                k_rep_feat = self.f_k(k)
                                k_rep_feat = k_rep_feat.repeat(1, 1, 1, 1, repeat_k)
                                v_rep_feat = self.f_v(v)
                                v_rep_feat = v_rep_feat.repeat(1, 1, 1, 1, repeat_v)
                                k = self.lin_k(torch.cat((
                                    torch.cat(self.m_k[:-1], dim=4), cm_k, k_rep_feat, k), dim=4))
                                v = self.lin_v(torch.cat((
                                    torch.cat(self.m_v[:-1], dim=4), cm_v, v_rep_feat, v), dim=4))
                                k_temp = k  # debug

                    else:
                        # 使用cross attention聚合记忆

                        # 聚合记忆缓存，用于后续和当前特征进行cross attention
                        if self.max_len == 1:
                            # 只记忆1次迭代时，不需要聚合
                            att_num = 1
                            mem_k = cm_k
                            mem_v = cm_v
                        else:
                            # 记忆缓存时间大于1，需要聚合记忆再做attention
                            if not self.mem_att:
                                # 使用线性层聚合记忆
                                # 只需要最后聚合的记忆和当前特征做一次cross att
                                att_num = 1
                                if len(self.m_k) == self.max_len:
                                    # 记忆缓存满了，直接用线性层聚合
                                    # 因为缓存里的最后一帧还没被压缩的替换，所以不取最后一帧
                                    mem_k = self.lin_k(torch.cat((
                                        torch.cat(self.m_k[:-1], dim=4), cm_k), dim=4))
                                    mem_v = self.lin_v(torch.cat((
                                        torch.cat(self.m_v[:-1], dim=4), cm_v), dim=4))
                                else:
                                    # 记忆缓存没满，复制一下
                                    repeat_k = self.max_len - len(self.m_k)
                                    repeat_v = self.max_len - len(self.m_v)
                                    if len(self.m_k) == 0:
                                        # 缓存里面啥也没有，当前帧多复制几次
                                        mem_k = self.lin_k(cm_k.repeat(1, 1, 1, 1, repeat_k))
                                        mem_v = self.lin_v(cm_v.repeat(1, 1, 1, 1, repeat_k))
                                    else:
                                        # 尽量使用缓存中的帧
                                        mem_k = self.lin_k(torch.cat((
                                            torch.cat(self.m_k[:-1], dim=4), cm_k.repeat(1, 1, 1, 1, repeat_k + 1)), dim=4))
                                        mem_v = self.lin_v(torch.cat((
                                            torch.cat(self.m_v[:-1], dim=4), cm_v.repeat(1, 1, 1, 1, repeat_v + 1)), dim=4))
                            else:
                                # 直接把不同时间的记忆和当前迭代分别做attention就好了
                                # attention的次数等于记忆的长度
                                if len(self.m_k) == 0:
                                    # 当缓存里是空的时，直接和当前特征自己做1次attention
                                    att_num = 1
                                else:
                                    # 当缓存里不是空的时，和缓存里的做attention
                                    att_num = len(self.m_k)

                        for att_idx in range(0, att_num):

                            # 记忆时间超过1并且不用线性层聚合才需要这些判断逻辑
                            # 也就是说，只有需要做多次cross attention才需要这些逻辑
                            if self.mem_att:
                                # 每次迭代是对不同时间的记忆做cross attention
                                if len(self.m_k) == 0:
                                    # 缓存里是空的，和更新过的自己做self attention
                                    mem_k = cm_k
                                    mem_v = cm_v
                                    pass
                                else:
                                    # 缓存里不是空的，和缓存里更新过的记忆以及当前更新的记忆做attention
                                    # 先取缓存里的记忆
                                    if att_idx != (att_num - 1):
                                        mem_k = self.m_k[:-1][att_idx]
                                        mem_v = self.m_v[:-1][att_idx]
                                    else:
                                        # 最后1次取当前压缩过的kv
                                        mem_k = cm_k
                                        mem_v = cm_v

                            # ===各种不同的cross attention选择===
                            if not self.time_att:
                                # 信息只在Nh Nw维度流动(空间维度)
                                q = q.reshape(B * T, self.num_heads, nH * nW, C // self.num_heads).contiguous()
                                mem_k = mem_k.reshape(B * T, self.num_heads, nH * nW, C // self.num_heads).contiguous()
                                mem_v = mem_v.reshape(B * T, self.num_heads, nH * nW, C // self.num_heads).contiguous()
                                cm_attn = (q @ mem_k.transpose(-2, -1)) * self.scale
                                cm_attn = cm_attn.softmax(dim=-1)
                                cm_x = (cm_attn @ mem_v).transpose(1, 2).reshape(B * T, nH * nW, C)
                                cm_x = self.cm_proj(cm_x)
                                # 恢复原来的shape
                                q = q.reshape(B, T, nH, nW, C).contiguous()
                                # mem_k = mem_k.reshape(B, T, nH, nW, C).contiguous()
                                # mem_v = mem_v.reshape(B, T, nH, nW, C).contiguous()

                            else:
                                # 信息将额外在T维度流动
                                if self.time_deco and not self.cs_win:
                                    # 解耦时间和空间注意力的vanilla attention
                                    # 时间注意力
                                    q = q.permute(0, 2, 3, 1, 4).reshape(B * nH * nW, T, C // self.num_heads, self.num_heads)\
                                        .permute(0, 3, 1, 2).contiguous()   # B*N, head, T, C//head
                                    mem_k = mem_k.permute(0, 2, 3, 1, 4).reshape(B * nH * nW, T, C // self.num_heads, self.num_heads)\
                                        .permute(0, 3, 1, 2).contiguous()
                                    mem_v = mem_v.permute(0, 2, 3, 1, 4).reshape(B * nH * nW, T, C // self.num_heads, self.num_heads)\
                                        .permute(0, 3, 1, 2).contiguous()
                                    cm_attn_t = (q @ mem_k.transpose(-2, -1)) * self.scale
                                    cm_attn_t = cm_attn_t.softmax(dim=-1)
                                    cm_x_t = (cm_attn_t @ mem_v).permute(0, 2, 3, 1).reshape(B, nH * nW, T, C)\
                                        .permute(0, 2, 1, 3).reshape(B * T, nH * nW, C)
                                    cm_x_t = self.cm_proj_t(cm_x_t)
                                    # 恢复qkv的shape
                                    q = q.permute(0, 2, 3, 1).reshape(B, nH, nW, T, C).permute(0, 3, 1, 2, 4).contiguous()
                                    mem_k = mem_k.permute(0, 2, 3, 1).reshape(B, nH, nW, T, C).permute(0, 3, 1, 2, 4).contiguous()
                                    mem_v = mem_v.permute(0, 2, 3, 1).reshape(B, nH, nW, T, C).permute(0, 3, 1, 2, 4).contiguous()

                                    # 空间注意力
                                    q = q.reshape(B * T, nH * nW, C // self.num_heads, self.num_heads).permute(0, 3, 1, 2)\
                                        .contiguous()   # BT, head, N, C//head
                                    mem_k = mem_k.reshape(B * T, nH * nW, C // self.num_heads, self.num_heads).permute(0, 3, 1, 2)\
                                        .contiguous()
                                    mem_v = mem_v.reshape(B * T, nH * nW, C // self.num_heads, self.num_heads).permute(0, 3, 1, 2)\
                                        .contiguous()
                                    cm_attn = (q @ mem_k.transpose(-2, -1)) * self.scale
                                    cm_attn = cm_attn.softmax(dim=-1)
                                    cm_x = (cm_attn @ mem_v).transpose(1, 2).reshape(B * T, nH * nW, C)
                                    cm_x = self.cm_proj(cm_x)
                                    # 恢复qkv的shape
                                    q = q.permute(0, 2, 3, 1).reshape(B, T, nH, nW, C).contiguous()
                                    # mem_k = mem_k.permute(0, 2, 3, 1).reshape(B, T, nH, nW, C).contiguous()
                                    # mem_v = mem_v.permute(0, 2, 3, 1).reshape(B, T, nH, nW, C).contiguous()

                                    # 暂时只使用相加融合两次查询
                                    cm_x += cm_x_t

                                elif self.temp_focal:
                                    # 基于temporal focal attention实现时空记忆聚合
                                    cm_x = self.cf_att(qkv=[q, mem_k, mem_v], mask_all=mask_all)

                                elif self.cs_win:
                                    # ===基于cswin attention聚合时空记忆和当前迭代===

                                    if self.time_deco:
                                        # 解耦时间和空间聚合，时间聚合使用vanilla attention
                                        q = q.permute(0, 2, 3, 1, 4).reshape(B * nH * nW, T, C // self.num_heads,
                                                                             self.num_heads) \
                                            .permute(0, 3, 1, 2).contiguous()  # B*N, head, T, C//head
                                        mem_k = mem_k.permute(0, 2, 3, 1, 4).reshape(B * nH * nW, T, C // self.num_heads,
                                                                                     self.num_heads) \
                                            .permute(0, 3, 1, 2).contiguous()
                                        mem_v = mem_v.permute(0, 2, 3, 1, 4).reshape(B * nH * nW, T, C // self.num_heads,
                                                                                     self.num_heads) \
                                            .permute(0, 3, 1, 2).contiguous()
                                        cm_attn_t = (q @ mem_k.transpose(-2, -1)) * self.scale
                                        cm_attn_t = cm_attn_t.softmax(dim=-1)
                                        cm_x_t = (cm_attn_t @ mem_v).permute(0, 2, 3, 1).reshape(B, nH * nW, T, C) \
                                            .permute(0, 2, 1, 3).reshape(B * T, nH * nW, C)
                                        cm_x_t = self.cm_proj_t(cm_x_t)
                                        # 恢复qkv的shape
                                        q = q.permute(0, 2, 3, 1).reshape(B, nH, nW, T, C).permute(0, 3, 1, 2,
                                                                                                   4).contiguous()
                                        mem_k = mem_k.permute(0, 2, 3, 1).reshape(B, nH, nW, T, C).permute(0, 3, 1, 2,
                                                                                                           4).contiguous()
                                        mem_v = mem_v.permute(0, 2, 3, 1).reshape(B, nH, nW, T, C).permute(0, 3, 1, 2,
                                                                                                           4).contiguous()

                                    # 是否加上CONV Path
                                    if self.conv_path:
                                        # 记忆v的short path
                                        mem_v_conv = \
                                            self.parallel_conv_cross(mem_v.reshape(B*T, nH, nW, C).permute(0, 3, 1, 2)
                                                                     .contiguous())
                                        # reshape mem_v_conv B*T, C, H, W -> B, T, H, W, C
                                        mem_v_conv = mem_v_conv.permute(0, 2, 3, 1).contiguous()\
                                            .reshape(B, T * nH * nW, C)

                                    cm_x1 = self.cs_att[0](qkv=[q[:, :, :, :, :C // 2],
                                                                mem_k[:, :, :, :, :C // 2],
                                                                mem_v[:, :, :, :, :C // 2]])
                                    cm_x2 = self.cs_att[1](qkv=[q[:, :, :, :, C // 2:],
                                                                mem_k[:, :, :, :, C // 2:],
                                                                mem_v[:, :, :, :, C // 2:]])
                                    cm_x = torch.cat([cm_x1, cm_x2], dim=2)

                                    # 是否加上CONV Path
                                    if self.conv_path:
                                        cm_x = cm_x.add(mem_v_conv)

                                    cm_x = self.cs_proj(cm_x)

                                    if self.time_deco:
                                        # 暂时只使用相加融合两次查询
                                        cm_x += cm_x_t.reshape(B, T * nH * nW, C)

                                else:
                                    # 不解耦时间和空间的vanilla attention
                                    q = q.reshape(B, self.num_heads, T * nH * nW, C // self.num_heads).contiguous()
                                    mem_k = mem_k.reshape(B, self.num_heads, T * nH * nW, C // self.num_heads).contiguous()
                                    mem_v = mem_v.reshape(B, self.num_heads, T * nH * nW, C // self.num_heads).contiguous()
                                    cm_attn = (q @ mem_k.transpose(-2, -1)) * self.scale
                                    cm_attn = cm_attn.softmax(dim=-1)
                                    cm_x = (cm_attn @ mem_v).transpose(1, 2).reshape(B * T, nH * nW, C)
                                    cm_x = self.cm_proj(cm_x)

                                    # 恢复原来的shape
                                    q = q.reshape(B, T, nH, nW, C).contiguous()
                                    # mem_k = mem_k.reshape(B, T, nH, nW, C).contiguous()
                                    # mem_v = mem_v.reshape(B, T, nH, nW, C).contiguous()

                                    # 除了解耦的，前面两个版本的qkv shape变换都需要检查

                            # cm_x_final用来存储不同时间记忆attention的结果
                            if att_idx == 0:
                                # 第一次直接初始化cm_x_final
                                cm_x_final = cm_x
                            else:
                                cm_x_final += cm_x

                        k_temp = k  # debug

                else:
                    # 仅存储局部帧的记忆
                    q_lf = q[:, :l_t, ...]
                    k_lf = k[:, :l_t, ...]
                    v_lf = v[:, :l_t, ...]

                    # 压缩上一个记忆缓存
                    if len(self.m_k) != 0:
                        cm_k = self.f_k(self.m_k[-1])
                        cm_v = self.f_v(self.m_v[-1])
                    else:
                        # 第一帧时没有记忆张量，使用当前的局部帧k，v
                        cm_k = self.f_k(k_lf)
                        cm_v = self.f_v(v_lf)

                    if self.align_cache:
                        # 在增强前将缓存里面所有的记忆与当前迭代的k v对齐
                        # 在记忆缓存的最后一帧被压缩后进行对齐
                        cm_k = cm_k.reshape(B * l_t, C // self.compression_factor, nH, nW)  # B*Lt, C_compress, nH, nW
                        cm_v = cm_v.reshape(B * l_t, C // self.compression_factor, nH, nW)  # B*Lt, C_compress, nH, nW
                        k_lf = k_lf.reshape(B * l_t, C, nH, nW)                             # B*Lt, C, nH, nW
                        v_lf = v_lf.reshape(B * l_t, C, nH, nW)                             # B*Lt, C, nH, nW

                        if not self.sub_token_align:
                            # 在token尺度估计光流完成对齐
                            token_flow_k = self.flow_head(torch.cat((cm_k, k_lf), dim=1)).reshape(B * l_t, nH, nW, 2)
                            token_flow_v = self.flow_head(torch.cat((cm_v, v_lf), dim=1)).reshape(B * l_t, nH, nW, 2)

                            cm_k = flow_warp(cm_k, token_flow_k)                             # B*Lt, C_compress, nH, nW
                            cm_v = flow_warp(cm_v, token_flow_v)                             # B*Lt, C_compress, nH, nW

                            cm_k = cm_k.reshape(B, l_t, nH, nW, C // self.compression_factor)
                            cm_v = cm_v.reshape(B, l_t, nH, nW, C // self.compression_factor)
                        else:
                            # 在sub-token尺度估计光流完成对齐
                            group_stride = C // self.sub_factor
                            group_stride_compressed = group_stride // self.compression_factor
                            cm_kk_list = []  # 防止两次梯度反串报错
                            cm_vv_list = []
                            # kk_lf_list = []  # 存储sub-token的kk_lf加快速度
                            for group_idx in range(0, self.sub_factor):
                                # 取出当前group的sub-token
                                cm_kk = cm_k[:,
                                        group_stride_compressed * group_idx:group_stride_compressed * (group_idx + 1),
                                        :, :]
                                cm_vv = cm_v[:,
                                        group_stride_compressed * group_idx:group_stride_compressed * (group_idx + 1),
                                        :, :]
                                kk_lf = k_lf[:, group_stride * group_idx:group_stride * (group_idx + 1), :, :]
                                vv_lf = v_lf[:, group_stride * group_idx:group_stride * (group_idx + 1), :, :]

                                # sub-token尺度的光流估计
                                token_flow_kk = self.flow_head(torch.cat((cm_kk, kk_lf), dim=1))\
                                    .reshape(B * l_t, nH, nW, 2)
                                token_flow_vv = self.flow_head(torch.cat((cm_vv, vv_lf), dim=1))\
                                    .reshape(B * l_t, nH, nW, 2)

                                # sub-token尺度的光流warp对齐
                                cm_kk = flow_warp(cm_kk, token_flow_kk)     # B*Lt, C_compress/sub_factor, nH, nW
                                cm_vv = flow_warp(cm_vv, token_flow_vv)     # B*Lt, C_compress/sub_factor, nH, nW
                                cm_kk_list.append(cm_kk)
                                cm_vv_list.append(cm_vv)

                            # 重组回完整的cm_kk_align, 作用相当于cm_k
                            cm_kk_align = torch.cat(cm_kk_list, dim=1).reshape(B, l_t, nH, nW, C // self.compression_factor)
                            cm_vv_align = torch.cat(cm_vv_list, dim=1).reshape(B, l_t, nH, nW, C // self.compression_factor)

                        # 对齐缓存里的其他帧，注意因为缓存里的最后一次迭代还没被压缩，所以不需要对齐，上面的就是对齐最后一次迭代的流程
                        self.m_k_aligned = []   # 对齐的长度会比不对齐的list长度少1
                        self.m_v_aligned = []   # 之所以新创建list是为了防止2次梯度反传报错，如果使用retain graph会导致显存消耗增加

                        if not self.sub_token_align:
                            # 在token尺度对齐缓存里的所有帧
                            for cache_index in range(0, len(self.m_k)-1):
                                k_mem = self.m_k[cache_index]
                                v_mem = self.m_v[cache_index]
                                k_mem = k_mem.reshape(B * l_t, C // self.compression_factor, nH,
                                                      nW)  # B*Lt, C_compress, nH, nW
                                v_mem = v_mem.reshape(B * l_t, C // self.compression_factor, nH,
                                                      nW)  # B*Lt, C_compress, nH, nW

                                # calc token flow
                                token_flow_k = self.flow_head(torch.cat((k_mem, k_lf), dim=1)).reshape(B * l_t, nH, nW, 2)
                                token_flow_v = self.flow_head(torch.cat((v_mem, v_lf), dim=1)).reshape(B * l_t, nH, nW, 2)

                                # warp tokens
                                k_mem = flow_warp(k_mem, token_flow_k)  # B*Lt, C, nH, nW
                                v_mem = flow_warp(v_mem, token_flow_v)  # B*Lt, C, nH, nW

                                # retrieve
                                k_mem = k_mem.reshape(B, l_t, nH, nW, C // self.compression_factor)
                                v_mem = v_mem.reshape(B, l_t, nH, nW, C // self.compression_factor)
                                self.m_k_aligned.append(k_mem)
                                self.m_v_aligned.append(v_mem)
                        else:
                            # 在sub-token尺度对齐缓存里的所有帧
                            for cache_index in range(0, len(self.m_k) - 1):
                                k_mem = self.m_k[cache_index]
                                v_mem = self.m_v[cache_index]
                                k_mem = k_mem.reshape(B * l_t, C // self.compression_factor, nH,
                                                      nW)  # B*Lt, C_compress, nH, nW
                                v_mem = v_mem.reshape(B * l_t, C // self.compression_factor, nH,
                                                      nW)  # B*Lt, C_compress, nH, nW

                                kk_mem_list = []  # 防止两次梯度反传报错
                                vv_mem_list = []
                                for group_idx in range(0, self.sub_factor):
                                    # 取出当前group的sub-token
                                    kk_mem = k_mem[:,
                                             group_stride_compressed * group_idx:group_stride_compressed * (
                                                     group_idx + 1),
                                             :, :]
                                    vv_mem = v_mem[:,
                                             group_stride_compressed * group_idx:group_stride_compressed * (
                                                     group_idx + 1),
                                             :, :]
                                    kk_lf = k_lf[:, group_stride * group_idx:group_stride * (group_idx + 1), :, :]
                                    vv_lf = v_lf[:, group_stride * group_idx:group_stride * (group_idx + 1), :, :]

                                    # sub-token尺度的光流估计
                                    token_flow_kk = self.flow_head(torch.cat((kk_mem, kk_lf), dim=1)) \
                                        .reshape(B * l_t, nH, nW, 2)
                                    token_flow_vv = self.flow_head(torch.cat((vv_mem, vv_lf), dim=1)) \
                                        .reshape(B * l_t, nH, nW, 2)

                                    # sub-token尺度的光流warp对齐
                                    kk_mem = flow_warp(kk_mem, token_flow_kk)  # B*Lt, C_compress/sub_factor, nH, nW
                                    vv_mem = flow_warp(vv_mem, token_flow_vv)  # B*Lt, C_compress/sub_factor, nH, nW
                                    kk_mem_list.append(kk_mem)
                                    vv_mem_list.append(vv_mem)

                                # 重组回完整的k_mem, 作用相当于k_mem
                                k_mem = torch.cat(kk_mem_list, dim=1).reshape(B, l_t, nH, nW,
                                                                                   C // self.compression_factor)
                                v_mem = torch.cat(vv_mem_list, dim=1).reshape(B, l_t, nH, nW,
                                                                                   C // self.compression_factor)
                                self.m_k_aligned.append(k_mem)
                                self.m_v_aligned.append(v_mem)

                        # 恢复当前k v的shape
                        k_lf = k_lf.reshape(B, l_t, nH, nW, C)
                        v_lf = v_lf.reshape(B, l_t, nH, nW, C)

                    # 增强局部帧的qkv
                    q_lf = self.lin_q(q_lf)
                    if len(self.m_k) == self.max_len:
                        # 缓存满了的情况，不需要补充临时的记忆张量
                        # 因为缓存里的最后一帧还没被压缩的替换，所以不取最后一帧的缓存，直接拿压缩完的cm就可
                        if not self.align_cache:
                            # 把没对齐的和当前iter的k v融合
                            if self.max_len > 1:    # 缓存长度大于1时可以cat
                                k_lf = self.lin_k(torch.cat((
                                    torch.cat(self.m_k[:-1], dim=4), cm_k, k_lf), dim=4))
                                v_lf = self.lin_v(torch.cat((
                                    torch.cat(self.m_v[:-1], dim=4), cm_v, v_lf), dim=4))
                            else:
                                # 当最大记忆时长只有1次迭代时，压缩过后的最后一个记忆就是全部的记忆了
                                k_lf = self.lin_k(torch.cat((cm_k, k_lf), dim=4))
                                v_lf = self.lin_v(torch.cat((cm_v, v_lf), dim=4))
                        elif self.align_cache and not self.sub_token_align:
                            # 在token尺度把对齐的和当前iter的k v融合
                            if self.max_len > 1:  # 缓存长度大于1时可以cat
                                k_lf = self.lin_k(torch.cat((
                                    torch.cat(self.m_k_aligned[:], dim=4), cm_k, k_lf), dim=4))
                                v_lf = self.lin_v(torch.cat((
                                    torch.cat(self.m_v_aligned[:], dim=4), cm_v, v_lf), dim=4))
                            else:
                                # 当最大记忆时长只有1次迭代时，压缩过后的最后一个记忆就是全部的记忆了
                                k_lf = self.lin_k(torch.cat((cm_k, k_lf), dim=4))
                                v_lf = self.lin_v(torch.cat((cm_v, v_lf), dim=4))
                        else:
                            # 在sub-token尺度把对齐的和当前iter的k v融合
                            if self.max_len > 1:  # 缓存长度大于1时可以cat
                                k_lf = self.lin_k(torch.cat((
                                    torch.cat(self.m_k_aligned[:], dim=4), cm_kk_align, k_lf), dim=4))
                                v_lf = self.lin_v(torch.cat((
                                    torch.cat(self.m_v_aligned[:], dim=4), cm_vv_align, v_lf), dim=4))
                            else:
                                # 当最大记忆时长只有1次迭代时，压缩过后的最后一个记忆就是全部的记忆了
                                k_lf = self.lin_k(torch.cat((cm_kk_align, k_lf), dim=4))
                                v_lf = self.lin_v(torch.cat((cm_vv_align, v_lf), dim=4))
                    else:
                        # 缓存还没有存满，需要复制当前帧的张量
                        repeat_k = self.max_len - len(self.m_k)
                        repeat_v = self.max_len - len(self.m_v)
                        if len(self.m_k) == 0:
                            # 缓存里啥也没有，直接把当前的全复制了
                            if not self.sub_token_align:
                                # 使用未对齐的或者token尺度对齐的cm_k
                                k_lf = self.lin_k(torch.cat((cm_k.repeat(1, 1, 1, 1, repeat_k), k_lf), dim=4))
                                v_lf = self.lin_v(torch.cat((cm_v.repeat(1, 1, 1, 1, repeat_v), v_lf), dim=4))
                            else:
                                # 使用对齐的sub-token级别cm_kk_align
                                k_lf = self.lin_k(torch.cat((cm_kk_align.repeat(1, 1, 1, 1, repeat_k), k_lf), dim=4))
                                v_lf = self.lin_v(torch.cat((cm_vv_align.repeat(1, 1, 1, 1, repeat_v), v_lf), dim=4))
                        else:
                            # 尽量使用缓存中的帧，不够的使用当前帧提取的代替
                            k_rep_feat = self.f_k(k_lf)
                            k_rep_feat = k_rep_feat.repeat(1, 1, 1, 1, repeat_k)
                            v_rep_feat = self.f_v(v_lf)
                            v_rep_feat = v_rep_feat.repeat(1, 1, 1, 1, repeat_v)
                            if not self.align_cache:
                                # 把没对齐的和当前iter的k v融合
                                k_lf = self.lin_k(torch.cat((
                                    torch.cat(self.m_k[:-1], dim=4), cm_k, k_rep_feat, k_lf), dim=4))
                                v_lf = self.lin_v(torch.cat((
                                    torch.cat(self.m_v[:-1], dim=4), cm_v, v_rep_feat, v_lf), dim=4))
                            elif not self.sub_token_align:
                                # 把token级别对齐的和当前iter的k v融合
                                k_lf = self.lin_k(torch.cat((
                                    torch.cat(self.m_k_aligned[:], dim=4), cm_k, k_rep_feat, k_lf), dim=4))
                                v_lf = self.lin_v(torch.cat((
                                    torch.cat(self.m_v_aligned[:], dim=4), cm_v, v_rep_feat, v_lf), dim=4))
                            else:
                                # 把sub-token级别对齐的和当前iter的k v融合
                                k_lf = self.lin_k(torch.cat((
                                    torch.cat(self.m_k_aligned[:], dim=4), cm_kk_align, k_rep_feat, k_lf), dim=4))
                                v_lf = self.lin_v(torch.cat((
                                    torch.cat(self.m_v_aligned[:], dim=4), cm_vv_align, v_rep_feat, v_lf), dim=4))

                    # 把增强后的局部帧qkv还原到所有的qkv中
                    q[:, :l_t, ...] = q_lf
                    k[:, :l_t, ...] = k_lf
                    v[:, :l_t, ...] = v_lf

            else:
                # 空间压缩当前的q k v
                c_q = self.pool_q(q)
                c_k = self.pool_k(k)
                c_v = self.pool_v(v)

                # 通道压缩时序的记忆力
                cm_k = self.f_k(self.m_k[-1])
                cm_v = self.f_v(self.m_v[-1])

                # 增强qkv
                c_q = self.lin_q(c_q)
                c_k = self.lin_k(torch.cat((self.m_k[:], c_k), dim=4))
                c_v = self.lin_v(torch.cat((self.m_v[:], c_v), dim=4))

                # 恢复qkv的尺度，加跳跃连接
                q = self.unpool_q(torch.cat((q, c_q), dim=1))
                k = self.unpool_k(torch.cat((k, c_k), dim=1))
                v = self.unpool_v(torch.cat((v, c_v), dim=1))

            # 把q, k, v存储回qkv，后面算窗口attention会用到
            # qkv[0] = q_temp
            # qkv[1] = k_temp
            # qkv[2] = v_temp
            # qkv[0], qkv[1], qkv[2] = q_temp, k_temp, v_temp

        # self attention
        # ===基于cswin attention进行自注意力===
        # 时间自注意力
        if self.time_deco:
            # 解耦时间和空间聚合，时间聚合使用vanilla attention
            q = q.permute(0, 2, 3, 1, 4).reshape(B * nH * nW, T, C // self.num_heads,
                                                 self.num_heads) \
                .permute(0, 3, 1, 2).contiguous()  # B*N, head, T, C//head
            k = k.permute(0, 2, 3, 1, 4).reshape(B * nH * nW, T, C // self.num_heads,
                                                         self.num_heads) \
                .permute(0, 3, 1, 2).contiguous()
            v = v.permute(0, 2, 3, 1, 4).reshape(B * nH * nW, T, C // self.num_heads,
                                                         self.num_heads) \
                .permute(0, 3, 1, 2).contiguous()
            self_attn_t = (q @ k.transpose(-2, -1)) * self.scale
            self_attn_t = self_attn_t.softmax(dim=-1)
            self_x_t = (self_attn_t @ v).permute(0, 2, 3, 1).reshape(B, nH * nW, T, C) \
                .permute(0, 2, 1, 3).reshape(B * T, nH * nW, C)
            self_x_t = self.self_proj_t(self_x_t)
            # 恢复qkv的shape
            q = q.permute(0, 2, 3, 1).reshape(B, nH, nW, T, C).permute(0, 3, 1, 2, 4).contiguous()
            k = k.permute(0, 2, 3, 1).reshape(B, nH, nW, T, C).permute(0, 3, 1, 2, 4).contiguous()
            v = v.permute(0, 2, 3, 1).reshape(B, nH, nW, T, C).permute(0, 3, 1, 2, 4).contiguous()

        # 是否加上CONV Path
        if self.conv_path:
            # 当前v的short path
            v_conv = \
                self.parallel_conv_self(v.reshape(B * T, nH, nW, C).permute(0, 3, 1, 2).contiguous())
            # reshape v_conv B*T, C, H, W -> B, T, H, W, C -> B, T*H*W, C
            v_conv = v_conv.permute(0, 2, 3, 1).contiguous().reshape(B, T * nH * nW, C)

        # 空间自注意力
        self_x1 = self.self_attn[0](qkv=[q[:, :, :, :, :C // 2],
                                       k[:, :, :, :, :C // 2],
                                       v[:, :, :, :, :C // 2]])
        self_x2 = self.self_attn[1](qkv=[q[:, :, :, :, C // 2:],
                                       k[:, :, :, :, C // 2:],
                                       v[:, :, :, :, C // 2:]])
        self_x = torch.cat([self_x1, self_x2], dim=2)

        # 是否加上CONV Path
        if self.conv_path:
            self_x = self_x.add(v_conv)

        self_x = self.self_proj(self_x)

        if self.time_deco:
            # 暂时只使用相加融合两次查询
            self_x += self_x_t.reshape(B, T * nH * nW, C)

        # memory ability
        if self.memory:
            if self.cross_att:
                # 将从记忆中查询到的特征与当前特征融合
                # if (len(self.m_k) == 0) and (self.max_len != 1):
                #     # 没有记忆的时候不需要融合
                #     pass
                # else:
                #     # 有记忆的时候需要聚合，并且当最长记忆时长为1时，还会在没有记忆的时候与自己做self-attention增强

                res_x = self.fusion_proj(torch.cat((self_x, cm_x_final), dim=2))     # 这里cm_x_final的形状调整到和默认的x一致
                self_x = self_x + res_x

            # 缓存更新过的记忆张量
            if not self.sub_token_align:
                # 存储没对齐或者token级别对齐的记忆
                try:
                    self.m_k[-1] = cm_k.detach()
                    self.m_v[-1] = cm_v.detach()
                except:
                    # 第一帧的时候记忆张量list为空，需要保证list除了最后一个元素，其他元素都是压缩过的
                    self.m_k.append(cm_k.detach())
                    self.m_v.append(cm_v.detach())
            else:
                # 存储sub-token级别对齐的记忆
                try:
                    self.m_k[-1] = cm_kk_align.detach()
                    self.m_v[-1] = cm_vv_align.detach()
                except:
                    # 第一帧的时候记忆张量list为空，需要保证list除了最后一个元素，其他元素都是压缩过的
                    self.m_k.append(cm_kk_align.detach())
                    self.m_v.append(cm_vv_align.detach())

            # 缓存当前时刻还没被压缩过的记忆张量，会在下一个时刻被压缩
            if not self.store_lf:
                # 局部帧和非局部帧都会被缓存
                self.m_k.append(k_temp.detach())    # debug
                self.m_v.append(v.detach())
            else:
                # 只缓存局部帧
                self.m_k.append(k_lf.detach())
                self.m_v.append(v_lf.detach())

            # 保持记忆力的最大长度
            if len(self.m_k) > self.max_len:
                self.m_k.pop(0)
                self.m_v.pop(0)

            # # 清除缓存的梯度
            # for mem_k, mem_v in zip(self.m_k, self.m_v):
            #     mem_k.requires_grad = False
            #     mem_v.requires_grad = False

        # 恢复形状 B T*H*W C -> B T H W C
        return self_x.reshape(B, T, nH, nW, C)


class Decoupled3DFocalTransformerBlock(nn.Module):
    r""" Decoupled 3D Focal Transformer Block by Hao.
    Args:
        dim (int): Number of input channels. Equal to hidden dim.
        num_heads (int): Number of attention heads.
        window_size (tuple[int]): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        focal_level (int):  The number level of focal window.
        focal_window (int):  Window size of each focal window.
        n_vecs (int): Required for F3N.
        t2t_params (int): T2T parameters for F3N.
    Revised by Hao:
        Add token fusion support and memory ability.
        token_fusion (bool):  Required for Token Fusion Manner.
        memory (bool): Required for memory ability. Using WindowAttentionMem replace the original WindowAttention.
        max_mem_len (int):  Max memory length. Unit: Forward.
        compression_factor (int):  Memory compression factor on channel dimension.
        mem_pool (bool): Whether use pooling to reduce memory spatial size.
        store_lf (bool): If True, only local frames will be cached in the memory. Only work with mem_pool=False.
        align_cache (bool): If True, memory cache will be aligned to current frames before fusion.
                            Only work with mem_pool=False and store_lf=True.
        sub_token_align (bool): If True, memory cache will be aligned at sub-token resolution.
        sub_factor (int): How many groups of sub-token alignment.
        cross_att (bool): Whether use cross attention to align memory and current token.
        time_att (bool): If True, use cross attention to align memory and current token additionally on T dimension.
        time_deco (bool): If True, the Time and Space Cross Att. will be decoupled to reduce cost.
        temp_focal (bool): If True, use temporal focal att to cross att time and space.
        cs_win (bool): If True, use cswin att to cross att time and space.
        mem_att (bool): If True, use cross att to fuse different memory with current feat instead of linear and att.
        cs_focal (bool): If True, use focal mech to upgrade cs win att.
        cs_focal_v2 (bool): If True, upgrade cswin att with same direction sliding window of pooled feat,
                            Only work with cs_focal=True.
        cs_win_strip (int): cs win attention strip width. Default: 1.
        mix_f3n (bool): If True, use MixF3N replace F3N.
        conv_path (bool): If True, add an additional conv path for attention.
        cs_sw (bool): If True, use sliding window logic to upgrade cswin attention.
        pool_strip (bool): I True, use different strip width and pooling to enhance strip window.
                            Only work when cs_win_strip=1.
        pool_sw (int): The strip width that using to pooling and enhance strip window.
                        Only work with pool_strip=True.
    """
    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(5, 9),
                 mlp_ratio=4.,
                 qkv_bias=True,
                 focal_level=2,
                 focal_window=(5, 9),
                 norm_layer=nn.LayerNorm,
                 n_vecs=None,
                 t2t_params=None,
                 token_fusion=False,
                 memory=False,
                 max_mem_len=4,
                 compression_factor=4,
                 mem_pool=False,
                 store_lf=False,
                 align_cache=False,
                 sub_token_align=False,
                 sub_factor=1,
                 cross_att=False,
                 time_att=False,
                 time_deco=False,
                 temp_focal=False,
                 cs_win=False,
                 mem_att=False,
                 cs_focal=False,
                 cs_focal_v2=False,
                 cs_win_strip=1,
                 mix_f3n=False,
                 conv_path=False,
                 cs_sw=False,
                 pool_strip=False,
                 pool_sw=2):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.expand_size = tuple(i // 2 for i in window_size)  # 窗口大小除以2是拓展大小
        self.mlp_ratio = mlp_ratio
        self.focal_level = focal_level
        self.focal_window = focal_window

        self.token_fusion = token_fusion
        self.memory = memory

        self.window_size_glo = self.window_size

        self.norm1 = norm_layer(dim)

        if not self.memory:
            # 使用不带有记忆的attention
            self.attn = Decoupled3DFocalAttentionMem(dim,
                                                     expand_size=self.expand_size,
                                                     window_size=self.window_size,
                                                     focal_window=focal_window,
                                                     focal_level=focal_level,
                                                     num_heads=num_heads,
                                                     qkv_bias=qkv_bias,
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
                                                     cs_win_strip=cs_win_strip,
                                                     conv_path=conv_path,
                                                     cs_sw=cs_sw,
                                                     pool_strip=pool_strip,
                                                     pool_sw=pool_sw)
        else:
            # 使用记忆增强的attention
            self.attn = Decoupled3DFocalAttentionMem(dim,
                                                     expand_size=self.expand_size,
                                                     window_size=self.window_size,
                                                     focal_window=focal_window,
                                                     focal_level=focal_level,
                                                     num_heads=num_heads,
                                                     qkv_bias=qkv_bias,
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
                                                     cs_win_strip=cs_win_strip,
                                                     conv_path=conv_path,
                                                     cs_sw=cs_sw,
                                                     pool_strip=pool_strip,
                                                     pool_sw=pool_sw)

        self.norm2 = norm_layer(dim)

        self.mix_f3n = mix_f3n
        if not mix_f3n:
            self.mlp = FusionFeedForward(dim, n_vecs=n_vecs, t2t_params=t2t_params)
        else:
            self.mlp = MixFusionFeedForward(dim, n_vecs=n_vecs, t2t_params=t2t_params)

    def forward(self, x):

        # if self.memory:
        #     # 记忆力需要额外传入局部帧的时间长度
        l_t = x[2]

        output_size = x[1]
        x = x[0]

        B, T, H, W, C = x.shape

        shortcut = x
        x = self.norm1(x)

        shifted_x = x

        if not self.memory:
            # default
            shifted_x = self.attn(shifted_x, mask_all=None)
        else:
            # memory build in, with l_t as input
            shifted_x = self.attn(shifted_x, mask_all=None, l_t=l_t)

        # FFN
        x = shortcut + shifted_x
        y = self.norm2(x)
        if not self.token_fusion:
            # default manner
            if not self.mix_f3n:
                x = x + self.mlp(y.view(B, T * H * W, C), output_size).view(
                    B, T, H, W, C)
            else:
                # MixF3N需要额外传递H, W
                x = x + self.mlp(y.view(B, T * H * W, C), output_size, T, H, W).view(
                    B, T, H, W, C)
        else:
            x = x + self.mlp(y.view(B, T * H * W, C), (H * 3, W * 3)).view(
                B, T, H, W, C)

        # if self.memory:
        #     # 记忆力需要额外传入局部帧的时间长度
        return x, output_size, l_t
        # else:
        #     # default
        #     return x, output_size
