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
        Add token fusion support.
        token_fusion (bool):  Required for Token Fusion Manner.
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
                 token_fusion=False):
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

        self.attn = WindowAttention(dim,
                                    expand_size=self.expand_size,
                                    window_size=self.window_size,
                                    focal_window=focal_window,
                                    focal_level=focal_level,
                                    num_heads=num_heads,
                                    qkv_bias=qkv_bias,
                                    pool_method=pool_method)

        self.norm2 = norm_layer(dim)
        self.mlp = FusionFeedForward(dim, n_vecs=n_vecs, t2t_params=t2t_params)

    def forward(self, x):
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
        attn_windows = self.attn(x_windows_all, mask_all=x_window_masks_all)

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

        return x, output_size
