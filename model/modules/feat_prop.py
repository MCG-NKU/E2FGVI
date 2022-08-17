"""
    BasicVSR++: Improving Video Super-Resolution with Enhanced Propagation and Alignment, CVPR 2022
"""
import torch
import torch.nn as nn

from mmcv.ops import ModulatedDeformConv2d, modulated_deform_conv2d
from mmcv.cnn import constant_init

from model.modules.flow_comp import flow_warp


class SecondOrderDeformableAlignment(ModulatedDeformConv2d):
    """Second-order deformable alignment module."""
    def __init__(self, *args, **kwargs):
        self.max_residue_magnitude = kwargs.pop('max_residue_magnitude', 10)

        super(SecondOrderDeformableAlignment, self).__init__(*args, **kwargs)

        self.conv_offset = nn.Sequential(
            nn.Conv2d(3 * self.out_channels + 4, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, 27 * self.deform_groups, 3, 1, 1),
        )

        self.init_offset()

    def init_offset(self):
        constant_init(self.conv_offset[-1], val=0, bias=0)

    def forward(self, x, extra_feat, flow_1, flow_2):
        extra_feat = torch.cat([extra_feat, flow_1, flow_2], dim=1)
        out = self.conv_offset(extra_feat)
        o1, o2, mask = torch.chunk(out, 3, dim=1)

        # offset
        offset = self.max_residue_magnitude * torch.tanh(
            torch.cat((o1, o2), dim=1))
        offset_1, offset_2 = torch.chunk(offset, 2, dim=1)
        offset_1 = offset_1 + flow_1.flip(1).repeat(1,
                                                    offset_1.size(1) // 2, 1,
                                                    1)
        offset_2 = offset_2 + flow_2.flip(1).repeat(1,
                                                    offset_2.size(1) // 2, 1,
                                                    1)
        offset = torch.cat([offset_1, offset_2], dim=1)

        # mask
        mask = torch.sigmoid(mask)

        # 默认使用的2阶对齐方法是dcn-v2，也就是调制可变形卷积，所谓的调制就是新增了一个和图像等大的mask，范围在0-1
        return modulated_deform_conv2d(x, offset, mask, self.weight, self.bias,
                                       self.stride, self.padding,
                                       self.dilation, self.groups,
                                       self.deform_groups)


class BidirectionalPropagation(nn.Module):
    def __init__(self, channel, flow_align=False):
        super(BidirectionalPropagation, self).__init__()
        modules = ['backward_', 'forward_']
        self.deform_align = nn.ModuleDict()
        self.backbone = nn.ModuleDict()
        self.channel = channel

        # 使用正确的对齐方式(True), 使用默认对齐方式(False)
        self.flow_align = flow_align

        for i, module in enumerate(modules):
            self.deform_align[module] = SecondOrderDeformableAlignment(
                2 * channel, channel, 3, padding=1, deform_groups=16)

            self.backbone[module] = nn.Sequential(
                nn.Conv2d((2 + i) * channel, channel, 3, 1, 1),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                nn.Conv2d(channel, channel, 3, 1, 1),
            )

        self.fusion = nn.Conv2d(2 * channel, channel, 1, 1, 0)

    def forward(self, x, flows_backward, flows_forward):
        """
        x shape : [b, t, c, h, w]
        return [b, t, c, h, w]
        """
        b, t, c, h, w = x.shape
        feats = {}
        feats['spatial'] = [x[:, i, :, :, :] for i in range(0, t)]

        for module_name in ['backward_', 'forward_']:

            feats[module_name] = []

            frame_idx = range(0, t)
            flow_idx = range(-1, t - 1)
            mapping_idx = list(range(0, len(feats['spatial'])))
            mapping_idx += mapping_idx[::-1]

            if 'backward' in module_name:
                frame_idx = frame_idx[::-1]
                flows = flows_backward
            else:
                flows = flows_forward

            feat_prop = x.new_zeros(b, self.channel, h, w)

            if not self.flow_align:
                # default backward时存在i和idx不对应的bug
                for i, idx in enumerate(frame_idx):
                    feat_current = feats['spatial'][mapping_idx[idx]]

                    if i > 0:
                        flow_n1 = flows[:, flow_idx[i], :, :, :]
                        # cond是使用warp后的此前特征，n1表示光流第一次warp
                        # 在backward里，作者想把 t5 warp 到 t4，但是用了 t1 到 t0 的光流，用错了
                        cond_n1 = flow_warp(feat_prop, flow_n1.permute(0, 2, 3, 1))

                        # initialize second-order features
                        feat_n2 = torch.zeros_like(feat_prop)
                        flow_n2 = torch.zeros_like(flow_n1)
                        cond_n2 = torch.zeros_like(cond_n1)
                        if i > 1:
                            feat_n2 = feats[module_name][-2]
                            flow_n2 = flows[:, flow_idx[i - 1], :, :, :]
                            flow_n2 = flow_n1 + flow_warp(
                                flow_n2, flow_n1.permute(0, 2, 3, 1))
                            cond_n2 = flow_warp(feat_n2,
                                                flow_n2.permute(0, 2, 3, 1))

                        cond = torch.cat([cond_n1, feat_current, cond_n2], dim=1)
                        feat_prop = torch.cat([feat_prop, feat_n2], dim=1)
                        feat_prop = self.deform_align[module_name](feat_prop, cond,
                                                                   flow_n1,
                                                                   flow_n2)

                    feat = [feat_current] + [
                        feats[k][idx]
                        for k in feats if k not in ['spatial', module_name]
                    ] + [feat_prop]

                    feat = torch.cat(feat, dim=1)
                    feat_prop = feat_prop + self.backbone[module_name](feat)
                    feats[module_name].append(feat_prop)
                #################################################################

            else:
                # 修正backward时存在i和idx不对应的bug
                for i, idx in enumerate(frame_idx):
                    feat_current = feats['spatial'][mapping_idx[idx]]

                    if i > 0:
                        if 'backward' in module_name:
                            flow_n1 = flows[:, flow_idx[idx]+1, :, :, :]
                            # print('Backward, using back flow: (%f) for warp to frame (%f)'
                            #       % (flow_idx[idx]+1, mapping_idx[idx]))
                        else:
                            flow_n1 = flows[:, flow_idx[i], :, :, :]
                            # print('Forward, using forward flow: (%f) for warp to frame (%f)'
                            #       % (flow_idx[i], mapping_idx[idx]))
                        cond_n1 = flow_warp(feat_prop, flow_n1.permute(0, 2, 3, 1))

                        # initialize second-order features
                        feat_n2 = torch.zeros_like(feat_prop)
                        flow_n2 = torch.zeros_like(flow_n1)
                        cond_n2 = torch.zeros_like(cond_n1)
                        if i > 1:
                            feat_n2 = feats[module_name][-2]
                            if 'backward' in module_name:
                                flow_n2 = flows[:, flow_idx[idx]+2, :, :, :]
                                # print('Backward, using second back flow: (%f) for warp to frame (%f)'
                                #       % (flow_idx[idx] + 2, mapping_idx[idx]))
                            else:
                                flow_n2 = flows[:, flow_idx[i - 1], :, :, :]
                                # print('Forward, using second forward flow: (%f) for warp to frame (%f)'
                                #       % (flow_idx[i - 1], mapping_idx[idx]))
                            flow_n2 = flow_n1 + flow_warp(
                                flow_n2, flow_n1.permute(0, 2, 3, 1))
                            cond_n2 = flow_warp(feat_n2,
                                                flow_n2.permute(0, 2, 3, 1))

                        cond = torch.cat([cond_n1, feat_current, cond_n2], dim=1)
                        feat_prop = torch.cat([feat_prop, feat_n2], dim=1)
                        feat_prop = self.deform_align[module_name](feat_prop, cond,
                                                                   flow_n1,
                                                                   flow_n2)

                    feat = [feat_current] + [
                        feats[k][idx]
                        for k in feats if k not in ['spatial', module_name]
                    ] + [feat_prop]

                    feat = torch.cat(feat, dim=1)
                    feat_prop = feat_prop + self.backbone[module_name](feat)
                    feats[module_name].append(feat_prop)
                ##################################################################

            if 'backward' in module_name:
                feats[module_name] = feats[module_name][::-1]

            # print('#' * 20)

        outputs = []
        for i in range(0, t):
            align_feats = [feats[k].pop(0) for k in feats if k != 'spatial']
            align_feats = torch.cat(align_feats, dim=1)
            outputs.append(self.fusion(align_feats))

        return torch.stack(outputs, dim=1) + x
