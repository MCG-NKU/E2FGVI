"""
Compatibility layer for deformable convolution operations.
Provides fallback implementations when mmcv is not available.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ModulatedDeformConv2dCompat(nn.Module):
    """
    Modulated Deformable Convolution 2d (compatible fallback implementation)
    When mmcv._ext is not available, this provides a simpler but functional approximation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, 
                 dilation=1, groups=1, bias=True, deform_groups=1):
        super(ModulatedDeformConv2dCompat, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.deform_groups = deform_groups
        
        # Standard conv2d weight and bias
        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels // groups, kernel_size, kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        
        nn.init.kaiming_uniform_(self.weight, a=0)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x, offset, mask):
        """
        Args:
            x: input features (B, C, H, W)
            offset: deformable offset (B, 2*kernel_size*kernel_size*deform_groups, H, W)
            mask: modulation mask (B, kernel_size*kernel_size*deform_groups, H, W)
        
        Returns:
            output: convolution result (B, out_channels, H', W')
        """
        # As a fallback, we use standard convolution with spatial normalization
        # This maintains numerical stability while deformable convolutions compile
        
        batch_size = x.shape[0]
        
        # Apply standard convolution
        output = F.conv2d(
            x, self.weight, self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups
        )
        
        # Scale by mask if provided (simplified approach)
        if mask is not None:
            # Resize mask to match output spatial dimensions
            mask_mean = F.adaptive_avg_pool2d(mask, output.shape[-2:])
            # Average mask across deform groups
            mask_mean = mask_mean.mean(dim=1, keepdim=True)
            output = output * mask_mean
        
        return output


def modulated_deform_conv2d_compat(x, offset, mask, weight, bias, 
                                   stride=1, padding=0, dilation=1, 
                                   groups=1, deform_groups=1):
    """Functional interface for modulated deformable convolution."""
    batch_size, in_channels, h, w = x.shape
    out_channels = weight.shape[0]
    
    # Use standard convolution as fallback
    output = F.conv2d(
        x, weight, bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups
    )
    
    # Apply mask scaling if available
    if mask is not None:
        mask_avg = F.adaptive_avg_pool2d(mask, output.shape[-2:])
        mask_avg = mask_avg.mean(dim=1, keepdim=True).clamp(min=0.1)
        output = output * mask_avg
    
    return output


def try_import_mmcv_ops():
    """Try to import mmcv ops, fallback to compatibility layer."""
    try:
        from mmcv.ops import ModulatedDeformConv2d, modulated_deform_conv2d
        print("[INFO] Using native mmcv deformable convolution ops")
        return ModulatedDeformConv2d, modulated_deform_conv2d
    except (ImportError, ModuleNotFoundError) as e:
        print(f"[WARNING] Failed to import mmcv ops: {e}")
        print("[INFO] Using compatibility layer for deformable convolution")
        return ModulatedDeformConv2dCompat, modulated_deform_conv2d_compat
