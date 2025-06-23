import torch
import torch.nn as nn


class Inception_Block_V1(nn.Module):
    """
    Inception block with multiple kernel sizes for multi-scale feature extraction.
    Original implementation from TimesNet paper.
    """
    def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
        super(Inception_Block_V1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        
        # Create multiple conv2d kernels with different sizes
        kernels = []
        for i in range(self.num_kernels):
            # Kernel sizes: 1x1, 3x3, 5x5, 7x7, 9x9, 11x11
            kernel_size = 2 * i + 1
            padding = i  # Same padding to maintain spatial dimensions
            kernels.append(
                nn.Conv2d(in_channels, out_channels, 
                         kernel_size=kernel_size, padding=padding, bias=False)
            )
        self.kernels = nn.ModuleList(kernels)
        
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights for all kernels"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        """
        Apply multiple kernels and aggregate results
        Args:
            x: Input tensor [batch, channels, height, width]
        Returns:
            Aggregated output tensor with same spatial dimensions
        """
        res_list = []
        for i in range(self.num_kernels):
            # Apply each kernel to the input
            conv_result = self.kernels[i](x)
            res_list.append(conv_result)
        
        # Stack and take mean across different kernel results
        # This aggregates multi-scale features
        res = torch.stack(res_list, dim=-1).mean(-1)
        return res