"""
UNet model implementation for medical image segmentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConvBlock(nn.Module):
    """Double convolution block for UNet."""

    def __init__(self, in_channels, out_channels, mid_channels=None, dropout=0.0):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels
            
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity()
        )

    def forward(self, x):
        return self.double_conv(x)


class DownBlock(nn.Module):
    """Downsampling block for UNet."""

    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConvBlock(in_channels, out_channels, dropout=dropout)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class UpBlock(nn.Module):
    """Upsampling block for UNet."""

    def __init__(self, in_channels, out_channels, dropout=0.0, bilinear=True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConvBlock(in_channels + out_channels, out_channels, dropout=dropout)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConvBlock(in_channels // 2 + out_channels, out_channels, dropout=dropout)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                          diff_y // 2, diff_y - diff_y // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Output convolution for UNet."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net architecture for medical image segmentation.
    
    Reference: https://arxiv.org/abs/1505.04597
    """

    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512], dropout=0.0, bilinear=True):
        """
        Initialize the UNet model.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            features: List of feature dimensions for each level
            dropout: Dropout probability
            bilinear: Whether to use bilinear upsampling
        """
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear
        

        self.inc = DoubleConvBlock(in_channels, features[0], dropout=dropout)
        

        self.down_blocks = nn.ModuleList()
        for i in range(len(features) - 1):
            self.down_blocks.append(DownBlock(features[i], features[i+1], dropout=dropout))
        

        self.up_blocks = nn.ModuleList()
        for i in range(len(features) - 1, 0, -1):

            self.up_blocks.append(UpBlock(features[i], features[i-1], dropout=dropout, bilinear=bilinear))
        

        self.outc = OutConv(features[0], out_channels)

    def forward(self, x):

        x_features = [self.inc(x)]
        
        for i in range(len(self.down_blocks)):
            x_features.append(self.down_blocks[i](x_features[-1]))

        x_out = x_features[-1]
        for i in range(len(self.up_blocks)):
            x_out = self.up_blocks[i](x_out, x_features[-(i+2)])

        logits = self.outc(x_out)
        
        return logits
