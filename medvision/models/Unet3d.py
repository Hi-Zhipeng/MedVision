import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv3D, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_channels=32):
        super(UNet3D, self).__init__()

        # encoder
        self.enc1 = DoubleConv3D(in_channels, base_channels)
        self.pool1 = nn.MaxPool3d(2)
        self.enc2 = DoubleConv3D(base_channels, base_channels*2)
        self.pool2 = nn.MaxPool3d(2)
        self.enc3 = DoubleConv3D(base_channels*2, base_channels*4)
        self.pool3 = nn.MaxPool3d(2)

        # bottleneck
        self.bottleneck = DoubleConv3D(base_channels*4, base_channels*8)

        # decoder
        self.up3 = nn.ConvTranspose3d(base_channels*8, base_channels*4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv3D(base_channels*8, base_channels*4)
        self.up2 = nn.ConvTranspose3d(base_channels*4, base_channels*2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv3D(base_channels*4, base_channels*2)
        self.up1 = nn.ConvTranspose3d(base_channels*2, base_channels, kernel_size=2, stride=2)
        self.dec1 = DoubleConv3D(base_channels*2, base_channels)

        # 输出层
        self.out_conv = nn.Conv3d(base_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # 编码
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))

        # 中间
        b = self.bottleneck(self.pool3(e3))

        # 解码
        d3 = self.up3(b)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.out_conv(d1)
    