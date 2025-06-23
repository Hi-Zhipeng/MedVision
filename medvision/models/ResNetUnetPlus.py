import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.double_conv = DoubleConv(in_channels, out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.down = nn.MaxPool2d(2)

    def forward(self, x):
        identity = self.downsample(x)
        out = self.double_conv(x)
        out = self.relu(out + identity)
        return self.down(out), out  # 返回下采样和跳跃连接


class VGGBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class ResNetUnetPlus(nn.Module):
    def __init__(self, in_channels=4, num_classes=3, deep_supervision=False):
        super().__init__()
        self.deep_supervision = deep_supervision
        filters = [32, 64, 128, 256, 512]

        # Encoder
        self.resblock1 = ResBlock(in_channels, filters[0])
        self.resblock2 = ResBlock(filters[0], filters[1])
        self.resblock3 = ResBlock(filters[1], filters[2])
        self.resblock4 = ResBlock(filters[2], filters[3])
        self.resblock5 = ResBlock(filters[3], filters[4])

        # Decoder with nested skip connections
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_1 = VGGBlock(filters[0]+filters[1], filters[0], filters[0])
        self.conv1_1 = VGGBlock(filters[1]+filters[2], filters[1], filters[1])
        self.conv2_1 = VGGBlock(filters[2]+filters[3], filters[2], filters[2])
        self.conv3_1 = VGGBlock(filters[3]+filters[4], filters[3], filters[3])

        self.conv0_2 = VGGBlock(filters[0]*2+filters[1], filters[0], filters[0])
        self.conv1_2 = VGGBlock(filters[1]*2+filters[2], filters[1], filters[1])
        self.conv2_2 = VGGBlock(filters[2]*2+filters[3], filters[2], filters[2])

        self.conv0_3 = VGGBlock(filters[0]*3+filters[1], filters[0], filters[0])
        self.conv1_3 = VGGBlock(filters[1]*3+filters[2], filters[1], filters[1])

        self.conv0_4 = VGGBlock(filters[0]*4+filters[1], filters[0], filters[0])

        # Final layers
        if self.deep_supervision:
            self.final1 = nn.Conv2d(filters[0], num_classes, 1)
            self.final2 = nn.Conv2d(filters[0], num_classes, 1)
            self.final3 = nn.Conv2d(filters[0], num_classes, 1)
            self.final4 = nn.Conv2d(filters[0], num_classes, 1)
        else:
            self.final = nn.Conv2d(filters[0], num_classes, 1)

    def forward(self, x):
        x, x0_0 = self.resblock1(x)
        x, x1_0 = self.resblock2(x)
        x, x2_0 = self.resblock3(x)
        x, x3_0 = self.resblock4(x)
        _, x4_0 = self.resblock5(x)

        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], dim=1))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], dim=1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], dim=1))

        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], dim=1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], dim=1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], dim=1))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], dim=1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], dim=1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], dim=1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], dim=1))

        if self.deep_supervision:
            out1 = self.final1(x0_1)
            out2 = self.final2(x0_2)
            out3 = self.final3(x0_3)
            out4 = self.final4(x0_4)
            return [out1, out2, out3, out4]
        else:
            return self.final(x0_4)