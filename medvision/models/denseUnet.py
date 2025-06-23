import torch
import torch.nn as nn
import torch.nn.functional as F


class SingleLevelDenseNet(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, num_conv=4, dropout=0.0):
        super().__init__()
        self.num_conv = num_conv
        self.filters = out_channels
        self.mid_channels = mid_channels or out_channels
        self.dropout = dropout

        self.convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            for _ in range(num_conv)
        ])
        self.bns = nn.ModuleList([
            nn.BatchNorm2d(out_channels) for _ in range(num_conv)
        ])

        self.pre_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.pre_conv(x)
        outs = [x]
        for i in range(self.num_conv):
            temp = self.convs[i](outs[-1])
            if i > 0:
                temp += sum(outs[:-1])
            temp = self.bns[i](temp)
            temp = F.relu(temp)
            outs.append(temp)
        return outs[-1]


class DownSample(nn.Module):
    def __init__(self, kernel_size=2, stride=2):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size, stride)

    def forward(self, x):
        pooled = self.pool(x)
        return pooled, x  # pooled for next layer, x for skip connection


class UpSampleAndConcat(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, dropout=0.0):
        super().__init__()
        mid_channels = mid_channels or out_channels
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels * 2, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class DenseUNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=3, base_filters=64, num_conv=4, dropout=0.0):
        super().__init__()
        f = base_filters
        self.enc1 = SingleLevelDenseNet(in_channels, f, num_conv=num_conv, dropout=dropout)
        self.down1 = DownSample()

        self.enc2 = SingleLevelDenseNet(f, f, num_conv=num_conv, dropout=dropout)
        self.down2 = DownSample()

        self.enc3 = SingleLevelDenseNet(f, f, num_conv=num_conv, dropout=dropout)
        self.down3 = DownSample()

        self.enc4 = SingleLevelDenseNet(f, f, num_conv=num_conv, dropout=dropout)
        self.down4 = DownSample()

        self.bottom = SingleLevelDenseNet(f, f, num_conv=num_conv, dropout=dropout)

        self.up4 = UpSampleAndConcat(f, f, dropout=dropout)
        self.dec4 = SingleLevelDenseNet(f, f, num_conv=num_conv, dropout=dropout)

        self.up3 = UpSampleAndConcat(f, f, dropout=dropout)
        self.dec3 = SingleLevelDenseNet(f, f, num_conv=num_conv, dropout=dropout)

        self.up2 = UpSampleAndConcat(f, f, dropout=dropout)
        self.dec2 = SingleLevelDenseNet(f, f, num_conv=num_conv, dropout=dropout)

        self.up1 = UpSampleAndConcat(f, f, dropout=dropout)
        self.dec1 = SingleLevelDenseNet(f, f, num_conv=num_conv, dropout=dropout)

        self.out_conv = nn.Conv2d(f, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.enc1(x)
        x, s1 = self.down1(x1)

        x2 = self.enc2(x)
        x, s2 = self.down2(x2)

        x3 = self.enc3(x)
        x, s3 = self.down3(x3)

        x4 = self.enc4(x)
        x, s4 = self.down4(x4)

        x = self.bottom(x)

        x = self.up4(x, s4)
        x = self.dec4(x)

        x = self.up3(x, s3)
        x = self.dec3(x)

        x = self.up2(x, s2)
        x = self.dec2(x)

        x = self.up1(x, s1)
        x = self.dec1(x)

        return self.out_conv(x)