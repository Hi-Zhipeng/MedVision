import torch
import torch.nn as nn
import torch.nn.functional as F


class VGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_convs=2):
        super().__init__()
        layers = []
        for i in range(num_convs):
            layers.append(
                nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=3, padding=1)
            )
            layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class FCN32s(nn.Module):
    def __init__(self, in_channels=3, num_classes=21):
        super().__init__()

        # VGG encoder blocks
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=100),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/2

        self.conv2 = VGGBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/4

        self.conv3 = VGGBlock(128, 256, num_convs=3)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/8

        self.conv4 = VGGBlock(256, 512, num_convs=3)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/16

        self.conv5 = VGGBlock(512, 512, num_convs=3)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/32

        # Fully connected layers
        self.fc6 = nn.Conv2d(512, 4096, kernel_size=7)
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout2d()

        self.fc7 = nn.Conv2d(4096, 4096, kernel_size=1)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout2d()

        # Score layer
        self.score_fr = nn.Conv2d(4096, num_classes, kernel_size=1)

        # Upsampling
        self.upscore = nn.ConvTranspose2d(
            num_classes, num_classes, kernel_size=64, stride=32, padding=16, bias=False
        )

    def forward(self, x):
        input_size = x.size()

        x = self.conv1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.pool3(x)

        x = self.conv4(x)
        x = self.pool4(x)

        x = self.conv5(x)
        x = self.pool5(x)

        x = self.relu6(self.fc6(x))
        x = self.drop6(x)

        x = self.relu7(self.fc7(x))
        x = self.drop7(x)

        x = self.score_fr(x)
        x = self.upscore(x)

        # Crop to match input size (padding correction)
        h_crop = x.size(2) - input_size[2]
        w_crop = x.size(3) - input_size[3]
        x = x[:, :, h_crop // 2: h_crop // 2 + input_size[2],
                 w_crop // 2: w_crop // 2 + input_size[3]]

        return x