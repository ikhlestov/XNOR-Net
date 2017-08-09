import torch.nn as nn


class UsualConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, avg_kernel_size=2, avg_stride=2, bn_size=0):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activ = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activ(x)
        x = self.max_pool(x)
        return x


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            UsualConv2d(3, 12, kernel_size=3, padding=1, stride=1),
            UsualConv2d(12, 24, kernel_size=3, padding=1, stride=1),
            UsualConv2d(24, 32, kernel_size=3, padding=1, stride=1),
            UsualConv2d(32, 48, kernel_size=3, padding=1, stride=1),
            UsualConv2d(48, 64, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 10, kernel_size=1)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size()[0], -1)
        return x
