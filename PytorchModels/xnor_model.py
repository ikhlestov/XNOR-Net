import torch
import torch.nn as nn

from .base_model import UsualConv2d


class _BinActive(torch.autograd.Function):

    def forward(self, input):
        self.save_for_backward(input)
        output = torch.sign(input)
        return output

    def backward(self, grad_output):
        # saved tensors - tuple of tensors with one element
        input, = self.saved_tensors
        grad_output[input.ge(1)] = 0
        grad_output[input.le(-1)] = 0
        return grad_output


class BinActive(nn.Module):
    def forward(self, x):
        return _BinActive()(x)


class BinConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, avg_kernel_size=2, avg_stride=2):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding)
        self.bn = nn.BatchNorm2d(in_channels)
        self.activ = BinActive()
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.bn(x)
        x = self.activ(x)
        x = self.conv(x)
        x = self.max_pool(x)
        return x


class XNORModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            BinConv2D(3, 12, kernel_size=3, padding=1, stride=1),
            BinConv2D(12, 24, kernel_size=3, padding=1, stride=1),
            BinConv2D(24, 32, kernel_size=3, padding=1, stride=1),
            BinConv2D(32, 48, kernel_size=3, padding=1, stride=1),
            BinConv2D(48, 64, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 10, kernel_size=1)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size()[0], -1)
        return x
