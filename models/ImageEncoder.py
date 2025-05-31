import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F


class ImageEncoder(nn.Module):
    """
    Takes in N number of images from an individual camera. Expected input shape: (B, seq_len, C, H, W), output shape: (B, seq_len, out_dim)
    """

    def __init__(self, seq_len, ch_in=3, out_dim=1280):
        super(ImageEncoder, self).__init__()
        self.mobile_net = MobileNetV2(seq_len, ch_in, out_dim)
        self.seq_len = seq_len

    def forward(self, x):
        y = self.mobile_net(x)
        return y


"""
Code below is adapted from a MobileNet V2 implementation on GitHub at https://github.com/jmjeon2/MobileNet-Pytorch.
"""


def dwise_conv(ch_in, stride=1):
    return nn.Sequential(
        # depthwise
        nn.Conv2d(
            ch_in,
            ch_in,
            kernel_size=3,
            padding=1,
            stride=stride,
            groups=ch_in,
            bias=False,
        ),
        nn.BatchNorm2d(ch_in),
        nn.ReLU6(inplace=True),
    )


def conv1x1(ch_in, ch_out):
    return nn.Sequential(
        nn.Conv2d(ch_in, ch_out, kernel_size=1, padding=0, stride=1, bias=False),
        nn.BatchNorm2d(ch_out),
        nn.ReLU6(inplace=True),
    )


def conv3x3(ch_in, ch_out, stride):
    return nn.Sequential(
        nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1, stride=stride, bias=False),
        nn.BatchNorm2d(ch_out),
        nn.ReLU6(inplace=True),
    )


class InvertedBlock(nn.Module):
    def __init__(self, ch_in, ch_out, expand_ratio, stride):
        super(InvertedBlock, self).__init__()

        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = ch_in * expand_ratio

        self.use_res_connect = self.stride == 1 and ch_in == ch_out

        layers = []
        if expand_ratio != 1:
            layers.append(conv1x1(ch_in, hidden_dim))
        layers.extend(
            [
                # dw
                dwise_conv(hidden_dim, stride=stride),
                # pw
                conv1x1(hidden_dim, ch_out),
            ]
        )

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.layers(x)
        else:
            return self.layers(x)


class MobileNetV2(nn.Module):
    def __init__(self, seq_len, ch_in=3, out_dim=1280):
        super(MobileNetV2, self).__init__()
        self.seq_len = seq_len
        self.out_dim = out_dim

        self.configs = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        self.stem_conv = conv3x3(ch_in, 32, stride=2)

        layers = []
        input_channel = 32
        for t, c, n, s in self.configs:
            for i in range(n):
                stride = s if i == 0 else 1
                layers.append(
                    InvertedBlock(
                        ch_in=input_channel, ch_out=c, expand_ratio=t, stride=stride
                    )
                )
                input_channel = c

        self.layers = nn.Sequential(*layers)

        self.last_conv = conv1x1(input_channel, out_dim)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        """
        This encoder assumes images come in in (B, seq_len, C, H, W), and outputs in shape (B, seq_len, out_dim)
        """
        # print(x.shape)
        B, seq_len, C, H, W = x.shape
        assert (
            seq_len == self.seq_len
        ), "Passed in incorrect seq_len during initialization"

        x_flat = x.view(-1, C, H, W)  # → (B*seq_len, C, H, W)

        y = self.stem_conv(x_flat)
        y = self.layers(y)
        y = self.last_conv(y)  # (B*seq_len, out_dim, H, W)
        y = self.avg_pool(y).view(
            -1, self.out_dim
        )  # (B*seq_len, out_dim, 1, 1) --[view]-> (B*seq_len, out_dim)
        return y.view(B, self.seq_len, self.out_dim)
