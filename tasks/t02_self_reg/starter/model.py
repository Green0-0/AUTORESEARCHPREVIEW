"""Starter model: ResNet-9-style CNN with GroupNorm (per-example statistics, so train and eval behave
identically). No dropout, no BatchNorm, no augmentation. Input: float32 NCHW in [0,1]. Output: logits (N,100).
Normalization is part of the model."""
import torch
import torch.nn as nn

MEAN = (0.5071, 0.4865, 0.4409)
STD = (0.2673, 0.2564, 0.2762)
WIDTH = 128


def conv_gn(cin, cout, pool=False):
    layers = [nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.GroupNorm(cout // 16, cout), nn.ReLU(inplace=True)]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class Residual(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.f = nn.Sequential(conv_gn(c, c), conv_gn(c, c))

    def forward(self, x):
        return x + self.f(x)


class Net(nn.Module):
    def __init__(self, num_classes=100, w=WIDTH):
        super().__init__()
        self.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(STD).view(1, 3, 1, 1))
        self.net = nn.Sequential(
            conv_gn(3, w),
            conv_gn(w, 2 * w, pool=True), Residual(2 * w),
            conv_gn(2 * w, 4 * w, pool=True),
            conv_gn(4 * w, 8 * w, pool=True), Residual(8 * w),
            nn.AdaptiveMaxPool2d(1), nn.Flatten(),
            nn.Linear(8 * w, num_classes, bias=False),
        )
        self.scale = 0.125  # fixed logit scale

    def forward(self, x):
        x = ((x - self.mean) / self.std).contiguous(memory_format=torch.channels_last)
        return self.net(x) * self.scale


def build_model():
    return Net()
