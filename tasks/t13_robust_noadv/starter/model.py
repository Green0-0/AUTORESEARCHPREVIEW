"""Starter model: ResNet-9 (DAWNBench style). Input: float32 NCHW in [0,1]. Output: logits (N,10).
Normalization is part of the model, as the harness contract requires."""
import torch
import torch.nn as nn

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


def conv_bn(cin, cout, pool=False):
    layers = [nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True)]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class Residual(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.f = nn.Sequential(conv_bn(c, c), conv_bn(c, c))

    def forward(self, x):
        return x + self.f(x)


class ResNet9(nn.Module):
    def __init__(self, num_classes=10, w=64):
        super().__init__()
        self.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(STD).view(1, 3, 1, 1))
        self.net = nn.Sequential(
            conv_bn(3, w),
            conv_bn(w, 2 * w, pool=True), Residual(2 * w),
            conv_bn(2 * w, 4 * w, pool=True),
            conv_bn(4 * w, 8 * w, pool=True), Residual(8 * w),
            nn.AdaptiveMaxPool2d(1), nn.Flatten(),
            nn.Linear(8 * w, num_classes, bias=False),
        )
        self.scale = 0.125  # fixed logit scale (standard ResNet-9 trick)

    def forward(self, x):
        return self.net((x - self.mean) / self.std) * self.scale


def build_model():
    return ResNet9()
