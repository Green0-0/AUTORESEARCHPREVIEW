"""The two visible fixed classifier architectures and their fixed training recipes.

Each arch module/entry exposes NAME, RECIPE (dict) and build_model(num_classes).
A third, withheld architecture with the same interface is loaded at eval time
from $AR_ROOT/private/t16_synthetic_data/withheld_arch.py (see evaluate.py).
"""
import torch.nn as nn


# ---------------------------------------------------------------- ConvNet-3
# The standard dataset-distillation evaluation net: 3 x [conv3x3-128, instance norm, ReLU, avgpool2].
CONVNET_RECIPE = dict(steps=1500, batch=512, lr=0.05, momentum=0.9, wd=5e-4,
                      warmup_frac=0.1, label_smoothing=0.0, opt="sgd")


def build_convnet(num_classes=100, width=128, depth=3):
    layers, c = [], 3
    for _ in range(depth):
        layers += [nn.Conv2d(c, width, 3, padding=1),
                   nn.GroupNorm(width, width, affine=True),   # = instance norm
                   nn.ReLU(inplace=True), nn.AvgPool2d(2)]
        c = width
    layers += [nn.Flatten(), nn.Linear(width * 4 * 4, num_classes)]
    return nn.Sequential(*layers)


# ---------------------------------------------------------------- ResNet-ish (ResNet-9 style, narrow)
RESNET_RECIPE = dict(steps=1500, batch=512, lr=0.1, momentum=0.9, wd=5e-4,
                     warmup_frac=0.1, label_smoothing=0.0, opt="sgd")


def _cbr(ci, co):
    return nn.Sequential(nn.Conv2d(ci, co, 3, padding=1, bias=False), nn.BatchNorm2d(co), nn.ReLU(inplace=True))


class _Res(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.f = nn.Sequential(_cbr(c, c), _cbr(c, c))

    def forward(self, x):
        return x + self.f(x)


def build_resnet(num_classes=100, w=(64, 128, 256)):
    return nn.Sequential(
        _cbr(3, w[0]),
        _cbr(w[0], w[1]), nn.MaxPool2d(2), _Res(w[1]),
        _cbr(w[1], w[2]), nn.MaxPool2d(2),
        _cbr(w[2], w[2]), nn.MaxPool2d(2), _Res(w[2]),
        nn.AdaptiveMaxPool2d(1), nn.Flatten(), nn.Linear(w[2], num_classes),
    )


VISIBLE = {
    "convnet": (CONVNET_RECIPE, build_convnet),
    "resnet": (RESNET_RECIPE, build_resnet),
}
