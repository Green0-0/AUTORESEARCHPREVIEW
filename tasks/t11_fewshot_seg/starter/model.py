"""Starter model: the provided ImageNet ResNet-50 (stride 32) + a 1x1-conv linear head. The harness upsamples the
(B,21,10,10) logits bilinearly to 320x320."""
import torch
import torch.nn as nn

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


class LinearSeg(nn.Module):
    def __init__(self, make_backbone, num_classes=21):
        super().__init__()
        self.backbone = make_backbone()
        self.head = nn.Conv2d(2048, num_classes, 1)
        self.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(STD).view(1, 3, 1, 1))

    def features(self, x):
        b = self.backbone
        x = (x - self.mean) / self.std
        x = b.maxpool(b.relu(b.bn1(b.conv1(x))))
        return b.layer4(b.layer3(b.layer2(b.layer1(x))))

    def forward(self, x):          # x: float (B,3,H,W) RGB in [0,1]
        return self.head(self.features(x))


def build_model(make_backbone):
    return LinearSeg(make_backbone)
