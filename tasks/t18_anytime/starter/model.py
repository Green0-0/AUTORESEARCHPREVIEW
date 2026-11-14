"""Starter anytime model: a small ResNet-9-style CNN with 3 early-exit heads.
Each stage reuses the previous stage's features, so later predictions cost only the extra layers."""
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

MEAN = (0.5071, 0.4865, 0.4409)
STD = (0.2673, 0.2564, 0.2762)


def cbr(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class Res(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.a, self.b = cbr(c, c), cbr(c, c)

    def forward(self, x):
        return x + self.b(self.a(x))


class Head(nn.Module):
    def __init__(self, c, n=100):
        super().__init__()
        self.fc = nn.Linear(c, n)

    def forward(self, x):
        return self.fc(F.adaptive_max_pool2d(x, 1).flatten(1))


class AnytimeNet(nn.Module):
    def __init__(self, w=(32, 64, 128, 256), num_classes=100):
        super().__init__()
        self.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(STD).view(1, 3, 1, 1))
        # stage 1 @32 -> 16 -> 8
        self.s1 = nn.Sequential(cbr(3, w[0]), nn.MaxPool2d(2), cbr(w[0], w[1]), nn.MaxPool2d(2))
        # stage 2 @8
        self.s2 = nn.Sequential(cbr(w[1], w[2]), Res(w[2]))
        # stage 3 @8 -> 4
        self.s3 = nn.Sequential(cbr(w[2], w[3]), Res(w[3]), nn.MaxPool2d(2))
        self.heads = nn.ModuleList([Head(w[1], num_classes), Head(w[2], num_classes), Head(w[3], num_classes)])

    def stages(self):
        return [self.s1, self.s2, self.s3]

    def forward_all(self, x):
        """All exits' logits (training). x: normalized float [B,3,32,32]."""
        outs = []
        for s, h in zip(self.stages(), self.heads):
            x = s(x)
            outs.append(h(x))
        return outs

    def anytime(self, x):
        """Anytime interface: x float [B,3,32,32] in [0,1]; yields logits [B,100], cheapest first.
        Lazy: stage k+1 is only computed when the caller asks for the next prediction."""
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=x.is_cuda):
            h = ((x - self.mean) / self.std).contiguous(memory_format=torch.channels_last)
            for s, head in zip(self.stages(), self.heads):
                h = s(h)
                yield head(h).float()


def load_model(artifact_dir, device):
    cfg = json.loads((Path(artifact_dir) / "config.json").read_text())
    m = AnytimeNet(w=tuple(cfg["widths"]))
    m.load_state_dict(torch.load(Path(artifact_dir) / "model.pt", map_location="cpu"))
    return m.to(device).to(memory_format=torch.channels_last).eval()
