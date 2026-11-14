"""Fixed parts of t19_data_ordering: data, model, augmentation, mode settings.

Nothing in this file may be changed by a solution.
"""
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

AR_ROOT = Path(os.environ.get("AR_ROOT", "/usr/xtmp/kg349/ar"))
AR_DATA = Path(os.environ.get("AR_DATA", AR_ROOT / "data"))
TASK_ID = "t19_data_ordering"
NUM_CLASSES = 100
MEAN = (0.5071, 0.4865, 0.4409)
STD = (0.2673, 0.2564, 0.2762)

# Fixed training recipe per mode. `budget_s` is the wall budget of the whole
# training process (python start -> model saved), scheduling included.
MODES = {
    "full":      dict(batch_size=512, total_steps=6000, budget_s=900.0, n_train=None, n_test=None, device="cuda"),
    "smoke":     dict(batch_size=512, total_steps=300,  budget_s=110.0, n_train=None, n_test=None, device="cuda"),
    "cpu-smoke": dict(batch_size=64,  total_steps=12,   budget_s=170.0, n_train=2048, n_test=512,  device="cpu"),
}
OPT = dict(lr=0.4, momentum=0.9, nesterov=True, weight_decay=5e-4, label_smoothing=0.2, warmup_frac=0.15)


def seed_value() -> int:
    return int(os.environ.get("AR_SEED", "0"))


def load_train(n=None):
    d = AR_DATA / TASK_ID
    x = np.load(d / "train_x.npy")
    y = np.load(d / "train_y.npy")
    if n is not None:
        x, y = x[:n], y[:n]
    return x, y


def load_test(n=None):
    d = AR_ROOT / "private" / TASK_ID
    x = np.load(d / "test_x.npy")
    y = np.load(d / "test_y.npy")
    if n is not None:
        x, y = x[:n], y[:n]
    return x, y


# ---------------------------------------------------------------- model (ResNet-9 style, 2x width, 26.4M params)
def conv_bn(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class Residual(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.a, self.b = conv_bn(c, c), conv_bn(c, c)

    def forward(self, x):
        return x + self.b(self.a(x))


class Net(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, w=(128, 256, 512, 1024)):
        super().__init__()
        self.body = nn.Sequential(
            conv_bn(3, w[0]),
            conv_bn(w[0], w[1]), nn.MaxPool2d(2), Residual(w[1]),
            conv_bn(w[1], w[2]), nn.MaxPool2d(2),
            conv_bn(w[2], w[3]), nn.MaxPool2d(2), Residual(w[3]),
            nn.AdaptiveMaxPool2d(1), nn.Flatten(),
        )
        self.fc = nn.Linear(w[3], num_classes, bias=False)

    def forward(self, x):
        return self.fc(self.body(x)) * 0.125


def build_model(device):
    torch.manual_seed(seed_value())  # fixed init distribution and seed
    m = Net().to(device)
    return m.to(memory_format=torch.channels_last)


# ---------------------------------------------------------------- GPU-side preprocessing / augmentation
def normalize(x_uint8):
    x = x_uint8.float().div_(255.0)
    mean = torch.tensor(MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


def augment(x_uint8, gen):
    """Fixed augmentation: random horizontal flip + random crop (reflect pad 4)."""
    x = normalize(x_uint8)
    B, dev = x.shape[0], x.device
    flip = torch.rand(B, generator=gen, device=dev) < 0.5
    x = torch.where(flip.view(B, 1, 1, 1), x.flip(3), x)
    xp = F.pad(x, (4, 4, 4, 4), mode="reflect")
    dy = torch.randint(0, 9, (B,), generator=gen, device=dev)
    dx = torch.randint(0, 9, (B,), generator=gen, device=dev)
    ar = torch.arange(32, device=dev)
    rows = (dy[:, None] + ar[None]).view(B, 1, 32, 1).expand(B, 3, 32, 32 + 8)
    xp = xp.gather(2, rows)
    cols = (dx[:, None] + ar[None]).view(B, 1, 1, 32).expand(B, 3, 32, 32)
    return xp.gather(3, cols)


@torch.no_grad()
def predict_logits(model, x_uint8, device, bs=2048, use_amp=True):
    """Eval-mode logits for uint8 images (no augmentation). Restores train mode if needed."""
    was_training = model.training
    model.eval()
    out = []
    for i in range(0, x_uint8.shape[0], bs):
        xb = normalize(x_uint8[i:i + bs].to(device, non_blocking=True)).contiguous(memory_format=torch.channels_last)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            out.append(model(xb).float())
    if was_training:
        model.train()
    return torch.cat(out) if out else torch.empty(0, NUM_CLASSES, device=device)
