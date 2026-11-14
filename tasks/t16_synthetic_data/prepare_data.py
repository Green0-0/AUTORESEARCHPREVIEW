#!/usr/bin/env python
"""t16_synthetic_data: derive the fixed task data. Idempotent.

Writes
  $AR_DATA/t16_synthetic_data/real10.npz   x uint8 (1000,32,32,3), y int64 (1000,), idx int64 (1000,)
      10 real CIFAR-100 *training* images per class, chosen with a fixed seed
      (independent of AR_SEED, so every run and every idea sees the same 1000).
  $AR_ROOT/private/t16_synthetic_data/withheld_arch.py
      The third (withheld) classifier architecture and its fixed training recipe.
      The solution never sees it; harness/evaluate.py imports it at eval time.

Usage: python prepare_data.py [--force]
"""
import hashlib
import os
import pickle
import sys
from pathlib import Path

import numpy as np

TASK_ID = "t16_synthetic_data"
SELECT_SEED = 16
PER_CLASS = 10

# The withheld architecture. Kept here (outside harness/) on purpose; do not
# copy it into harness/ or TASK.md.
WITHHELD_SRC = r'''
"""Withheld classifier for t16_synthetic_data (private; written by prepare_data.py)."""
import torch
import torch.nn as nn

NAME = "withheld"

# Fixed training recipe (same semantics as the visible recipes in harness/archs.py).
RECIPE = dict(steps=1500, batch=512, lr=0.05, momentum=0.9, wd=5e-4,
              warmup_frac=0.1, label_smoothing=0.0, opt="sgd")


class _Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return x + self.fn(x)


def build_model(num_classes=100):
    """ConvMixer-style patch network: 2x2 patch embedding, depthwise/pointwise mixing."""
    dim, depth, k = 256, 6, 5
    layers = [nn.Conv2d(3, dim, kernel_size=2, stride=2), nn.GELU(), nn.BatchNorm2d(dim)]
    for _ in range(depth):
        layers += [
            _Residual(nn.Sequential(nn.Conv2d(dim, dim, k, groups=dim, padding=k // 2),
                                    nn.GELU(), nn.BatchNorm2d(dim))),
            nn.Conv2d(dim, dim, 1), nn.GELU(), nn.BatchNorm2d(dim),
        ]
    layers += [nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(dim, num_classes)]
    return nn.Sequential(*layers)
'''


def _paths():
    ar_data = Path(os.environ.get("AR_DATA", "/usr/xtmp/kg349/ar/data"))
    ar_root = Path(os.environ.get("AR_ROOT", str(ar_data.parent)))
    return ar_data, ar_data / TASK_ID, ar_root / "private" / TASK_ID


def _load_train(ar_data):
    with open(ar_data / "cifar-100-python" / "train", "rb") as f:
        d = pickle.load(f, encoding="latin1")
    x = d["data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).copy()
    y = np.asarray(d["fine_labels"], dtype=np.int64)
    return x, y


def main():
    force = "--force" in sys.argv
    ar_data, out_dir, priv_dir = _paths()
    out_dir.mkdir(parents=True, exist_ok=True)
    priv_dir.mkdir(parents=True, exist_ok=True)

    real = out_dir / "real10.npz"
    if force or not real.exists():
        x, y = _load_train(ar_data)
        rng = np.random.default_rng(SELECT_SEED)
        idx = np.concatenate([np.sort(rng.choice(np.flatnonzero(y == c), PER_CLASS, replace=False))
                              for c in range(100)])
        tmp = out_dir / "real10.tmp.npz"
        np.savez(tmp, x=x[idx], y=y[idx], idx=idx.astype(np.int64))
        os.replace(tmp, real)
        print(f"wrote {real}")
    else:
        print(f"exists {real}")
    d = np.load(real)
    assert d["x"].shape == (1000, 32, 32, 3) and d["x"].dtype == np.uint8
    assert np.all(np.bincount(d["y"], minlength=100) == PER_CLASS)
    print("real10 idx sha1:", hashlib.sha1(d["idx"].tobytes()).hexdigest()[:12])

    w = priv_dir / "withheld_arch.py"
    if force or not w.exists() or w.read_text() != WITHHELD_SRC:
        w.write_text(WITHHELD_SRC)
        print(f"wrote {w}")
    else:
        print(f"exists {w}")


if __name__ == "__main__":
    main()
