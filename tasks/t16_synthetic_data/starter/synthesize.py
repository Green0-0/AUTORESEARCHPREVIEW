"""Starter: identity + simple augmentation copies.

Each of the 10 real images per class is kept once and joined by 9 randomly
flipped / translated (4px reflect-pad crop) copies -> 100 per class, uint8.
"""
import argparse
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget-s", type=float, default=900)
    ap.add_argument("--mode", default="full")
    a = ap.parse_args()

    d = np.load(a.real)
    x, y = d["x"], d["y"]                      # (1000,32,32,3) uint8, (1000,)
    rng = np.random.default_rng(a.seed)
    copies = 10
    xpad = np.pad(x, ((0, 0), (4, 4), (4, 4), (0, 0)), mode="reflect")
    xs, ys = [x], [y]
    for _ in range(copies - 1):
        out = np.empty_like(x)
        dy, dx = rng.integers(0, 9, len(x)), rng.integers(0, 9, len(x))
        flip = rng.random(len(x)) < 0.5
        for i in range(len(x)):
            im = xpad[i, dy[i]:dy[i] + 32, dx[i]:dx[i] + 32]
            out[i] = im[:, ::-1] if flip[i] else im
        xs.append(out)
        ys.append(y)
    xs, ys = np.concatenate(xs), np.concatenate(ys).astype(np.int64)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(Path(a.out) / "synthetic.npz", x=xs, y=ys)
    print(f"wrote {len(xs)} images, per-class max {np.bincount(ys).max()}")


if __name__ == "__main__":
    main()
