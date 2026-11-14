"""Idempotent data prep for t18_anytime (CIFAR-100, fine labels).

Writes
  $AR_DATA/t18_anytime/train.npz        x uint8 [50000,32,32,3], y int64 [50000]   (given to the solution)
  $AR_DATA/t18_anytime/smoke_train.npz  first 5000 of train (for --smoke / --cpu-smoke training)
  $AR_DATA/t18_anytime/smoke_eval.npz   last 2000 of train (for --smoke / --cpu-smoke eval; NOT test data)
  $AR_ROOT/private/t18_anytime/test.npz x uint8 [10000,32,32,3], y int64 [10000]   (eval only; never shown)
"""
import os
import pickle
from pathlib import Path

import numpy as np

AR_ROOT = Path(os.environ.get("AR_ROOT", "/usr/xtmp/kg349/ar"))
AR_DATA = Path(os.environ.get("AR_DATA", AR_ROOT / "data"))
SRC = AR_DATA / "cifar-100-python"
OUT = AR_DATA / "t18_anytime"
PRIV = AR_ROOT / "private" / "t18_anytime"


def load(split):
    with open(SRC / split, "rb") as f:
        d = pickle.load(f, encoding="latin1")
    x = np.asarray(d["data"], dtype=np.uint8).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).copy()
    y = np.asarray(d["fine_labels"], dtype=np.int64)
    return x, y


def save(path, x, y):
    if path.exists():
        with np.load(path) as z:
            if z["x"].shape == x.shape and np.array_equal(z["y"], y):
                print("ok (exists)", path)
                return
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, x=x, y=y)
    os.replace(tmp, path)
    print("wrote", path, x.shape)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    PRIV.mkdir(parents=True, exist_ok=True)
    xtr, ytr = load("train")
    xte, yte = load("test")
    assert xtr.shape == (50000, 32, 32, 3) and xte.shape == (10000, 32, 32, 3)
    assert ytr.max() == 99 and yte.max() == 99
    save(OUT / "train.npz", xtr, ytr)
    save(OUT / "smoke_train.npz", xtr[:5000], ytr[:5000])
    save(OUT / "smoke_eval.npz", xtr[-2000:], ytr[-2000:])
    save(PRIV / "test.npz", xte, yte)


if __name__ == "__main__":
    main()
