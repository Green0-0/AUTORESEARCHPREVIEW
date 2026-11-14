"""Idempotent data prep for t19_data_ordering.

Writes
  $AR_DATA/t19_data_ordering/train_x.npy   uint8 (50000, 3, 32, 32)
  $AR_DATA/t19_data_ordering/train_y.npy   int64 (50000,)  fine labels
  $AR_ROOT/private/t19_data_ordering/test_x.npy, test_y.npy   (10000 test images; never shown to solutions)
"""
import os
import pickle
import sys
from pathlib import Path

import numpy as np

AR_ROOT = Path(os.environ.get("AR_ROOT", "/usr/xtmp/kg349/ar"))
AR_DATA = Path(os.environ.get("AR_DATA", AR_ROOT / "data"))
SRC = AR_DATA / "cifar-100-python"
TRAIN_DIR = AR_DATA / "t19_data_ordering"
PRIV_DIR = AR_ROOT / "private" / "t19_data_ordering"


def load(split):
    with open(SRC / split, "rb") as f:
        d = pickle.load(f, encoding="latin1")
    x = np.asarray(d["data"], dtype=np.uint8).reshape(-1, 3, 32, 32)
    y = np.asarray(d["fine_labels"], dtype=np.int64)
    return x, y


def write(dirpath, prefix, x, y):
    dirpath.mkdir(parents=True, exist_ok=True)
    for name, arr in ((f"{prefix}_x.npy", x), (f"{prefix}_y.npy", y)):
        p = dirpath / name
        if p.exists():
            old = np.load(p, mmap_mode="r")
            if old.shape == arr.shape and np.array_equal(old, arr):
                continue
        tmp = dirpath / (name + ".tmp.npy")
        np.save(tmp, arr)
        os.replace(tmp, p)


def main():
    if not SRC.exists():
        sys.exit(f"missing {SRC}")
    xtr, ytr = load("train")
    xte, yte = load("test")
    assert xtr.shape == (50000, 3, 32, 32) and xte.shape == (10000, 3, 32, 32)
    write(TRAIN_DIR, "train", xtr, ytr)
    write(PRIV_DIR, "test", xte, yte)
    os.chmod(PRIV_DIR, 0o700)
    print(f"train -> {TRAIN_DIR}  test -> {PRIV_DIR}")


if __name__ == "__main__":
    main()
