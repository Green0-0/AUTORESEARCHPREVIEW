"""Idempotent data prep for t07_activation_tuning.

Writes
  $AR_DATA/t07_activation_tuning/train_x.npy   uint8 (50000, 3, 32, 32)
  $AR_DATA/t07_activation_tuning/train_y.npy   int64 (50000,)  fine labels
  $AR_ROOT/private/t07_activation_tuning/test_x.npy / test_y.npy   (10000 test images; eval only)
from the official CIFAR-100 python pickles in $AR_DATA/cifar-100-python/.
"""
import os
import pickle
from pathlib import Path

import numpy as np

TASK = "t07_activation_tuning"


def _load(path):
    with open(path, "rb") as f:
        d = pickle.load(f, encoding="latin1")
    x = np.asarray(d["data"], dtype=np.uint8).reshape(-1, 3, 32, 32)
    y = np.asarray(d["fine_labels"], dtype=np.int64)
    return x, y


def _save(dirpath: Path, x, y):
    dirpath.mkdir(parents=True, exist_ok=True)
    for name, arr in (("x", x), ("y", y)):
        tmp = dirpath / f".{name}.tmp.npy"
        np.save(tmp, arr)
        os.replace(tmp, dirpath / f"{dirpath.name_prefix}_{name}.npy" if hasattr(dirpath, "name_prefix") else tmp)


def main():
    ar_data = Path(os.environ["AR_DATA"])
    ar_root = Path(os.environ.get("AR_ROOT", ar_data.parent))
    src = ar_data / "cifar-100-python"
    out_pub = ar_data / TASK
    out_priv = ar_root / "private" / TASK
    jobs = [("train", src / "train", out_pub), ("test", src / "test", out_priv)]
    for split, pkl, outdir in jobs:
        fx, fy = outdir / f"{split}_x.npy", outdir / f"{split}_y.npy"
        if fx.exists() and fy.exists():
            print(f"[prepare] {split}: exists ({fx})")
            continue
        x, y = _load(pkl)
        outdir.mkdir(parents=True, exist_ok=True)
        for arr, dst in ((x, fx), (y, fy)):
            tmp = dst.with_name(dst.stem + ".tmp.npy")
            np.save(tmp, arr)
            os.replace(tmp, dst)
        print(f"[prepare] {split}: wrote {x.shape} {y.shape} -> {outdir}")
    os.chmod(out_priv, 0o700)


if __name__ == "__main__":
    main()
