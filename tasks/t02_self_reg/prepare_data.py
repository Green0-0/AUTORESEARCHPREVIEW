"""Idempotent data prep for t02_self_reg.

Picks a fixed 5-images-per-class subset of the CIFAR-100 training set (seed SUBSET_SEED) and writes
  $AR_DATA/t02_self_reg/train_x.npy     uint8 (500, 3, 32, 32) NCHW RGB, sorted by (class, original index)
  $AR_DATA/t02_self_reg/train_y.npy     int64 (500,) fine labels
  $AR_DATA/t02_self_reg/train_idx.npy   int64 (500,) indices into the official CIFAR-100 train pickle
  $AR_DATA/t02_self_reg/meta.json       seed + sha256
  $AR_ROOT/private/t02_self_reg/test_x.npy / test_y.npy   full 10k CIFAR-100 test set (harness eval only)
from the official CIFAR-100 python pickles in $AR_DATA/cifar-100-python/.
Re-running is a no-op if the files exist and their hashes match; a mismatch is an error (never silently
re-drawn, so every run of the task sees the same 500 images).
"""
import hashlib
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np

TASK = "t02_self_reg"
SUBSET_SEED = 2002
PER_CLASS = 5


def _load(path):
    with open(path, "rb") as f:
        d = pickle.load(f, encoding="latin1")
    x = np.asarray(d["data"], dtype=np.uint8).reshape(-1, 3, 32, 32)
    y = np.asarray(d["fine_labels"], dtype=np.int64)
    return x, y


def _sha(*arrs):
    h = hashlib.sha256()
    for a in arrs:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def _save(arr, dst: Path):
    tmp = dst.with_name(dst.stem + ".tmp.npy")
    np.save(tmp, arr)
    os.replace(tmp, dst)


def subset_indices(y):
    rng = np.random.default_rng(SUBSET_SEED)
    idx = []
    for c in range(100):
        cls = np.flatnonzero(y == c)
        idx.extend(sorted(rng.choice(cls, PER_CLASS, replace=False).tolist()))
    return np.asarray(idx, dtype=np.int64)          # grouped by class, ascending index within class


def main():
    ar_data = Path(os.environ["AR_DATA"])
    ar_root = Path(os.environ.get("AR_ROOT", ar_data.parent))
    src = ar_data / "cifar-100-python"
    if not src.is_dir():
        sys.exit(f"missing {src}")
    pub, priv = ar_data / TASK, ar_root / "private" / TASK
    pub.mkdir(parents=True, exist_ok=True); priv.mkdir(parents=True, exist_ok=True)

    xtr, ytr = _load(src / "train")
    idx = subset_indices(ytr)
    x, y = np.ascontiguousarray(xtr[idx]), ytr[idx]
    assert x.shape == (500, 3, 32, 32) and np.all(np.bincount(y, minlength=100) == PER_CLASS)
    sha = _sha(x, y)
    meta_p = pub / "meta.json"
    if meta_p.exists() and all((pub / f).exists() for f in ("train_x.npy", "train_y.npy", "train_idx.npy")):
        old = json.loads(meta_p.read_text())
        disk = _sha(np.load(pub / "train_x.npy"), np.load(pub / "train_y.npy"))
        if old.get("sha256") != sha or disk != sha:
            sys.exit(f"[prepare] subset hash mismatch in {pub}: refusing to overwrite (delete it to re-draw)")
        print(f"[prepare] train subset: exists, sha ok ({pub})")
    else:
        for arr, name in ((x, "train_x"), (y, "train_y"), (idx, "train_idx")):
            _save(arr, pub / f"{name}.npy")
        meta_p.write_text(json.dumps({"seed": SUBSET_SEED, "per_class": PER_CLASS, "n": int(len(y)),
                                      "sha256": sha, "source": "cifar-100-python/train"}, indent=1))
        print(f"[prepare] train subset: wrote {x.shape} -> {pub}")

    fx, fy = priv / "test_x.npy", priv / "test_y.npy"
    if fx.exists() and fy.exists():
        print(f"[prepare] test: exists ({priv})")
    else:
        xte, yte = _load(src / "test")
        assert xte.shape == (10000, 3, 32, 32)
        _save(xte, fx); _save(yte, fy)
        print(f"[prepare] test: wrote {xte.shape} -> {priv}")
    os.chmod(priv, 0o700)


if __name__ == "__main__":
    main()
