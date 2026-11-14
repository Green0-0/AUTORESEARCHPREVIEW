"""Idempotent data prep for t15_lwf (class-incremental Split CIFAR-100).

Writes
  $AR_DATA/t15_lwf/train.npz           x uint8 [50000,3,32,32], y int64 (RELABELED: task t owns labels 10t..10t+9)
  $AR_DATA/t15_lwf/class_order.json    the fixed permutation of original CIFAR-100 fine labels
  $AR_ROOT/private/t15_lwf/test.npz    x uint8 [10000,3,32,32], y int64 (relabeled the same way; never visible to solutions)

The class order is fixed by CLASS_ORDER_SEED, independent of AR_SEED.
"""
import json, os, pickle, sys
from pathlib import Path
import numpy as np

CLASS_ORDER_SEED = 1993
NUM_TASKS, CLASSES_PER_TASK = 10, 10

def main():
    data = Path(os.environ.get("AR_DATA", "/usr/xtmp/kg349/ar/data"))
    root = Path(os.environ.get("AR_ROOT", str(data.parent)))
    out, priv = data / "t15_lwf", root / "private" / "t15_lwf"
    out.mkdir(parents=True, exist_ok=True); priv.mkdir(parents=True, exist_ok=True)
    order = np.random.RandomState(CLASS_ORDER_SEED).permutation(100)
    remap = np.empty(100, dtype=np.int64); remap[order] = np.arange(100)
    meta = pickle.load(open(data / "cifar-100-python" / "meta", "rb"), encoding="latin1")
    names = meta["fine_label_names"]
    spec = {"seed": CLASS_ORDER_SEED, "order": order.tolist(),
            "tasks": [[names[c] for c in order[t*10:(t+1)*10]] for t in range(NUM_TASKS)]}
    for split, dst in (("train", out / "train.npz"), ("test", priv / "test.npz")):
        if dst.exists() and (out / "class_order.json").exists():
            print("exists", dst); continue
        d = pickle.load(open(data / "cifar-100-python" / split, "rb"), encoding="latin1")
        x = np.asarray(d["data"], dtype=np.uint8).reshape(-1, 3, 32, 32)
        y = remap[np.asarray(d["fine_labels"], dtype=np.int64)]
        np.savez(dst, x=x, y=y)
        print("wrote", dst, x.shape, np.bincount(y).min(), np.bincount(y).max())
    (out / "class_order.json").write_text(json.dumps(spec, indent=1))
    return 0

if __name__ == "__main__":
    sys.exit(main())
