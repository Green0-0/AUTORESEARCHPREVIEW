"""Idempotent data prep for t13_robust_noadv.

Writes
  $AR_DATA/t13_robust_noadv/train.npz        x uint8 (50000,32,32,3) NHWC, y int64   (solution sees this)
  $AR_DATA/t13_robust_noadv/train_tiny.npz   first 512 train images (cpu-smoke)
  $AR_ROOT/private/t13_robust_noadv/test.npz x uint8 (10000,32,32,3), y int64        (harness eval only)
  meta.json next to each with sha256 of the arrays.
"""
import hashlib, json, os, pickle, sys
from pathlib import Path
import numpy as np

TASK = "t13_robust_noadv"


def _load(paths):
    xs, ys = [], []
    for p in paths:
        with open(p, "rb") as f:
            d = pickle.load(f, encoding="bytes")
        xs.append(d[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1))
        ys.append(np.asarray(d[b"labels"], dtype=np.int64))
    return np.ascontiguousarray(np.concatenate(xs)).astype(np.uint8), np.concatenate(ys)


def _sha(x, y):
    h = hashlib.sha256(); h.update(x.tobytes()); h.update(y.tobytes()); return h.hexdigest()


def _write(path: Path, x, y):
    sha = _sha(x, y)
    meta = path.with_suffix(".meta.json")
    if path.exists() and meta.exists() and json.loads(meta.read_text()).get("sha256") == sha:
        print(f"ok (exists) {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, x=x, y=y)
    os.replace(tmp, path)
    meta.write_text(json.dumps({"sha256": sha, "n": int(len(y)), "shape": list(x.shape)}))
    print(f"wrote {path} n={len(y)}")


def main():
    data = Path(os.environ.get("AR_DATA", "/usr/xtmp/kg349/ar/data"))
    root = Path(os.environ.get("AR_ROOT", "/usr/xtmp/kg349/ar"))
    src = data / "cifar-10-batches-py"
    if not src.is_dir():
        sys.exit(f"missing {src}")
    xtr, ytr = _load([src / f"data_batch_{i}" for i in range(1, 6)])
    xte, yte = _load([src / "test_batch"])
    assert xtr.shape == (50000, 32, 32, 3) and xte.shape == (10000, 32, 32, 3)
    _write(data / TASK / "train.npz", xtr, ytr)
    _write(data / TASK / "train_tiny.npz", xtr[:512], ytr[:512])
    priv = root / "private" / TASK
    _write(priv / "test.npz", xte, yte)
    try:
        os.chmod(priv, 0o700)
    except OSError:
        pass


if __name__ == "__main__":
    main()
