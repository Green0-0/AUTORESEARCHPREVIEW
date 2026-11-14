"""Build $AR_ROOT/decoy_private/: same file names, shapes and dtypes as $AR_ROOT/private/, random content.

realjudge/jail.sh (claude mode) mounts this over $AR_ROOT/private so that Claude's own `run.sh --cpu-smoke`
self-tests run end to end (at chance accuracy) without exposing any real test data. Re-run after adding a task.
usage: python scripts/make_decoy_private.py
"""
import json, os, shutil
from pathlib import Path
import numpy as np

ROOT = Path(os.environ.get("AR_ROOT", "/usr/xtmp/kg349/ar"))
SRC, FINAL = ROOT / "private", ROOT / "decoy_private"
DST = ROOT / "decoy_private.new"   # built here, then swapped in atomically (running jails keep their old mount)
rng = np.random.default_rng(12345)

DECOY_ARCH = '''"""DECOY withheld architecture (the real one is not visible here)."""
import torch.nn as nn

NAME = "decoy"
RECIPE = dict(steps=1500, batch=512, lr=0.05, momentum=0.9, wd=5e-4,
              warmup_frac=0.1, label_smoothing=0.0, opt="sgd")


def build_model(num_classes=100):
    return nn.Sequential(nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                         nn.Linear(32, num_classes))
'''


def fake(a: np.ndarray) -> np.ndarray:
    if a.dtype == np.bool_:
        return rng.random(a.shape) < a.mean()
    if np.issubdtype(a.dtype, np.integer):
        lo, hi = int(a.min()), int(a.max())
        return rng.integers(lo, hi + 1, a.shape).astype(a.dtype)
    return rng.standard_normal(a.shape).astype(a.dtype)


def main():
    if DST.exists():
        for p in DST.rglob("*"):
            p.chmod(0o755 if p.is_dir() else 0o644)
        shutil.rmtree(DST)
    for f in sorted(SRC.rglob("*")):
        if f.is_dir() or "__pycache__" in f.parts:
            continue
        out = DST / f.relative_to(SRC)
        out.parent.mkdir(parents=True, exist_ok=True)
        if f.suffix == ".npy":
            np.save(out, fake(np.load(f)))
        elif f.suffix == ".npz":
            d = np.load(f)
            np.savez(out, **{k: fake(d[k]) for k in d.files})
        elif f.suffix == ".json":
            meta = json.loads(f.read_text())
            meta = {k: ("decoy" if k in ("sha256", "md5") else v) for k, v in meta.items()}
            out.write_text(json.dumps(meta))
        elif f.name == "withheld_arch.py":
            out.write_text(DECOY_ARCH)
        else:
            raise SystemExit(f"no decoy rule for {f}")
        print("decoy", out)
    # raw CIFAR-100 test pickle (t16's evaluator reads it directly): same structure, random content.
    # Its content never changes, so it is built once and never rebuilt in place (running jails mount it).
    import pickle
    raw = ROOT / "decoy_raw" / "cifar-100-python"
    if not (raw / "test").exists():
        raw.mkdir(parents=True, exist_ok=True)
        with open(ROOT / "data" / "cifar-100-python" / "test", "rb") as f:
            d = pickle.load(f, encoding="bytes")
        d = {k: (fake(np.asarray(v)) if k in (b"data", b"fine_labels", b"coarse_labels") else v) for k, v in d.items()}
        for k in (b"fine_labels", b"coarse_labels"):
            d[k] = [int(v) for v in d[k]]
        d = {k.decode(): (v.decode() if isinstance(v, bytes) else v) for k, v in d.items()}   # str keys, as latin1 loads give
        with open(raw / "test", "wb") as f:
            pickle.dump(d, f)
        shutil.copy(ROOT / "data" / "cifar-100-python" / "meta", raw / "meta")   # class names only
        print("decoy", raw / "test")
    for top in (DST, raw.parent):
        for p in top.rglob("*"):
            p.chmod(0o555 if p.is_dir() else 0o444)
    # atomic swap: running Claude jails bind-mounted the previous dir by inode and keep seeing it (kept as .old)
    old = ROOT / "decoy_private.old"
    if old.exists():
        for p in old.rglob("*"):
            p.chmod(0o755 if p.is_dir() else 0o644)
        shutil.rmtree(old)
    if FINAL.exists():
        FINAL.rename(old)
    DST.rename(FINAL)
    print("decoys swapped into", FINAL)


if __name__ == "__main__":
    main()
