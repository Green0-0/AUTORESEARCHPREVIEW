"""Idempotent data prep for t10_label_noise (Label-Noise Survivor).

Writes
  $AR_DATA/t10_label_noise/train_noisy.npz   x uint8 [50000,3,32,32], y int64 noisy labels   (solution-visible)
  $AR_DATA/t10_label_noise/meta.json         noise convention, seed, sha256 of the noisy labels
  $AR_ROOT/private/t10_label_noise/test.npz  x uint8 [10000,3,32,32], y int64 clean test labels (eval only)
  $AR_ROOT/private/t10_label_noise/train_clean_labels.npy, noise_mask.npy (analysis only, never given to a solution)

Noise convention (symmetric "flip" noise): exactly 40% of the 50,000 training indices
(20,000, chosen uniformly without replacement) get a new label drawn uniformly from the
9 classes OTHER than the true one. The new label never coincides with the true label,
so the realised noise rate is exactly 40%. Fixed seed NOISE_SEED; re-running reproduces
the identical file (verified by sha256).
"""
import hashlib, json, os, pickle, sys
from pathlib import Path
import numpy as np

TASK = "t10_label_noise"
NOISE_RATE = 0.40
NOISE_SEED = 20261110
N_CLASSES = 10


def load_batch(p):
    with open(p, "rb") as f:
        d = pickle.load(f, encoding="bytes")
    return d[b"data"].reshape(-1, 3, 32, 32).astype(np.uint8), np.array(d[b"labels"], dtype=np.int64)


def main():
    data = Path(os.environ.get("AR_DATA", "/usr/xtmp/kg349/ar/data"))
    root = Path(os.environ.get("AR_ROOT", str(data.parent)))
    src = data / "cifar-10-batches-py"
    pub, priv = data / TASK, root / "private" / TASK
    pub.mkdir(parents=True, exist_ok=True); priv.mkdir(parents=True, exist_ok=True)

    xs, ys = zip(*[load_batch(src / f"data_batch_{i}") for i in range(1, 6)])
    x, y = np.concatenate(xs), np.concatenate(ys)
    xt, yt = load_batch(src / "test_batch")

    rng = np.random.default_rng(NOISE_SEED)
    n = len(y); k = int(round(NOISE_RATE * n))
    idx = rng.choice(n, size=k, replace=False)
    offs = rng.integers(1, N_CLASSES, size=k)          # 1..9 -> never the true class
    y_noisy = y.copy(); y_noisy[idx] = (y[idx] + offs) % N_CLASSES
    mask = y_noisy != y
    assert mask.sum() == k
    sha = hashlib.sha256(y_noisy.tobytes()).hexdigest()

    meta = {"task": TASK, "noise_rate": NOISE_RATE, "realised_noise_rate": float(mask.mean()),
            "noise_seed": NOISE_SEED, "convention": "symmetric flip: 40% of indices, new label uniform over the 9 other classes",
            "n_train": int(n), "noisy_labels_sha256": sha}
    mf = pub / "meta.json"
    if mf.exists():
        old = json.loads(mf.read_text())
        if old.get("noisy_labels_sha256") != sha:
            sys.exit(f"ERROR: existing {mf} has different noisy labels; refusing to overwrite")
        print("already prepared (sha matches):", sha); return
    np.savez(pub / "train_noisy.npz", x=x, y=y_noisy)
    np.savez(priv / "test.npz", x=xt, y=yt)
    np.save(priv / "train_clean_labels.npy", y)
    np.save(priv / "noise_mask.npy", mask)
    mf.write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
