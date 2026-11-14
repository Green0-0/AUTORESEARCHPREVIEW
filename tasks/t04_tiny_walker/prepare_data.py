"""Idempotent: writes the private held-out evaluation seeds for t04_tiny_walker.

$AR_ROOT/private/t04_tiny_walker/eval_seeds.npy   int64 (100,), drawn once with seed 404 from [1e9, 2^31 - 1)
$AR_ROOT/private/t04_tiny_walker/meta.json        env id, ranges, sha256 of eval_seeds.npy
Training seeds must lie in [0, 1e9) (ctx['train_seed_range']), so they are disjoint from these by construction.
Smoke / cpu-smoke evaluate the first 20 / 8 seeds. Offline; no data download (the env is gymnasium's box2d).
usage: python prepare_data.py
"""
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True   # keep harness/ free of __pycache__ (it is copied read-only)
sys.path.insert(0, str(Path(__file__).resolve().parent / "harness"))
import t04lib as L  # noqa: E402

N_EVAL, DRAW_SEED = 100, 404
LO, HI = 10**9, 2**31 - 1


def main():
    d = Path(os.environ.get("AR_ROOT", "/usr/xtmp/kg349/ar")) / "private" / "t04_tiny_walker"
    rng = np.random.default_rng(DRAW_SEED)
    out = []
    while len(out) < N_EVAL:   # distinct draws, in draw order
        s = int(rng.integers(LO, HI))
        if s not in out:
            out.append(s)
    seeds = np.array(out, dtype=np.int64)
    assert seeds.min() >= L.TRAIN_SEED_RANGE[1]
    p = d / "eval_seeds.npy"
    if p.exists():
        old = np.load(p)
        if not np.array_equal(old, seeds):
            raise SystemExit(f"{p} exists with different content; refusing to re-draw")
        print("exists, identical:", p)
    else:
        d.mkdir(parents=True, exist_ok=True)
        np.save(p, seeds)
        print("wrote", p)
    meta = {"env_id": L.ENV_ID, "n_eval": N_EVAL, "draw_seed": DRAW_SEED, "seed_range": [LO, HI],
            "train_seed_range": list(L.TRAIN_SEED_RANGE), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
    (d / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
