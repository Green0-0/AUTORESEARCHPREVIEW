"""Training step (process 1): snapshot solution/policy.py, call the solution's train(ctx) under a timer, check
the returned parameters (count, dtypes, finiteness, .npz size) and save the artifact. The evaluation seeds are
never loaded in this process.

usage: python train_wrapper.py <solution_dir> <out_dir> <mode: full|smoke|cpu-smoke>
writes <out_dir>/artifact/{policy.npz, code/policy.py} and <out_dir>/train_report.json
"""
import json
import os
import resource
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import t04lib as L  # noqa: E402


def _cpu_s():
    s, c = resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)
    return s.ru_utime + s.ru_stime + c.ru_utime + c.ru_stime


def main():
    sol, out, mode = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve(), sys.argv[3]
    art = out / "artifact"
    if art.exists():
        shutil.rmtree(art)
    (art / "code").mkdir(parents=True)
    scratch = out / "scratch"
    scratch.mkdir(exist_ok=True)
    rep = {"mode": mode, "ok": False}
    seed = int(os.environ.get("AR_SEED", "0"))
    cfg = L.MODES[mode]
    try:
        # snapshot policy.py BEFORE training: evaluation uses this copy, so train() cannot rewrite it
        if not (sol / "policy.py").is_file():
            raise FileNotFoundError("solution/policy.py missing")
        shutil.copy(sol / "policy.py", art / "code" / "policy.py")
        rep["policy_sha256"] = L.sha_file(art / "code" / "policy.py")
        sys.path.insert(0, str(sol))
        train_mod = L.import_file(sol / "train.py", "train")   # as `train`, like a plain import
        n_cpus = L.n_cpus_for(mode)
        budget = cfg["budget_s"]
        np.random.seed(seed)
        t0, c0 = time.time(), _cpu_s()
        ctx = {"mode": mode, "seed": seed, "n_cpus": n_cpus, "time_budget_s": budget, "deadline": t0 + budget,
               "env_id": L.ENV_ID, "train_seed_range": L.TRAIN_SEED_RANGE, "max_params": L.MAX_PARAMS,
               "max_bytes": L.MAX_BYTES, "scratch_dir": str(scratch),
               "count_params": lambda p: L.count_params(p, art / "code" / "policy.py")}
        params = train_mod.train(ctx)
        wall = time.time() - t0
        cpu = _cpu_s() - c0
        rep.update({"train_wall_s": round(wall, 2), "time_budget_s": budget, "n_cpus": n_cpus,
                    "within_budget": bool(wall <= budget + L.GRACE_S),
                    "cpu_s_reaped": round(cpu, 1), "cpu_util_reaped": round(cpu / max(wall * n_cpus, 1e-9), 3)})
        prep, perrs = L.check_params(params)
        rep["params"] = prep
        rep["param_errors"] = perrs
        if not perrs:
            np.savez(art / "policy.npz", **params)
            rep["npz_bytes"] = (art / "policy.npz").stat().st_size
        rep["policy_unchanged_during_train"] = (sol / "policy.py").is_file() and \
            L.sha_file(sol / "policy.py") == rep["policy_sha256"]
        rep["ok"] = not perrs
        if perrs:
            rep["error"] = "; ".join(perrs)
    except Exception as e:  # noqa: BLE001
        rep["error"] = f"{type(e).__name__}: {e}"
        rep["traceback"] = traceback.format_exc()
        print(rep["traceback"], file=sys.stderr)
    shutil.rmtree(scratch, ignore_errors=True)
    for p in art.iterdir():   # only policy.npz + code/policy.py may live in the artifact
        if p.name not in ("policy.npz", "code"):
            shutil.rmtree(p) if p.is_dir() else p.unlink()
    (out / "train_report.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps({k: v for k, v in rep.items() if k != "traceback"}, indent=2))
    sys.stdout.flush()
    os._exit(0 if rep["ok"] else 1)   # don't wait on stray non-daemon children


if __name__ == "__main__":
    main()
