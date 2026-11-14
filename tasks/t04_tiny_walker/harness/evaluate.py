"""Evaluation step (process 2): load artifact/policy.npz + the policy.py snapshot, apply the parameter / size /
policy-file rules, then roll out the policy on the fixed held-out terrain seeds (deterministic) and write
metric.json. Metric = mean undiscounted episode return (higher is better).

usage: python evaluate.py <out_dir> <mode> <t0> <solution_dir>
"""
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import t04lib as L  # noqa: E402

METRIC = "heldout_mean_return"


def main():
    out, mode, t0, sol = Path(sys.argv[1]).resolve(), sys.argv[2], float(sys.argv[3]), Path(sys.argv[4]).resolve()
    cfg = L.MODES[mode]
    rep, errs = {}, []
    value = -1e9
    try:
        tr = json.loads((out / "train_report.json").read_text()) if (out / "train_report.json").exists() else {}
        rep["train"] = {k: tr.get(k) for k in ("ok", "error", "train_wall_s", "time_budget_s", "within_budget",
                                                "n_cpus", "cpu_util_reaped", "policy_unchanged_during_train")}
        scan = json.loads((out / "scan.json").read_text()) if (out / "scan.json").exists() else {}
        rep["scan"] = {"hard_ok": scan.get("hard_ok"), "hard_violations": scan.get("hard_violations", [])[:10],
                       "n_review_flags": scan.get("n_review_flags"), "review_flags": scan.get("review_flags", [])[:20]}
        if not tr.get("ok"):
            errs.append(f"training step failed: {tr.get('error', 'no train_report.json')}")
        if tr.get("ok") and not tr.get("within_budget", False):
            errs.append("train() exceeded the time budget")
        if not scan.get("hard_ok", False):
            errs.append("static scan: hard violations (see constraint_report.scan)")
        if tr.get("ok") and not tr.get("policy_unchanged_during_train", False):
            errs.append("solution/policy.py was modified or removed during train()")
        npz, pol = out / "artifact" / "policy.npz", out / "artifact" / "code" / "policy.py"
        if not npz.is_file() or not pol.is_file():
            raise FileNotFoundError("artifact/policy.npz or the policy.py snapshot is missing")
        if (sol / "policy.py").is_file() and L.sha_file(sol / "policy.py") != L.sha_file(pol):
            errs.append("policy.py snapshot differs from solution/policy.py")
        with np.load(npz, allow_pickle=False) as z:
            params = {k: z[k] for k in z.files}
        prep, perrs = L.check_params(params)
        ps = L.scan_policy(pol)
        n_params = (prep.get("n_array_params") or 0) + ps["n_literal_params"]
        size = npz.stat().st_size
        rep.update({"n_params": n_params, "n_array_params": prep.get("n_array_params"),
                    "n_literal_params": ps["n_literal_params"], "arrays": prep.get("arrays"),
                    "npz_bytes": size, "max_params": L.MAX_PARAMS, "max_bytes": L.MAX_BYTES,
                    "policy_scan": {k: ps[k] for k in ("literals", "string_chars", "source_bytes", "errors")
                                    if k in ps}})
        errs += perrs + [f"policy.py: {e}" for e in ps["errors"]]
        rep["params_ok"] = n_params <= L.MAX_PARAMS
        rep["size_ok"] = size <= L.MAX_BYTES
        if not rep["params_ok"]:
            errs.append(f"{n_params} parameters > {L.MAX_PARAMS}")
        if not rep["size_ok"]:
            errs.append(f"policy.npz is {size} bytes > {L.MAX_BYTES}")
        if ps["errors"]:
            raise RuntimeError("policy.py violates the policy-file rules; not evaluated")

        seeds = np.load(L.private_dir() / "eval_seeds.npy")[: cfg["n_eval"]]
        n_procs = L.n_cpus_for(mode)
        te = time.time()
        res = L.evaluate_seeds(pol, npz, seeds, n_procs)
        rep["eval_s"] = round(time.time() - te, 1)
        # determinism: a fresh Policy in this process must reproduce episode 0 exactly
        mod = L.import_file(pol)
        again = L.rollout(mod.Policy({k: v.copy() for k, v in params.items()}), L.make_env(), int(seeds[0]))
        rep["deterministic"] = again["return"] == res[0]["return"] and again["steps"] == res[0]["steps"]
        if not rep["deterministic"]:
            errs.append(f"non-deterministic policy: episode 0 gave {res[0]['return']} then {again['return']}")
        if not all(r["action_ok"] for r in res):
            errs.append("policy returned non-finite or wrongly shaped actions")
        rets = np.array([r["return"] for r in res])
        value = float(rets.mean())
        rep.update({"env_id": L.ENV_ID, "n_episodes": len(res), "return_sd": round(float(rets.std(ddof=1)), 3)
                    if len(res) > 1 else None, "return_median": round(float(np.median(rets)), 3),
                    "return_min": round(float(rets.min()), 3), "return_max": round(float(rets.max()), 3),
                    "frac_fell": round(float(np.mean([r["fell"] for r in res])), 3),
                    "frac_ge_300": round(float(np.mean(rets >= 300)), 3),
                    "mean_steps": round(float(np.mean([r["steps"] for r in res])), 1)})
    except Exception as e:  # noqa: BLE001
        errs.append(f"eval error: {type(e).__name__}: {e}")
        rep["traceback"] = traceback.format_exc()[-3000:]
        print(rep["traceback"], file=sys.stderr)
    rep["errors"] = errs
    m = {"metric": METRIC, "value": value, "valid": not errs, "constraint_report": rep,
         "wall_s": round(time.time() - t0, 1), "mode": mode}
    (out / "metric.json").write_text(json.dumps(m, indent=1))
    print(json.dumps({k: m[k] for k in ("metric", "value", "valid", "wall_s", "mode")}), "errors:", errs)


if __name__ == "__main__":
    main()
