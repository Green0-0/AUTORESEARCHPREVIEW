"""Evaluation step (process 2): rebuild the model from artifact/code/model.py, load the state_dict strictly,
run the post-training checks, then compute top-5 (metric) and top-1 on the CIFAR-100 test set in fp32
(TF32 off), with the global RNG seeded once to EVAL_SEED and model.eval(). Writes <out_dir>/metric.json.

usage: python evaluate.py <out_dir> <mode> <t_start_epoch_s>
"""
import json
import os
import sys
import time
import traceback
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import t02lib as L  # noqa: E402

METRIC = "cifar100_test_top5"


def main():
    out, mode, t_start = Path(sys.argv[1]).resolve(), sys.argv[2], float(sys.argv[3])
    t0 = time.time()
    cfg = L.MODES[mode]
    rep = {"mode": mode, "errors": [], "review_flags": []}
    valid, value = True, 0.0
    tr_p = out / "train_report.json"
    tr = json.loads(tr_p.read_text()) if tr_p.exists() else {"ok": False, "error": "no train_report.json"}
    rep["train"] = {k: tr.get(k) for k in ("ok", "error", "train_wall_s", "time_budget_s", "within_budget",
                                           "n_train", "n_params", "max_mem_gb")}
    if not tr.get("ok"):
        valid = False; rep["errors"].append(f"training failed: {tr.get('error')}")
    if tr.get("within_budget") is False:
        valid = False; rep["errors"].append("training exceeded the time budget")
    guard = tr.get("guard")
    rep["guard"] = guard
    if guard is None:
        valid = False; rep["errors"].append("no guard report (training did not complete)")
    else:
        if not guard.get("ok"):
            valid = False; rep["errors"] += [f"guard: {e}" for e in guard.get("errors", [])]
        rep["review_flags"] += guard.get("review_flags", [])
    scan_p = out / "scan.json"
    scan = json.loads(scan_p.read_text()) if scan_p.exists() else None
    rep["scan"] = scan
    if scan is not None:
        if not scan.get("hard_ok", True):
            valid = False; rep["errors"].append("static scan: hard violation (see scan.hard_violations)")
    else:
        rep["review_flags"].append("static scan missing")

    try:
        art = out / "artifact"
        if not (art / "model.pt").exists():
            raise RuntimeError("artifact/model.pt missing")
        dev = torch.device("cpu" if mode == "cpu-smoke" else "cuda")
        if mode == "cpu-smoke":
            torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", min(4, os.cpu_count() or 4))))
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        X, Y = L.load_test(dev, cfg["n_test"])
        mod = L.import_from(art / "code", "model")
        model = mod.build_model()
        sd = torch.load(art / "model.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(sd, strict=True)
        model = model.float().to(dev)
        rep["n_params"] = int(sum(p.numel() for p in model.parameters()))

        tc = time.time()
        cb = cfg["check_bs"]
        chk, errs = L.model_checks(model, X[:cb], X[cb:2 * cb])
        rep["checks"] = chk; rep["checks_s"] = round(time.time() - tc, 1)
        if errs:
            valid = False; rep["errors"] += errs
        if not chk.get("logits_shape_ok"):
            raise RuntimeError(errs[0])
        if chk.get("forward_stochastic"):
            rep["review_flags"].append("forward is stochastic (allowed: randomness inside forward is applied "
                                       "identically at test, seeded with EVAL_SEED)")

        # scored pass
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        te = time.time()
        L.seed_all(L.EVAL_SEED)
        bs = cfg["eval_bs"]
        c1 = c5 = 0
        with torch.no_grad():
            for i in range(0, len(Y), bs):
                z = model(X[i:i + bs]).float()
                if not torch.isfinite(z).all():
                    raise RuntimeError("non-finite logits on the test set")
                top5 = z.topk(5, dim=1).indices
                yb = Y[i:i + bs]
                c1 += int((top5[:, 0] == yb).sum()); c5 += int((top5 == yb[:, None]).any(1).sum())
        n = len(Y)
        value = c5 / n
        rep.update({"top5": value, "top1": c1 / n, "n_test": n, "eval_forward_s": round(time.time() - te, 1),
                    "eval_seed": L.EVAL_SEED, "eval_batch": bs})
    except Exception as e:  # noqa: BLE001
        valid = False
        rep["errors"].append(f"eval failed: {type(e).__name__}: {e}")
        rep["traceback"] = traceback.format_exc()[-3000:]
    rep["eval_wall_s"] = round(time.time() - t0, 1)
    m = {"metric": METRIC, "value": float(value), "valid": bool(valid), "constraint_report": rep,
         "wall_s": round(time.time() - t_start, 1), "mode": mode}
    (out / "metric.json").write_text(json.dumps(m, indent=1))
    print(json.dumps({k: m[k] for k in ("metric", "value", "valid", "wall_s", "mode")}))
    print(json.dumps({"top1": rep.get("top1"), "errors": rep["errors"], "review_flags": rep["review_flags"]}))


if __name__ == "__main__":
    main()
