"""Evaluation step for t18_anytime. Loads the trained artifact via <solution>/model.py:load_model,
counts per-stage FLOPs on probe batches, runs batched anytime inference on the eval set,
and writes <out>/eval.json with the metric and the constraint report."""
import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import anytime as A  # noqa: E402

EVAL_BS = 500
FLOP_REL_TOL = 0.01
AGREE_MIN = 0.95


def load_eval_data(mode):
    root = Path(os.environ.get("AR_ROOT", "/usr/xtmp/kg349/ar"))
    data = Path(os.environ.get("AR_DATA", root / "data"))
    if mode == "full":
        path = root / "private" / "t18_anytime" / "test.npz"
    else:  # smoke modes never touch the test set
        path = data / "t18_anytime" / "smoke_eval.npz"
    with np.load(path) as z:
        x, y = z["x"], z["y"]
    if mode == "cpu-smoke":
        x, y = x[:1000], y[:1000]
    return x, y, str(path)


def import_solution(sol_dir):
    sys.path.insert(0, str(sol_dir))
    spec = importlib.util.spec_from_file_location("solution_model", Path(sol_dir) / "model.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solution", required=True)
    ap.add_argument("--artifact", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", default="full", choices=["full", "smoke", "cpu-smoke"])
    a = ap.parse_args()
    seed = int(os.environ.get("AR_SEED", "0"))
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if (a.mode != "cpu-smoke" and torch.cuda.is_available()) else "cpu")
    report = {"device": str(device), "B_min_flops": A.B_MIN, "B_max_flops": A.B_MAX,
              "distribution": "log-uniform FLOPs/image", "errors": []}
    valid = True

    xnp, ynp, path = load_eval_data(a.mode)
    report["eval_data"] = path
    report["n_eval"] = int(len(ynp))
    x_all = torch.from_numpy(xnp).to(device).permute(0, 3, 1, 2).float().div_(255.0).contiguous()
    y_all = torch.from_numpy(ynp).to(device)

    mod = import_solution(a.solution)
    model = mod.load_model(a.artifact, device)
    if hasattr(model, "eval"):
        model.eval()

    # 1) FLOP schedule on two probe batches of different size and content
    t0 = time.time()
    c1, p1 = A.count_stage_flops(model, x_all[0:64])
    c2, _ = A.count_stage_flops(model, x_all[64:192])
    report["flop_count_s"] = round(time.time() - t0, 2)
    report["stage_flops_per_image"] = c1
    if len(c1) == 0:
        raise SystemExit("model emitted no predictions")
    same = len(c1) == len(c2) and all(abs(u - v) <= FLOP_REL_TOL * max(u, v) + 1e3 for u, v in zip(c1, c2))
    report["check_flops_input_independent"] = bool(same)
    if not same:
        valid = False
        report["errors"].append(f"per-image FLOPs differ across probe batches: {c1} vs {c2}")

    # 2) predictions for an image must not depend on the rest of the batch
    xb = torch.cat([x_all[0:32], x_all[192:224]])
    pb = A.run_stages(model, xb)
    agree = [float((u[:32] == v[:32]).float().mean()) for u, v in zip(p1, pb)] if len(pb) == len(p1) else [0.0]
    report["batch_independence_agreement"] = agree
    if len(pb) != len(p1) or min(agree) < AGREE_MIN:
        valid = False
        report["errors"].append("predictions depend on batch composition or stage count varies")

    # 3) batched inference over the full eval set (uncounted, timed)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    correct = None
    for i in range(0, len(y_all), EVAL_BS):
        preds = A.run_stages(model, x_all[i:i + EVAL_BS])
        if len(preds) != len(c1):
            valid = False
            report["errors"].append("number of stages varies between batches")
            break
        hits = torch.stack([(p == y_all[i:i + EVAL_BS]).sum() for p in preds])
        correct = hits if correct is None else correct + hits
    if device.type == "cuda":
        torch.cuda.synchronize()
    infer_s = time.time() - t0
    accs = (correct.double() / len(y_all)).tolist()
    report["infer_s"] = round(infer_s, 2)
    report["effective_tflops"] = round(c1[-1] * len(y_all) / max(infer_s, 1e-9) / 1e12, 3)

    w = A.stage_weights(c1)
    value = A.expected_accuracy(c1, accs)
    report["stages"] = [{"k": k + 1, "cum_flops_per_image": c, "top1": acc, "weight": wk,
                         "P_reached": A.survival(c)} for k, (c, acc, wk) in enumerate(zip(c1, accs, w))]
    report["P_no_prediction"] = 1.0 - A.survival(c1[0])
    report["final_stage_top1"] = accs[-1]
    report["final_stage_only_score"] = accs[-1] * A.survival(c1[-1])
    report["oracle_best_stage_top1"] = max(accs)
    out = {"metric": "expected_top1_at_interruption", "value": value, "valid": valid,
           "constraint_report": report}
    Path(a.out, "eval.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: report[k] for k in ("stage_flops_per_image", "infer_s", "effective_tflops")}))
    for s in report["stages"]:
        print(f"stage {s['k']}: {s['cum_flops_per_image']/1e6:9.2f} MFLOP  top1 {s['top1']:.4f}  "
              f"P_reached {s['P_reached']:.3f}  weight {s['weight']:.3f}")
    print(f"expected top-1 at interruption: {value:.4f}  valid={valid}")


if __name__ == "__main__":
    main()
