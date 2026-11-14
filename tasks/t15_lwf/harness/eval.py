"""t15_lwf evaluation (separate process from training).

python eval.py --solution DIR --out DIR --mode M --wall-start EPOCH

Rebuilds solution.build_model(), loads artifact/model.pt (strict) and artifact/aux.pt, and asks
solution.predict(model, aux, x_uint8) for 100-way logits on the CIFAR-100 test set (no task id).
Writes <out>/metric.json, merging the constraint checks from train_report.json.
"""
import argparse, json, os, sys, time, traceback
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solution", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", default="full", choices=list(C.MODES))
    ap.add_argument("--wall-start", type=float, default=None)
    a = ap.parse_args()
    cfg = C.MODES[a.mode]
    out = Path(a.out)
    metric = dict(metric="final_top1_100way", value=0.0, valid=False, constraint_report={},
                  wall_s=None, mode=a.mode)
    cr = metric["constraint_report"]
    try:
        tr = json.loads((out / "train_report.json").read_text())
        cr.update({k: tr.get(k) for k in ("checks", "stream_s", "budget_s", "tasks_completed",
                   "max_aux_state_bytes", "max_scanned_aux_bytes", "max_unaccounted_cuda_bytes",
                   "final_aux_file_bytes", "aux_limit_bytes", "files_written_in_solution", "errors")})
        cr["per_task_learn_s"] = [r["learn_s"] for r in tr.get("tasks", [])]
        cr["model_param_bytes"] = tr["tasks"][-1]["model_param_bytes"] if tr.get("tasks") else None
    except Exception as e:
        tr = {"ok": False}
        cr["errors"] = [f"no train report: {e}"]
    try:
        if not tr.get("ok"):
            raise RuntimeError("training step failed; see train_report.json")
        dev = torch.device(cfg["device"])
        d = np.load(C.private_dir() / "test.npz")
        x_np, y_np = d["x"], d["y"]
        keep = C.subset_per_class(y_np, cfg["test_per_class"], seed=1)
        x = torch.from_numpy(x_np[keep]).to(dev)
        y = torch.from_numpy(y_np[keep]).to(dev)
        for k in ("AR_DATA", "AR_ROOT"):
            os.environ.pop(k, None)
        sol = C.load_solution(Path(a.solution))
        art = out / "artifact"
        model = sol.build_model().to(dev)
        model.load_state_dict(torch.load(art / "model.pt", map_location=dev, weights_only=True), strict=True)
        model.eval()
        aux = torch.load(art / "aux.pt", map_location=dev, weights_only=True)
        t0 = time.perf_counter()
        preds = []
        with torch.no_grad():
            for i in range(0, len(x), 2000):
                logits = sol.predict(model, aux, x[i:i + 2000])
                if logits.shape != (min(2000, len(x) - i), C.NUM_CLASSES):
                    raise ValueError(f"predict must return [B,100] logits, got {tuple(logits.shape)}")
                preds.append(logits.argmax(1))
        pred = torch.cat(preds)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        cr["eval_s"] = round(time.perf_counter() - t0, 2)
        correct = (pred == y).float()
        metric["value"] = round(correct.mean().item() * 100, 3)
        task_of = torch.div(y, C.CPT, rounding_mode="floor")
        cr["per_task_acc"] = [round(correct[task_of == t].mean().item() * 100, 2) for t in range(C.NUM_TASKS)]
        cr["pred_task_hist"] = torch.bincount(torch.div(pred, C.CPT, rounding_mode="floor"),
                                              minlength=C.NUM_TASKS).tolist()
        checks = tr.get("checks", {})
        metric["valid"] = bool(checks) and all(checks.values())
    except Exception as e:
        cr.setdefault("errors", [])
        cr["errors"] = (cr["errors"] or []) + ["eval: " + "".join(traceback.format_exception(e))[-3000:]]
        traceback.print_exc()
    if a.wall_start is not None:
        metric["wall_s"] = round(time.time() - a.wall_start, 2)
    (out / "metric.json").write_text(json.dumps(metric, indent=1))
    print(json.dumps({k: metric[k] for k in ("metric", "value", "valid", "wall_s", "mode")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
