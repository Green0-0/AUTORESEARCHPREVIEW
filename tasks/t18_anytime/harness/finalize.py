"""Combine train/eval status into <out>/metric.json (contract format)."""
import argparse
import json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--mode", required=True)
ap.add_argument("--train-rc", type=int, required=True)
ap.add_argument("--eval-rc", type=int, required=True)
ap.add_argument("--train-s", type=float, required=True, help="ms")
ap.add_argument("--eval-s", type=float, required=True, help="ms")
ap.add_argument("--train-budget", type=float, required=True)
a = ap.parse_args()
out = Path(a.out)
ev = out / "eval.json"
res = json.loads(ev.read_text()) if (a.eval_rc == 0 and ev.exists()) else \
    {"metric": "expected_top1_at_interruption", "value": 0.0, "valid": False, "constraint_report": {"errors": []}}
rep = res["constraint_report"]
rep.update({"train_rc": a.train_rc, "eval_rc": a.eval_rc, "train_s": round(a.train_s / 1000, 1),
            "eval_s": round(a.eval_s / 1000, 1), "train_budget_s": a.train_budget})
if a.train_rc != 0:
    res["valid"] = False
    rep["errors"].append("training timed out (budget overrun)" if a.train_rc in (124, 137)
                         else f"training failed rc={a.train_rc} (see train.log)")
if a.eval_rc != 0:
    res["valid"] = False
    rep["errors"].append(f"evaluation failed rc={a.eval_rc} (see eval.log)")
res["wall_s"] = round((a.train_s + a.eval_s) / 1000, 1)
res["mode"] = a.mode
(out / "metric.json").write_text(json.dumps(res, indent=1))
print(json.dumps({k: res[k] for k in ("metric", "value", "valid", "wall_s", "mode")}), rep["errors"])
