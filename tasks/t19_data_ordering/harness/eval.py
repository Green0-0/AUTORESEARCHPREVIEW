"""Separate evaluation step: loads <out>/artifact/model.pt, reads the private CIFAR-100 test set,
merges the training report and the static check, writes <out>/metric.json."""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import check  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solution", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", default="full", choices=list(common.MODES))
    ap.add_argument("--train-rc", type=int, default=0)
    ap.add_argument("--wall", type=float, default=None, help="wall seconds of the training process as seen by run.sh")
    args = ap.parse_args()
    out = Path(args.out)
    cfg = common.MODES[args.mode]
    rp = out / "train_report.json"
    rep = json.loads(rp.read_text()) if rp.exists() else {"ok": False, "error": f"no train_report (rc={args.train_rc})"}
    if args.train_rc == 124:
        rep["ok"], rep["error"] = False, "killed by timeout (budget overrun)"
    value, eval_err = 0.0, None
    mp = out / "artifact" / "model.pt"
    if mp.exists():
        try:
            dev = torch.device("cuda" if (cfg["device"] == "cuda" and torch.cuda.is_available()) else "cpu")
            model = common.Net().to(dev).to(memory_format=torch.channels_last)
            model.load_state_dict(torch.load(mp, map_location=dev, weights_only=True))
            xs, ys = common.load_test(cfg["n_test"])
            x = torch.from_numpy(xs).to(dev)
            y = torch.from_numpy(ys).to(dev)
            logits = common.predict_logits(model, x, dev, use_amp=dev.type == "cuda")
            value = float((logits.argmax(1) == y).float().mean())
        except Exception as e:
            eval_err = f"{type(e).__name__}: {e}"
    wall = args.wall if args.wall is not None else rep.get("wall_s", float("nan"))
    static = check.static_check(args.solution)
    hist = rep.get("histogram", {})
    within = (wall <= cfg["budget_s"]) and bool(rep.get("within_budget", False))
    valid = bool(rep.get("ok")) and within and bool(rep.get("presentations_ok")) and eval_err is None and static["ok"]
    cr = dict(train_ok=bool(rep.get("ok")), train_error=rep.get("error"), eval_error=eval_err,
              within_budget=within, budget_s=cfg["budget_s"], presentations_ok=rep.get("presentations_ok"),
              batch_size=rep.get("batch_size"), total_steps=rep.get("total_steps"),
              scheduler_timing_s=rep.get("timing"), probe_examples=rep.get("probe_examples"),
              loop_s=rep.get("loop_s"), histogram={k: v for k, v in hist.items() if k != "count_histogram"},
              count_histogram=hist.get("count_histogram"), static_check=static)
    m = dict(metric="test_top1", value=value, valid=valid, constraint_report=cr, wall_s=wall, mode=args.mode)
    (out / "metric.json").write_text(json.dumps(m, indent=1))
    print(json.dumps({k: m[k] for k in ("metric", "value", "valid", "wall_s", "mode")}))


if __name__ == "__main__":
    main()
