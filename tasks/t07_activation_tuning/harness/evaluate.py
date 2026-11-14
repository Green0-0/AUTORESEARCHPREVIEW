"""Evaluation step: rebuild the fixed MLP from the artifact, compute CIFAR-100 test top-1, run checks.

usage: python evaluate.py <out_dir> <mode> <wall_s_so_far>
Reads <out_dir>/artifact/{code/activations.py, model.pt} and <out_dir>/train_report.json,
writes <out_dir>/metric.json.
"""
import json
import os
import sys
import time
import traceback
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import t07lib as L  # noqa: E402


def main():
    out, mode = Path(sys.argv[1]).resolve(), sys.argv[2]
    t_start = float(sys.argv[3]) if len(sys.argv) > 3 else time.time()
    art = out / "artifact"
    rep = {}
    value, valid = 0.0, False
    try:
        tr = json.loads((out / "train_report.json").read_text())
    except Exception:  # noqa: BLE001
        tr = {"ok": False, "error": "train_report.json missing"}
    rep["train"] = {k: v for k, v in tr.items() if k != "traceback"}
    try:
        if not tr.get("ok"):
            raise RuntimeError(f"training step failed: {tr.get('error')}")
        device = torch.device("cpu" if mode == "cpu-smoke" or not torch.cuda.is_available() else "cuda")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.manual_seed(int(os.environ.get("AR_SEED", "0")))
        act_mod = L.import_from(art / "code", "activations")
        model = L.build_model(act_mod)
        sd = torch.load(art / "model.pt", map_location="cpu", weights_only=True)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        rep["state_dict_ok"] = not missing and not unexpected
        if missing or unexpected:
            rep["state_dict_errors"] = {"missing": list(missing), "unexpected": list(unexpected)}
        model = model.to(device).float().eval()
        rep["params"] = L.param_report(model)
        rep.update(L.structure_report(model))

        x, y = L.load_split("test", device, limit=1000 if mode == "cpu-smoke" else None)
        correct = 0
        with torch.no_grad():
            for i in range(0, x.shape[0], 2000):
                xb = x[i:i + 2000].float().div_(255.0)
                correct += (model(xb).argmax(1) == y[i:i + 2000]).sum().item()
        value = correct / x.shape[0]
        xa = x[:512].float() / 255.0
        xb2 = x[-512:].float() / 255.0
        rep.update(L.behaviour_report(model, xa, xb2))
        valid = bool(rep["state_dict_ok"] and rep["params"]["params_ok"] and rep["structure_ok"]
                     and rep["behaviour_ok"] and tr.get("structure_ok") and tr.get("within_budget")
                     and tr["params_after"]["params_ok"])
    except Exception as e:  # noqa: BLE001
        rep["error"] = f"{type(e).__name__}: {e}"
        traceback.print_exc()
    metric = {"metric": "cifar100_test_top1", "value": float(value), "valid": valid,
              "constraint_report": rep, "wall_s": round(time.time() - t_start, 2), "mode": mode}
    (out / "metric.json").write_text(json.dumps(metric, indent=2))
    print(json.dumps({k: metric[k] for k in ("metric", "value", "valid", "wall_s", "mode")}))


if __name__ == "__main__":
    main()
