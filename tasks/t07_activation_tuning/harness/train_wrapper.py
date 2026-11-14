"""Training step: build the fixed MLP with the solution's activations, call solution train(), save artifact.

usage: python train_wrapper.py <solution_dir> <out_dir> <mode: full|smoke|cpu-smoke>
"""
import json
import os
import random
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import t07lib as L  # noqa: E402

GRACE_S = 10.0


def main():
    sol, out, mode = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve(), sys.argv[3]
    art = out / "artifact"
    if art.exists():
        shutil.rmtree(art)
    art.mkdir(parents=True)
    rep = {"mode": mode, "ok": False}
    seed = int(os.environ.get("AR_SEED", "0"))
    try:
        if mode == "cpu-smoke":
            device = torch.device("cpu")
            torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 4)))
            x, y = L.load_split("train", device, limit=2048)
        else:
            if not torch.cuda.is_available():
                raise RuntimeError("GPU mode requested but CUDA is unavailable")
            device = torch.device("cuda")
            x, y = L.load_split("train", device)
            torch.zeros(1, device=device).sum().item()  # CUDA init outside the timer
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

        # copy solution code into the artifact (eval rebuilds activations from this copy)
        code = art / "code"
        shutil.copytree(sol, code, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pt", "*.pth",
                                                                 "*.npy", "*.npz", "data", "out*", "runs"))
        act_mod = L.import_from(sol, "activations")
        train_mod = L.import_from(sol, "train")
        model = L.build_model(act_mod).to(device)
        rep["params_before"] = L.param_report(model)

        budget = L.TRAIN_BUDGET_S[mode]
        t0 = time.time()
        ctx = {"mode": mode, "device": device, "time_budget_s": budget, "deadline": t0 + budget,
               "seed": seed, "width": L.WIDTH, "num_classes": L.NUM_CLASSES}
        model.train()
        train_mod.train(model, x, y, ctx)
        if device.type == "cuda":
            torch.cuda.synchronize()
        wall = time.time() - t0
        rep.update({"train_wall_s": round(wall, 2), "time_budget_s": budget,
                    "within_budget": bool(wall <= budget + GRACE_S)})
        rep.update(L.structure_report(model))
        rep["params_after"] = L.param_report(model)
        sd = {k: v.detach().to("cpu") for k, v in model.state_dict().items()}
        torch.save(sd, art / "model.pt")
        rep["ok"] = True
    except Exception as e:  # noqa: BLE001
        rep["error"] = f"{type(e).__name__}: {e}"
        rep["traceback"] = traceback.format_exc()
        print(rep["traceback"], file=sys.stderr)
    (out / "train_report.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps({k: v for k, v in rep.items() if k != "traceback"}, indent=2))
    sys.exit(0 if rep["ok"] else 1)


if __name__ == "__main__":
    main()
