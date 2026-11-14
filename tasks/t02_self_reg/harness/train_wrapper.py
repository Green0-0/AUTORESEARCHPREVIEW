"""Training step (process 1): load the 500-image subset onto the device, build the solution's model,
install the training-input guard, call the solution's train(model, x, y, ctx) under a timer, save the artifact.
The test set is never loaded in this process.

usage: python train_wrapper.py <solution_dir> <out_dir> <mode: full|smoke|cpu-smoke>
writes <out_dir>/artifact/{model.pt, code/} and <out_dir>/train_report.json
"""
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import t02lib as L  # noqa: E402


def main():
    sol, out, mode = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve(), sys.argv[3]
    art = out / "artifact"
    if art.exists():
        shutil.rmtree(art)
    art.mkdir(parents=True)
    rep = {"mode": mode, "ok": False}
    seed = int(os.environ.get("AR_SEED", "0"))
    cfg = L.MODES[mode]
    guard = None
    try:
        if mode == "cpu-smoke":
            device = torch.device("cpu")
            torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", min(4, os.cpu_count() or 4))))
        else:
            if not torch.cuda.is_available():
                raise RuntimeError("GPU mode requested but CUDA is unavailable")
            device = torch.device("cuda")
            torch.zeros(1, device=device).sum().item()  # CUDA init outside the timer
        x, y = L.load_train(device, cfg["n_train"])
        rep["n_train"] = int(len(y))

        # the artifact carries a copy of the solution code; eval rebuilds the model from it
        shutil.copytree(sol, art / "code", ignore=shutil.ignore_patterns(
            "__pycache__", "*.pyc", "*.pt", "*.pth", "*.npy", "*.npz", "data", "out*", "runs"))
        L.seed_all(seed)
        model_mod = L.import_from(sol, "model")
        train_mod = L.import_from(sol, "train")
        model = model_mod.build_model().to(device)
        rep["n_params"] = int(sum(p.numel() for p in model.parameters()))

        guard = L.InputGuard(model, x).install()
        budget = cfg["train_budget_s"]
        t0 = time.time()
        ctx = {"mode": mode, "device": device, "time_budget_s": budget, "deadline": t0 + budget,
               "seed": seed, "num_classes": L.NUM_CLASSES}
        model.train()
        train_mod.train(model, x, y, ctx)
        if device.type == "cuda":
            torch.cuda.synchronize()
        wall = time.time() - t0
        guard.remove()
        rep.update({"train_wall_s": round(wall, 2), "time_budget_s": budget,
                    "within_budget": bool(wall <= budget + L.GRACE_S)})
        rep["guard"] = guard.report()
        sd = {k: v.detach().to("cpu") for k, v in model.state_dict().items()}
        torch.save(sd, art / "model.pt")
        rep["ok"] = True
        if device.type == "cuda":
            rep["max_mem_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    except Exception as e:  # noqa: BLE001
        rep["error"] = f"{type(e).__name__}: {e}"
        rep["traceback"] = traceback.format_exc()
        print(rep["traceback"], file=sys.stderr)
        if guard is not None and "guard" not in rep:
            try:
                rep["guard"] = guard.report()
            except Exception:  # noqa: BLE001
                pass
    (out / "train_report.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps({k: v for k, v in rep.items() if k != "traceback"}, indent=2))
    sys.exit(0 if rep["ok"] else 1)


if __name__ == "__main__":
    main()
