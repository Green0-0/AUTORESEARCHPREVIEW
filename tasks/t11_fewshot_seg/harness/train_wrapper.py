"""Training step (process 1): load the 100 few-shot training images onto the device, build the solution's model
with build_model(make_backbone) (make_backbone = the ImageNet ResNet-50), call train(model, ctx) under a timer,
check structure + weight provenance, save the artifact. Validation data is never loaded in this process.

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
import t11lib as L  # noqa: E402


def main():
    sol, out, mode = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve(), sys.argv[3]
    art = out / "artifact"
    if art.exists():
        shutil.rmtree(art)
    art.mkdir(parents=True)
    scratch = out / "scratch"
    scratch.mkdir(exist_ok=True)
    rep = {"mode": mode, "ok": False}
    seed = int(os.environ.get("AR_SEED", "0"))
    cfg = L.MODES[mode]
    try:
        if mode == "cpu-smoke":
            device = torch.device("cpu")
            torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", min(4, os.cpu_count() or 4))))
        else:
            if not torch.cuda.is_available():
                raise RuntimeError("GPU mode requested but CUDA is unavailable")
            device = torch.device("cuda")
            torch.zeros(1, device=device).sum().item()  # CUDA init outside the timer
        t_load = time.time()
        x, y, hw, meta = L.load_train(device)
        L.imagenet_state()
        rep["load_s"] = round(time.time() - t_load, 1)
        rep["n_train"] = int(x.shape[0])

        shutil.copytree(sol, art / "code", ignore=shutil.ignore_patterns(
            "__pycache__", "*.pyc", "*.pt", "*.pth", "*.npy", "*.npz", "*.safetensors", "*.bin", "data", "out*",
            "runs"))
        audit = L.LoadAudit(scratch)
        budget = cfg["budget_s"]
        if device.type == "cuda":
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        t0 = time.time()   # the budget covers importing the solution, build_model() and train()
        with audit:
            model_mod = L.import_from(sol, "model")
            train_mod = L.import_from(sol, "train")
            L.seed_all(seed)
            model = model_mod.build_model(L.make_backbone_pretrained).to(device)
            ref_shapes = {k: tuple(v.shape) for k, v in model.state_dict().items()}
            ctx = {"mode": mode, "device": device, "time_budget_s": budget, "deadline": t0 + budget,
                   "seed": seed, "num_classes": L.NUM_CLASSES, "ignore_index": L.IGNORE, "res": L.RES,
                   "classes": meta["classes"], "mean": L.MEAN, "std": L.STD,
                   "train_x": x, "train_y": y, "train_hw": hw, "voc_ids": meta["voc_ids"],
                   "make_backbone": L.make_backbone_pretrained, "scratch_dir": str(scratch)}
            model.train()
            train_mod.train(model, ctx)
            if device.type == "cuda":
                torch.cuda.synchronize()
            wall = time.time() - t0
        rep.update({"train_wall_s": round(wall, 2), "time_budget_s": budget,
                    "within_budget": bool(wall <= budget + L.GRACE_S)})
        srep, serrs = L.structure_report(model, ref_shapes)
        rep["structure"] = srep; rep["structure_errors"] = serrs
        rep["torch_load_events"] = audit.events[:20]
        rep["weight_load_violations"] = audit.violations()[:20]
        sd = {k: (v.detach().to("cpu", torch.float32) if v.is_floating_point() else v.detach().cpu()).clone()
              for k, v in model.state_dict().items()}
        rep["backbone_rel_change"] = L.backbone_drift(sd)
        torch.save(sd, art / "model.pt")
        rep["ok"] = True
        if device.type == "cuda":
            rep["max_mem_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    except Exception as e:  # noqa: BLE001
        rep["error"] = f"{type(e).__name__}: {e}"
        rep["traceback"] = traceback.format_exc()
        print(rep["traceback"], file=sys.stderr)
    (out / "train_report.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps({k: v for k, v in rep.items() if k != "traceback"}, indent=2))
    sys.exit(0 if rep["ok"] else 1)


if __name__ == "__main__":
    main()
