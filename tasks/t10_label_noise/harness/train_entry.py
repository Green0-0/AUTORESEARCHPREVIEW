"""Training step (own process). Loads ONLY the noisy-label train file, hands it to solution.train().
usage: train_entry.py <solution_dir> <artifact_dir> <mode> <report.json>"""
import importlib.util, json, os, random, sys, time, traceback
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import MODES, HOLDOUT, N_CLASSES, data_dir  # noqa: E402


def main():
    sol_dir, art_dir, mode, rep_path = sys.argv[1:5]
    cfg = MODES[mode]
    seed = int(os.environ.get("AR_SEED", "0"))
    rep = {"mode": mode, "seed": seed, "ok": False}
    try:
        d = np.load(data_dir() / "train_noisy.npz")
        x, y = d["x"], d["y"]                       # noisy labels only; clean labels are not in this file
        if mode != "full":
            x, y = x[:-HOLDOUT], y[:-HOLDOUT]
        if cfg["n_train"]:
            x, y = x[:cfg["n_train"]], y[:cfg["n_train"]]
        # hide data locations from the solution
        for k in ("AR_DATA", "AR_ROOT", "AR_REPO"):
            os.environ.pop(k, None)
        import torch
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        dev = torch.device(cfg["device"])
        if dev.type == "cuda":
            assert torch.cuda.is_available(), "no GPU"
        images = torch.from_numpy(x).to(dev)                     # uint8 [N,3,32,32]
        labels = torch.from_numpy(y).to(dev)                     # int64 [N], 40% wrong
        if dev.type == "cuda":
            torch.cuda.synchronize()
        sys.path.insert(0, sol_dir)
        spec = importlib.util.spec_from_file_location("solution", Path(sol_dir) / "solution.py")
        sol = importlib.util.module_from_spec(spec); sys.modules["solution"] = sol
        spec.loader.exec_module(sol)
        Path(art_dir).mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        info = {"mode": mode, "budget_s": cfg["train_budget_s"], "deadline": t0 + cfg["train_budget_s"],
                "device": dev, "seed": seed, "n_classes": N_CLASSES}
        rep["t_start"] = t0
        sol.train(images, labels, art_dir, info)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        rep["train_s"] = time.time() - t0
        rep["ok"] = True
    except BaseException:
        rep["error"] = traceback.format_exc()[-4000:]
    Path(rep_path).write_text(json.dumps(rep, indent=1))
    sys.exit(0 if rep["ok"] else 1)


if __name__ == "__main__":
    main()
