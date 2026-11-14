"""Evaluation step (own process). Loads the artifact via solution.load_predictor() and scores it.
full mode: clean CIFAR-10 test top-1 (labels read here, never passed to the solution).
smoke modes: top-1 against the NOISY labels of the held-out last HOLDOUT train images (no test feedback).
usage: eval_entry.py <solution_dir> <artifact_dir> <mode> <report.json>"""
import importlib.util, json, os, sys, time, traceback
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import MODES, HOLDOUT, N_CLASSES, data_dir, private_dir  # noqa: E402


def main():
    sol_dir, art_dir, mode, rep_path = sys.argv[1:5]
    cfg = MODES[mode]
    rep = {"mode": mode, "ok": False}
    try:
        if mode == "full":
            d = np.load(private_dir() / "test.npz"); x, y = d["x"], d["y"]
            rep["metric"] = "clean_test_top1"
        else:
            d = np.load(data_dir() / "train_noisy.npz"); x, y = d["x"][-HOLDOUT:], d["y"][-HOLDOUT:]
            rep["metric"] = "noisy_holdout_top1"
        for k in ("AR_DATA", "AR_ROOT", "AR_REPO"):
            os.environ.pop(k, None)
        import torch
        dev = torch.device(cfg["device"])
        images = torch.from_numpy(x).to(dev)
        sys.path.insert(0, sol_dir)
        spec = importlib.util.spec_from_file_location("solution", Path(sol_dir) / "solution.py")
        sol = importlib.util.module_from_spec(spec); sys.modules["solution"] = sol
        spec.loader.exec_module(sol)
        t0 = time.time()
        predict = sol.load_predictor(art_dir, dev)
        preds = []
        bs = 1000 if dev.type == "cuda" else 250
        with torch.inference_mode():
            for i in range(0, len(images), bs):
                out = predict(images[i:i + bs])
                assert out.shape == (min(bs, len(images) - i), N_CLASSES), f"bad logits shape {tuple(out.shape)}"
                preds.append(out.float().argmax(1).cpu())
        rep["eval_s"] = time.time() - t0
        p = torch.cat(preds).numpy()
        rep["value"] = float((p == y).mean())
        rep["n"] = int(len(y))
        rep["ok"] = True
    except BaseException:
        rep["error"] = traceback.format_exc()[-4000:]
    Path(rep_path).write_text(json.dumps(rep, indent=1))
    sys.exit(0 if rep["ok"] else 1)


if __name__ == "__main__":
    main()
