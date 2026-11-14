"""Shared constants for the t10_label_noise harness (self-contained; torch/numpy + env vars only)."""
import os
from pathlib import Path

TASK = "t10_label_noise"
N_CLASSES = 10
HOLDOUT = 1000           # smoke modes: last HOLDOUT train images held out, scored against NOISY labels

# per-mode limits (seconds). train budget is what the solution's train() call may take.
MODES = {
    "full":      {"train_budget_s": 900, "train_kill_s": 960, "eval_limit_s": 180, "device": "cuda", "n_train": None},
    "smoke":     {"train_budget_s": 60,  "train_kill_s": 100, "eval_limit_s": 60,  "device": "cuda", "n_train": None},
    "cpu-smoke": {"train_budget_s": 90,  "train_kill_s": 130, "eval_limit_s": 40,  "device": "cpu",  "n_train": 2048},
}
ARTIFACT_MAX_BYTES = 512 * 2**20


def data_dir() -> Path:
    return Path(os.environ.get("AR_DATA", "/usr/xtmp/kg349/ar/data")) / TASK


def private_dir() -> Path:
    ar_root = os.environ.get("AR_ROOT") or str(Path(os.environ.get("AR_DATA", "/usr/xtmp/kg349/ar/data")).parent)
    return Path(ar_root) / "private" / TASK
