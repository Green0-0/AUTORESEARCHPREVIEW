"""Judge anchors (plan §9.1): the two reference ideas with known positions used by the ranking judge
(J5) and the optional anchor facts of every judge strategy.

* ``STARTER_IDEAS[task]``: a short description of the task's starter solution (``tasks/<task>/starter``,
  written from the starter code and README). Its known judging result is 0.0, since the normalization floor
  is the starter's mean over seeds.
* ``reference_ideas()[task]``: the strong known method (``scripts/make_anchor_ref.py`` REF_IDEAS, i.e. the
  ``anchor_ref_<task>`` idea whose real result is the normalization ``ref``). Its known judging result is 1.0.
* ``anchor_facts(task)``: the two judge-only facts that state the starter (score 0) and reference (score 1)
  metric values from ``tasks/<task>/task.yaml`` (same text as analysis/rejudge_calibration.py).
"""
from __future__ import annotations

import importlib.util
from functools import lru_cache
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]

STARTER_IDEAS: dict[str, str] = {
    "t07_activation_tuning": (
        "Starter baseline: plain ReLU as the activation of all three hidden layers of the fixed MLP (no activation "
        "parameters). Trained with AdamW (lr 3e-3, weight decay 0.05) and a one-cycle schedule, label smoothing 0.1, "
        "GPU flip + 4-pixel reflect-crop augmentation, batch 2048, bf16 autocast, about 1500 epochs within the budget."),
    "t10_label_noise": (
        "Starter baseline: plain cross-entropy (label smoothing 0.1) on the noisy labels with no noise handling at all. "
        "An airbench-like ResNet (widths 64/128/256/512, ~6.5M params), GPU flip and crop augmentation, bf16, batch "
        "1024, SGD-Nesterov lr 0.4 with warmup and linear decay for 40 epochs; flip test-time augmentation at eval."),
    "t15_lwf": (
        "Starter baseline: plain fine-tuning. A ResNet-9-style CNN (6.6M params) is trained on each task in turn with "
        "cross-entropy only (30 epochs per task, batch 256, SGD with OneCycle, GPU augmentation, bf16), with no "
        "auxiliary state, no exemplars and no distillation, so earlier tasks are almost completely forgotten."),
    "t09_tokenizer_transfer": (
        "Starter baseline: naive transfer. The new 32k-token embeddings are initialized randomly (N(0, 0.02)) with a "
        "tied LM head, then the whole GPT-2 model is trained on random 1024-token windows of the Japanese data with "
        "AdamW (lr 6e-4, betas 0.9/0.95, weight decay 0.1, clip 1.0), batch 16x1024, bf16, 3% warmup then cosine "
        "decay to 10%, for the whole budget."),
    "t13_robust_noadv": (
        "Starter baseline: plain cross-entropy training of a ResNet-9 (6.6M params) with GPU flip/crop augmentation, "
        "bf16, batch 512 and 30 epochs of OneCycle SGD. No robustness measure of any kind, so PGD robust accuracy "
        "is essentially zero."),
    "t16_synthetic_data": (
        "Starter baseline: keep each of the 10 real images per class once and add 9 random crop/flip copies of it, "
        "giving 100 images per class (10,000 uint8 images). The synthetic set is not optimized in any way."),
    "t18_anytime": (
        "Starter baseline: a ResNet-9-style CNN (widths 32/64/128/256) with 3 exits (max-pool + linear) after its "
        "stages, trained jointly with the mean cross-entropy (label smoothing 0.1) of the exits, GPU crop+flip, bf16, "
        "batch 512, SGD-Nesterov with a time-based one-cycle schedule that fills the budget. No distillation; the "
        "first exit already costs ~11 MFLOP."),
    "t19_data_ordering": (
        "Starter baseline: a uniform random reshuffle of the training set every epoch (standard i.i.d. ordering), "
        "with no difficulty-, loss- or curriculum-based scheduling."),
    "t04_tiny_walker": (
        "Starter baseline: a linear policy (24 observations -> 4 actions) trained with basic antithetic random search "
        "(ARS-V1 style): each iteration evaluates theta +/- sigma*delta for 2 x n_cpus Gaussian directions on fresh "
        "training terrains and steps along the return-difference-weighted mean direction, with no observation "
        "normalization, no reward shaping and no direction selection, for the 10-minute budget on 16 cores. It learns "
        "to stand still without falling (return about +6) but never learns to walk."),
    "t11_fewshot_seg": (
        "Starter baseline: frozen ImageNet ResNet-50 backbone (stride 32, BN in eval mode) with a 1x1-conv linear head "
        "on the last stage, logits upsampled bilinearly, cross-entropy on the 100 few-shot images with no augmentation, "
        "AdamW at a constant lr 1e-3 for the whole budget (about 15k steps), so it overfits."),
    "t02_self_reg": (
        "Starter baseline: a 2x-wide ResNet-9-style CNN with GroupNorm (26.4M params), plain cross-entropy, no "
        "dropout, no augmentation and no other regularization, trained for about 3.6 of the 15 minutes."),
}

ANCHOR_SHORT: dict[str, str] = {   # one-line names of the reference methods, used in the anchor facts
    "t07_activation_tuning": "per-channel BatchNorm + PReLU activations with AdamW one-cycle, label smoothing and mixup",
    "t10_label_noise": "Early-Learning Regularization (ELR) with strong augmentation and mixup",
    "t15_lwf": "iCaRL-style exemplar memory within the 1 MiB budget with distillation",
    "t09_tokenizer_transfer": "sub-token-average (FVT) embedding initialization with an embedding-only warmup",
    "t13_robust_noadv": "input-gradient regularization (double backpropagation) with Gaussian noise augmentation",
    "t16_synthetic_data": "dataset distillation by distribution matching (DM)",
    "t18_anytime": "a multi-exit network with a very cheap first exit and self-distillation",
    "t19_data_ordering": "online batch selection by loss",
    "t02_self_reg": "multi-view test-time-consistent augmentation",
    "t04_tiny_walker": "Augmented Random Search with observation normalization on a small MLP policy",
    "t11_fewshot_seg": "a DeepLabv3+ decoder with the backbone fine-tuned at a low learning rate and strong augmentation",
}


@lru_cache(maxsize=1)
def reference_ideas() -> dict[str, str]:
    """REF_IDEAS from scripts/make_anchor_ref.py (loaded by path: scripts/ is not a package)."""
    p = REPO / "scripts" / "make_anchor_ref.py"
    spec = importlib.util.spec_from_file_location("_ar_make_anchor_ref", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return dict(mod.REF_IDEAS)


def task_meta(task_id: str) -> dict:
    return yaml.safe_load((REPO / "tasks" / task_id / "task.yaml").read_text())


def anchor_facts(task_id: str) -> list[str]:
    """Judge-only facts naming the 0 and 1 anchors. Raises KeyError/ValueError if unavailable."""
    t = task_meta(task_id)
    if t.get("floor") is None or t.get("ref") is None:
        raise ValueError(f"{task_id}: task.yaml floor/ref not set; no anchor facts")
    return [f"Under this exact budget and harness, the provided starter solution reaches {t['metric']} = {t['floor']:.4g}; "
            f"treat starter-level results as score 0.",
            f"A strong known method ({ANCHOR_SHORT[task_id]}) reaches {t['metric']} = {t['ref']:.4g} under the same "
            f"budget; treat that level as score 1.0. Scores scale linearly between these two anchors."]


def ranking_references(task_id: str) -> list[dict]:
    """[{"label","description","score"}] for the ranking judge: starter = 0.0, reference method = 1.0."""
    refs = reference_ideas()
    if task_id not in STARTER_IDEAS or task_id not in refs:
        raise KeyError(f"no ranking references for task {task_id!r}")
    return [{"label": "A", "description": STARTER_IDEAS[task_id], "score": 0.0},
            {"label": "B", "description": refs[task_id], "score": 1.0}]
