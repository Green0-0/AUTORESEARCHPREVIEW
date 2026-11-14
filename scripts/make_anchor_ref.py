"""Create one-idea "anchor" runs whose single idea is a well-known strong method for the task.

Running these through the real-eval chain (python -m ar.realjudge submit) does two jobs:
  1. an end-to-end test of the Claude implement -> GPU run -> Claude review chain;
  2. the `ref` anchor for real_score normalization (plan §3.3).

usage: python scripts/make_anchor_ref.py [task_id ...]     (default: every task in REF_IDEAS)
Creates $AR_ROOT/runs/anchor_ref_<task_id>/ with idea i000000 in status "promoted".
"""
import sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ar import config as C, journal as J  # noqa: E402

REF_IDEAS = {
    "t07_activation_tuning": (
        "Normalized learnable activations. In each hidden layer use, as the activation applied to Wx, a "
        "per-channel BatchNorm1d followed by a per-channel PReLU (learnable negative slope per neuron); "
        "these add only ~0.1% parameters. Train with AdamW, a one-cycle learning-rate schedule, label "
        "smoothing 0.1, GPU flip/crop augmentation and mixup (alpha 0.2), using most of the time budget."),
    "t10_label_noise": (
        "Early-Learning Regularization (ELR, Liu et al. 2020). Train a fast ResNet with cross-entropy plus "
        "the ELR term lambda*log(1 - <p, t>), where t is a per-example temporal ensemble (EMA, beta 0.7) "
        "of the model's own softmax predictions, so the model keeps agreeing with what it learned early "
        "instead of memorizing the noisy labels. Use lambda around 3, strong GPU augmentation, and mixup."),
    "t15_lwf": (
        "iCaRL-style class-incremental learning within the 1 MiB auxiliary budget. Keep an exemplar memory "
        "of as many uint8 32x32x3 images as fit in 1 MiB (about 330), chosen per class by herding and "
        "rebalanced as classes arrive. Train each task on new data plus the exemplars, with knowledge "
        "distillation from a frozen copy of the previous model on old-class logits, and classify with the "
        "nearest-mean-of-exemplars rule in feature space (or bias-corrected logits)."),
    "t13_robust_noadv": (
        "Input-gradient regularization with noise augmentation. Train a ResNet with cross-entropy plus a "
        "double-backpropagation penalty on the L1/L2 norm of the loss gradient with respect to the input "
        "(no adversarial examples are generated; the gradient norm is only penalized), together with "
        "Gaussian noise augmentation (sigma about 0.1) and a smooth activation (softplus/GELU). Tune the "
        "penalty weight so clean accuracy stays above the floor."),
    "t16_synthetic_data": (
        "Dataset distillation by distribution matching (DM, Zhao & Bilen 2023). Initialize 100 synthetic "
        "images per class from augmented copies of the 10 real images, then optimize the synthetic pixels "
        "so that the mean embeddings of synthetic and real images match per class under many randomly "
        "initialized ConvNet feature extractors, with the same differentiable augmentation (crop, flip, "
        "color) applied to both sides."),
    "t18_anytime": (
        "Multi-exit network with a very cheap first exit and self-distillation. Use a ResNet backbone with "
        "4-5 exit heads placed so per-image cost roughly doubles from one exit to the next, the first "
        "costing only about 2-3 MFLOP. Train all exits jointly with a weighted sum of cross-entropy losses "
        "plus distillation from the final exit's softened predictions into the earlier exits."),
    "t09_tokenizer_transfer": (
        "Embedding transfer by sub-token averaging (FVT). Initialize each new token's input embedding as the mean "
        "of the original GPT-2 embeddings of the GPT-2 tokens it decomposes into (re-tokenize the new token's "
        "string with the GPT-2 tokenizer), tie the LM head to it, then first train only the embedding/head for "
        "about 15% of the budget with the transformer body frozen, and finally fine-tune the whole model on the "
        "Japanese training data for the rest of the budget."),
    "t04_tiny_walker": (
        "Augmented Random Search (ARS V2-t, Mania et al. 2018). Use a 24-16-4 "
        "tanh MLP whose input is the observation normalized by running mean/std statistics (stored with the "
        "policy). Each iteration perturbs the flat weights along 48 Gaussian directions (+/-, noise 0.05, one "
        "training terrain per pair, rollouts spread over all cores), keeps the 16 best directions and steps "
        "along their reward differences scaled by the std of their returns (step 0.03), for the whole budget. "
        "During training replace the -100 fall penalty by 0 so the search is not dominated by falls."),
    "t02_self_reg": (
        "Test-time-consistent augmentation with multi-view averaging. Put the augmentation inside the model's "
        "forward: for each input, take K=8 views (random crop with 4-pixel reflect padding + horizontal flip) "
        "and return the mean of the logits over the views, so train-time and test-time processing are "
        "identical. Use a narrower ResNet with GroupNorm, label smoothing 0.1 and strong weight decay, train "
        "for most of the 15-minute budget, and keep an EMA of the weights as the final model."),
    "t19_data_ordering": (
        "Online batch selection by loss (Loshchilov & Hutter 2015). Keep a running per-example loss "
        "estimate from the training feedback; sample each batch with probability decaying exponentially "
        "in the loss rank (high-loss examples more often), mixed with uniform sampling so every example "
        "keeps being presented; anneal the selection pressure towards uniform in the final 30% of steps."),
    "t11_fewshot_seg": (
        "DeepLabv3+ decoder with backbone fine-tuning and strong augmentation. Build the backbone with a dilated "
        "last stage (output stride 16), add an ASPP module (rates 6/12/18 plus image pooling, 256 channels) and a "
        "DeepLabv3+ decoder that fuses the ASPP output with a 48-channel projection of the layer1 features. "
        "Fine-tune the whole backbone at 1/10 of the head learning rate (backbone BatchNorm statistics frozen), "
        "with SGD-nesterov and a warmup + poly schedule over the full budget, batch 16, bf16 autocast, and strong "
        "GPU augmentation (random scale 0.5-2.0, random crop/translation, horizontal flip, colour jitter). "
        "Average the logits with a horizontally flipped copy at test time."),
}


def make(task_id: str) -> Path:
    run_id = f"anchor_ref_{task_id}"
    run_dir = Path(C.runs_root() if hasattr(C, "runs_root") else "/usr/xtmp/kg349/ar/runs") / run_id
    cfg = C.from_dict({"run_id": run_id, "task_id": task_id, "arm": "optimized", "seed": 0})
    jn = J.Journal.init(run_dir, cfg)
    now = time.time()
    with jn.transaction("anchor", "create") as st:
        iid = J.new_idea_id(st)
        st["ideas"][iid] = {
            "id": iid, "created_at": now, "wave": 0, "arm": "anchor", "prompt": {"config": None},
            "description": REF_IDEAS[task_id], "summary": REF_IDEAS[task_id][:200], "tags": [],
            "surrogate_scores": [], "s_cal": None, "F": None, "F_detail": None, "scored_seq": None,
            "real_score": None, "real": None, "status": "promoted", "times": {"promoted": now},
            "claim": None, "error": None, "retries": {},
        }
    print(run_id, iid, run_dir)
    return run_dir


if __name__ == "__main__":
    for t in (sys.argv[1:] or list(REF_IDEAS)):
        make(t)
