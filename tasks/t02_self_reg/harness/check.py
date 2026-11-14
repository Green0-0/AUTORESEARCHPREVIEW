"""Static scan of the solution for t02_self_reg (no training-only regularization; train data only).
Heuristic and advisory: the runtime guard (train_wrapper.py) and the functional checks (evaluate.py) are the
real enforcement. Hard violations invalidate; everything else is a review flag for the reviewer.

  train-side files (train.py and any non-model*.py helper): augmentation / noise / mixing / regularization
      ops outside the model, functional re-implementations of the forward, file reads.
  model files (model*.py and any non-train helper): train/eval- or grad-dependent branches, dropout/BN
      (the runtime checks decide), randomness in forward (allowed, it is also applied at test).

usage: python check.py <solution_dir>   -> JSON on stdout
"""
import ast, json, re, sys
from pathlib import Path

HARD = [
    (r"cifar-100-python|cifar-10-batches|/private/|\btest_[xy]\b|\bAR_ROOT\b|\bAR_DATA\b",
     "reference to raw CIFAR / test / private data (the solution only gets x, y via train())"),
    (r"torchvision\.datasets|torch\.hub|load_state_dict_from_url|pretrained\s*=\s*True|from_pretrained|"
     r"\bweights\s*=\s*['\"A-Z]|urlopen|\brequests\.|\bwget\b|huggingface_hub|\btimm\b",
     "external data / pretrained weights / network access"),
    (r"_forward_pre_hooks|_forward_hooks|_global_forward|register_module_forward|_global_module_hooks",
     "touching module hook registries (the harness guard)"),
]
HARD_TRAIN = [
    (r"^\s*(import|from)\s+(kornia|albumentations|torchvision\.transforms|augly)\b|\btransforms\.(Random|AutoAugment|RandAugment|TrivialAugment|AugMix|ColorJitter)",
     "augmentation library used in the training script (augmentation must live in model.forward)"),
]
REVIEW_TRAIN = [
    (r"\.flip\(|torch\.flip|torch\.roll|\.roll\(|F\.pad\(|\binterpolate\(|grid_sample|affine_grid|\bcrop\w*|"
     r"\bresize\w*|\brot90\b", "augmentation-like tensor op outside the model"),
    (r"(?i)\b(mixup|cutmix|cutout|manifold_mix\w*|random_erasing|augment\w*|jitter\w*|noise\w*|mix_\w+|lam\b|lambda_mix)\b",
     "augmentation / mixing identifier outside the model"),
    (r"\brand(n|_like|n_like|int)?\(|bernoulli|\.normal_\(|\.uniform_\(|\bBeta\(|beta\(|dirichlet|\bpoisson\(",
     "random tensor in the training script (data perturbation? randperm/index sampling is fine)"),
    (r"(?i)dropout|drop_path|droppath|stochastic_?depth|BatchNorm|batch_norm|shake",
     "training-only regularization layer referenced in the training script"),
    (r"\bF\.(conv\dd|linear|group_norm|layer_norm)\(|functional_call|torch\.func\b|\.forward\(",
     "functional forward / direct .forward() call outside model (bypassing model(x)?)"),
    (r"\bopen\(|np\.load|torch\.load|pickle|\bglob\b|os\.listdir", "file access (train.py gets its data from the harness)"),
    (r"copy\.deepcopy|\bteacher\b|\bema\b", "second network copy (EMA / teacher): fine if it only sees raw images"),
]
REVIEW_MODEL = [
    (r"self\.training|is_grad_enabled|inference_mode|is_inference|torch\.jit\.is_scripting|requires_grad",
     "forward may branch on train/eval or grad state (must behave identically at train and test)"),
    (r"(?i)dropout|drop_path|droppath|stochastic_?depth|BatchNorm|batch_norm|InstanceNorm|track_running_stats|momentum\s*=",
     "dropout / BN-like layer (allowed only if it behaves identically in train and eval; runtime check decides)"),
    (r"\brand(n|_like|n_like|int|perm)?\(|bernoulli|\.normal_\(|\.uniform_\(|Generator\(",
     "randomness in the model (allowed: in-forward augmentation is also applied at test)"),
    (r"\bglobal\b|\bstep_count\b|\bepoch\b|\bschedule\w*|\bcurriculum\b",
     "forward state that may change over training (augmentation strength must not depend on training progress at test)"),
]


def _doc_lines(src):
    """Line numbers covered by docstrings / bare string statements (skipped by the scan)."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return set()
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            out.update(range(node.lineno, (node.end_lineno or node.lineno) + 1))
    return out


def scan(root):
    root = Path(root)
    hard, train_flags, model_flags, files = [], [], [], []
    for f in sorted(list(root.rglob("*.py")) + list(root.rglob("*.sh"))):
        if "__pycache__" in f.parts:
            continue
        files.append(str(f.relative_to(root)))
        try:
            src = f.read_text(errors="replace")
        except OSError:
            continue
        lines = src.splitlines()
        doc = _doc_lines(src) if f.suffix == ".py" else set()
        is_train = f.name == "train.py" or (not f.name.startswith("model"))
        is_model = f.name.startswith("model") or f.name != "train.py"
        pats_h = HARD + (HARD_TRAIN if f.name == "train.py" else [])
        for n, line in enumerate(lines, 1):
            s = line.strip()
            if s.startswith("#") or n in doc:
                continue
            code = line.split("#", 1)[0] if "#" in line and "'" not in line and '"' not in line else line
            rec = {"file": str(f.relative_to(root)), "line": n, "text": s[:200]}
            for pat, why in pats_h:
                if re.search(pat, code):
                    hard.append({**rec, "why": why})
            if is_train:
                for pat, why in REVIEW_TRAIN:
                    if re.search(pat, code):
                        train_flags.append({**rec, "why": why})
            if is_model:
                for pat, why in REVIEW_MODEL:
                    if re.search(pat, code):
                        model_flags.append({**rec, "why": why})
    return {"files": files, "hard_violations": hard, "hard_ok": not hard,
            "train_flags": train_flags[:60], "n_train_flags": len(train_flags),
            "model_flags": model_flags[:60], "n_model_flags": len(model_flags)}


if __name__ == "__main__":
    print(json.dumps(scan(sys.argv[1]), indent=1))
