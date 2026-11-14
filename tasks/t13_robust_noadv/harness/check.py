"""Static scan of solution code for the 'no adversarial examples during training' rule
and for test-data access. Heuristic: hard violations invalidate; everything else is
recorded as review flags for the human/Claude reviewer.

usage: python check.py <dir> [<dir> ...]   -> JSON on stdout
"""
import json, re, sys
from pathlib import Path

HARD = [
    (r"^\s*(import|from)\s+(torchattacks|foolbox|advertorch|autoattack|cleverhans|robustbench|art\.attacks|art\.estimators)\b",
     "attack/robust-model library import"),
    (r"\bimport\s+art\b", "adversarial-robustness-toolbox import"),
    (r"test_batch|cifar-10-batches-py|cifar-100-python|/private/|private/t13|test\.npz",
     "reference to raw CIFAR / test data (solution may only read the --train file)"),
]
REVIEW = [
    (r"autograd\.grad\s*\(", "torch.autograd.grad (input gradients? fine for e.g. gradient-norm regularisation, NOT for building perturbed inputs)"),
    (r"requires_grad(_\s*\(|\s*=\s*True)", "requires_grad on a tensor (inputs?)"),
    (r"\.sign\s*\(", "sign() (FGSM/PGD-style step?)"),
    (r"(?i)\b(fgsm|pgd|bim|deepfool|carlini|cw_loss|adversar\w*|attack\w*|perturb\w*)\b", "attack-related identifier"),
    (r"\b(torch\.func|functorch|jacrev|jacfwd|vjp|jvp|hessian)\b", "functional autodiff (input Jacobians?)"),
    (r"\bx\w*\.grad\b|\bimg\w*\.grad\b|\binp\w*\.grad\b|\bdata\.grad\b", "gradient attribute of an input-like tensor"),
    (r"(?i)torch\.hub|pretrained\s*=\s*True|weights\s*=\s*['\"A-Z]|urlopen|requests\.|wget|load_state_dict_from_url",
     "external weights / network access"),
]
# applied only to model files (anything named model*.py, or inside the artifact dir): forward-pass masking risks
REVIEW_MODEL = [
    (r"no_grad|\.detach\(\)|inference_mode|torch\.round|\.round\(|floor\(|quantiz|jpeg|argmax", "possible non-differentiable op (check model.forward for gradient masking)"),
    (r"(?i)\brand(n|int|_like|n_like|perm)?\b|dropout|bernoulli", "randomness (check model.forward is deterministic in eval mode)"),
]


def scan(dirs):
    dirs = list(dirs)
    hard, review, files = [], [], []
    for d in dirs:
        d = Path(d)
        if not d.exists():
            continue
        for f in sorted(list(d.rglob("*.py")) + list(d.rglob("*.sh"))):
            if "__pycache__" in f.parts:
                continue
            files.append(str(f))
            try:
                lines = f.read_text(errors="replace").splitlines()
            except OSError:
                continue
            is_model = f.name.startswith("model") or (len(dirs) > 1 and Path(dirs[-1]) in f.parents)
            pats = REVIEW + (REVIEW_MODEL if is_model else [])
            for n, line in enumerate(lines, 1):
                if line.lstrip().startswith("#"):
                    continue
                for pat, why in HARD:
                    if re.search(pat, line):
                        hard.append({"file": str(f), "line": n, "why": why, "text": line.strip()[:200]})
                for pat, why in pats:
                    if re.search(pat, line):
                        review.append({"file": str(f), "line": n, "why": why, "text": line.strip()[:200]})
    return {"files": files, "hard_violations": hard, "review_flags": review[:60],
            "n_review_flags": len(review), "hard_ok": not hard}


if __name__ == "__main__":
    print(json.dumps(scan(sys.argv[1:]), indent=1))
