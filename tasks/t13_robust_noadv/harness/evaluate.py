"""Harness evaluation for t13_robust_noadv. Loads <artifact>/model.py:build_model() + weights.pt,
computes clean accuracy on the CIFAR-10 test set and white-box L-inf PGD-20 robust accuracy
(eps=4/255, step 1/255, uniform random start, fixed seed) on the first N_PGD test images
(eps=8/255 reported as a secondary number),
plus gradient-masking diagnostics. Writes <out>/metric.json."""
import argparse, importlib.util, json, math, os, sys, time, traceback
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

CLEAN_FLOOR = 0.85            # full mode only; see README for how it was chosen
EPS, STEPS = 4 / 255, 20          # scored eps (changed from 8/255 on 2026-11-08: 8/255 compressed honest
EPS_SECONDARY = 8 / 255           # methods into 0-3%; see README). Step size is eps/4 for every eps.
ATTACK_SEED = 13013
SIZES = {"full": (10000, 2000), "smoke": (2000, 256), "cpu-smoke": (128, 16)}  # (n_clean, n_pgd)
BS = {"full": 500, "smoke": 256, "cpu-smoke": 16}


def load_model(art: Path, dev):
    spec = importlib.util.spec_from_file_location("solution_model", art / "model.py")
    sys.path.insert(0, str(art))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    model = mod.build_model()
    sd = torch.load(art / "weights.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(sd, strict=True)
    model = model.float().to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def margin_loss(z, y):
    zy = z.gather(1, y[:, None]).squeeze(1)
    other = z.clone()
    other.scatter_(1, y[:, None], -float("inf"))
    return (other.max(1).values - zy).sum()


def pgd(model, x, y, loss_name, eps, alpha, steps, seed):
    """Returns (robust_any_iterate, robust_final, zero_grad_frac, nonfinite) as bool tensors / floats.
    A sample counts as robust only if correctly classified at every iterate (incl. the random start
    and the final point)."""
    g = torch.Generator(device=x.device); g.manual_seed(seed)
    delta = (torch.rand(x.shape, generator=g, device=x.device) * 2 - 1) * eps
    xa = (x + delta).clamp(0, 1)
    ok = torch.ones(len(y), dtype=torch.bool, device=x.device)
    zero_frac, nonfinite = 0.0, False
    for s in range(steps):
        xa = xa.detach().requires_grad_(True)
        z = model(xa)
        if z.dtype != torch.float32:
            z = z.float()
        ok &= z.argmax(1) == y
        loss = F.cross_entropy(z, y, reduction="sum") if loss_name == "ce" else margin_loss(z, y)
        (grad,) = torch.autograd.grad(loss, xa)
        if not torch.isfinite(grad).all():
            nonfinite = True
            grad = torch.nan_to_num(grad, 0.0, 0.0, 0.0)
        if s == 0:
            zero_frac = float((grad.flatten(1).abs().amax(1) == 0).float().mean())
        xa = torch.min(torch.max(xa.detach() + alpha * grad.sign(), x - eps), x + eps).clamp(0, 1)
    with torch.no_grad():
        fin = model(xa).float().argmax(1) == y
    return ok & fin, fin, zero_frac, nonfinite


def batched_pgd(model, X, Y, loss_name, eps, steps, bs, seed):
    rob, fin, zf, nf = [], [], [], False
    for i in range(0, len(Y), bs):
        r, f, z, n = pgd(model, X[i:i + bs], Y[i:i + bs], loss_name, eps, eps / 4, steps, seed + i)
        rob.append(r); fin.append(f); zf.append(z * len(r)); nf |= n
    return torch.cat(rob), torch.cat(fin), sum(zf) / len(Y), nf


@torch.no_grad()
def predict(model, X, bs):
    return torch.cat([model(X[i:i + bs]).float().argmax(1) for i in range(0, len(X), bs)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", default="full")
    ap.add_argument("--train-rc", type=int, default=0)
    ap.add_argument("--train-wall", type=float, default=0.0)
    ap.add_argument("--budget-s", type=float, default=1200)
    ap.add_argument("--scan", default=None)
    ap.add_argument("--t-start", type=float, default=None)
    a = ap.parse_args()
    t0 = time.time()
    out = Path(a.out); art = Path(a.artifact)
    rep = {"mode": a.mode, "train_rc": a.train_rc, "train_wall_s": round(a.train_wall, 1), "budget_s": a.budget_s,
           "attack": {"norm": "Linf", "eps": "4/255", "step": "1/255", "secondary_eps": "8/255", "steps": STEPS, "random_start": True,
                      "seed": ATTACK_SEED, "losses": ["ce", "margin"], "criterion": "correct at every iterate"},
           "errors": []}
    valid = True
    value = 0.0
    scan = json.loads(Path(a.scan).read_text()) if a.scan and Path(a.scan).exists() else None
    rep["scan"] = scan
    if scan is not None and not scan.get("hard_ok", True):
        valid = False; rep["errors"].append("static scan: hard violation (see scan.hard_violations)")
    if a.train_rc != 0:
        valid = False
        rep["errors"].append("training exceeded budget (timeout)" if a.train_rc in (124, 137) else f"training exited rc={a.train_rc}")
    try:
        dev = torch.device("cpu" if a.mode == "cpu-smoke" else "cuda")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if a.mode == "cpu-smoke":
            torch.set_num_threads(min(4, os.cpu_count() or 1))
        root = Path(os.environ.get("AR_ROOT") or Path(os.environ.get("AR_DATA", "/usr/xtmp/kg349/ar/data")).parent)
        d = np.load(root / "private" / "t13_robust_noadv" / "test.npz")
        n_clean, n_pgd = SIZES[a.mode]
        bs = BS[a.mode]
        X = torch.from_numpy(d["x"][:n_clean]).permute(0, 3, 1, 2).float().div(255).contiguous().to(dev)
        Y = torch.from_numpy(d["y"][:n_clean]).to(dev)
        model = load_model(art, dev)
        n_params = sum(p.numel() for p in model.parameters())
        rep["n_params"] = n_params
        if model.training:
            raise RuntimeError("model.eval() did not put the model in eval mode")

        # determinism (randomised defences are not allowed: PGD without EOT would be meaningless)
        with torch.no_grad():
            z1 = model(X[:bs]).float(); z2 = model(X[:bs]).float()
        rep["logits_shape_ok"] = list(z1.shape) == [min(bs, n_clean), 10]
        if not rep["logits_shape_ok"]:
            raise RuntimeError(f"model must return logits of shape (N,10), got {list(z1.shape)}")
        det_diff = float((z1 - z2).abs().max() / (z1.abs().max() + 1e-12))
        rep["determinism_rel_diff"] = det_diff
        if not torch.isfinite(z1).all():
            raise RuntimeError("non-finite logits")
        if det_diff > 1e-3:
            valid = False; rep["errors"].append("model forward is not deterministic in eval mode")

        tc = time.time()
        pred = predict(model, X, bs)
        clean_correct = pred == Y
        clean = float(clean_correct.float().mean())
        rep["clean_acc"] = clean; rep["n_clean"] = n_clean; rep["clean_eval_s"] = round(time.time() - tc, 1)

        Xp, Yp, cp = X[:n_pgd], Y[:n_pgd], clean_correct[:n_pgd]
        tp = time.time()
        rce, fce, zce, nfce = batched_pgd(model, Xp, Yp, "ce", EPS, STEPS, bs, ATTACK_SEED)
        rmg, fmg, zmg, nfmg = batched_pgd(model, Xp, Yp, "margin", EPS, STEPS, bs, ATTACK_SEED + 7)
        rob = cp & rce & rmg
        value = float(rob.float().mean())
        rep.update({
            "n_pgd": n_pgd, "pgd_eval_s": round(time.time() - tp, 1),
            "clean_acc_pgd_subset": float(cp.float().mean()),
            "robust_acc": value,
            "robust_acc_ce": float((cp & rce).float().mean()),
            "robust_acc_margin": float((cp & rmg).float().mean()),
            "robust_acc_ce_final_iterate_only": float((cp & fce).float().mean()),
        })
        r8ce, _, _, _ = batched_pgd(model, Xp, Yp, "ce", EPS_SECONDARY, STEPS, bs, ATTACK_SEED)
        r8mg, _, _, _ = batched_pgd(model, Xp, Yp, "margin", EPS_SECONDARY, STEPS, bs, ATTACK_SEED + 7)
        rep["robust_acc_eps8_secondary"] = float((cp & r8ce & r8mg).float().mean())   # not scored

        # gradient-masking diagnostics (flags for the reviewer; they do not invalidate by themselves)
        n_diag = min(500, n_pgd)
        with torch.no_grad():
            g = torch.Generator(device=dev); g.manual_seed(ATTACK_SEED + 99)
            noise = (torch.rand(Xp.shape, generator=g, device=dev) * 2 - 1) * EPS
            noise_acc = float((cp & (predict(model, (Xp + noise).clamp(0, 1), bs) == Yp)).float().mean())
        r16, _, _, _ = batched_pgd(model, Xp[:n_diag], Yp[:n_diag], "margin", 4 * EPS, STEPS, bs, ATTACK_SEED + 11)
        r16 = float((cp[:n_diag] & r16).float().mean())
        r8_sub = float(rob[:n_diag].float().mean())
        diag = {"zero_input_grad_frac_ce": zce, "zero_input_grad_frac_margin": zmg,
                "nonfinite_grad": bool(nfce or nfmg), "uniform_noise_acc_eps4": noise_acc,
                "robust_acc_margin_eps16_first%d" % n_diag: r16, "robust_acc_eps4_first%d" % n_diag: r8_sub}
        flags = []
        if max(zce, zmg) > 0.02: flags.append("many exactly-zero input gradients")
        if nfce or nfmg: flags.append("non-finite input gradients")
        if rep["robust_acc_ce"] - rep["robust_acc_margin"] > 0.05: flags.append("CE attack much weaker than margin attack (loss saturation?)")
        if rep["robust_acc_ce_final_iterate_only"] - rep["robust_acc_ce"] > 0.05: flags.append("large gap final-iterate vs any-iterate (noisy/shattered gradients?)")
        if r16 > 0.5 * max(r8_sub, 1e-9) and r16 > 0.05: flags.append("eps=16/255 attack barely beats eps=4/255 (attack not converging?)")
        if value > 0.6: flags.append("robust accuracy implausibly high for no adversarial training")
        diag["flags"] = flags
        rep["masking_diagnostics"] = diag

        floor_applied = a.mode == "full"
        rep["clean_floor"] = CLEAN_FLOOR; rep["clean_floor_applied"] = floor_applied
        rep["clean_floor_ok"] = clean >= CLEAN_FLOOR
        if floor_applied and clean < CLEAN_FLOOR:
            valid = False; rep["errors"].append(f"clean accuracy {clean:.4f} below floor {CLEAN_FLOOR}")
    except Exception as e:
        valid = False
        rep["errors"].append(f"eval failed: {type(e).__name__}: {e}")
        rep["traceback"] = traceback.format_exc()[-3000:]
    rep["eval_wall_s"] = round(time.time() - t0, 1)
    wall = time.time() - a.t_start if a.t_start else a.train_wall + rep["eval_wall_s"]
    m = {"metric": "pgd20_robust_acc", "value": value, "valid": bool(valid), "constraint_report": rep,
         "wall_s": round(wall, 1), "mode": a.mode}
    (out / "metric.json").write_text(json.dumps(m, indent=1))
    print(json.dumps({k: m[k] for k in ("metric", "value", "valid", "wall_s", "mode")}))
    print(json.dumps({k: rep.get(k) for k in ("clean_acc", "robust_acc", "robust_acc_ce", "robust_acc_margin", "errors")}))


if __name__ == "__main__":
    main()
