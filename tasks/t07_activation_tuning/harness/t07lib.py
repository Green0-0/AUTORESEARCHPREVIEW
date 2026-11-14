"""Fixed network skeleton, data loading and constraint checks for t07_activation_tuning.

The harness owns the architecture: input (3x32x32 -> 3072, fixed normalisation) ->
Linear(3072,W) -> f1 -> Linear(W,W) -> f2 -> Linear(W,W) -> f3 -> Linear(W,100) -> logits.
The solution only supplies f1, f2, f3 (nn.Modules mapping (B, W) -> (B, W)).
"""
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

TASK = "t07_activation_tuning"
WIDTH = 1024
IN_DIM = 3 * 32 * 32
NUM_CLASSES = 100
NUM_HIDDEN = 3
ACT_FRACTION_LIMIT = 0.20          # act_params <= 20% of (linear + act) params
TRAIN_BUDGET_S = {"full": 900.0, "smoke": 60.0, "cpu-smoke": 45.0}
CIFAR100_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR100_STD = (0.2673, 0.2564, 0.2762)


class MLPNet(nn.Module):
    """y = L4 f3(L3 f2(L2 f1(L1 x))). Input: float images in [0,1], shape (B,3,32,32)."""

    def __init__(self, acts):
        super().__init__()
        acts = list(acts)
        if len(acts) != NUM_HIDDEN:
            raise ValueError(f"build_activations must return {NUM_HIDDEN} modules, got {len(acts)}")
        dims = [IN_DIM] + [WIDTH] * NUM_HIDDEN + [NUM_CLASSES]
        self.linears = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(NUM_HIDDEN + 1)])
        self.acts = nn.ModuleList(acts)
        self.register_buffer("mean", torch.tensor(CIFAR100_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(CIFAR100_STD).view(1, 3, 1, 1), persistent=False)

    def forward(self, x):
        h = ((x - self.mean) / self.std).flatten(1)
        for lin, act in zip(self.linears[:-1], self.acts):
            h = act(lin(h))
        return self.linears[-1](h)


# ---------------------------------------------------------------- data
def data_dirs():
    ar_data = Path(os.environ["AR_DATA"])
    ar_root = Path(os.environ.get("AR_ROOT", str(ar_data.parent)))
    return ar_data / TASK, ar_root / "private" / TASK


def load_split(split, device, limit=None):
    pub, priv = data_dirs()
    d = pub if split == "train" else priv
    x = np.load(d / f"{split}_x.npy", mmap_mode="r")
    y = np.load(d / f"{split}_y.npy")
    if limit is not None:
        x, y = x[:limit], y[:limit]
    return (torch.from_numpy(np.array(x)).to(device),
            torch.from_numpy(np.array(y)).to(device))


# ---------------------------------------------------------------- solution loading
def import_from(code_dir, modname):
    code_dir = str(Path(code_dir).resolve())
    if code_dir not in sys.path:
        sys.path.insert(0, code_dir)
    path = Path(code_dir) / f"{modname}.py"
    if not path.exists():
        raise FileNotFoundError(f"solution file missing: {path}")
    spec = importlib.util.spec_from_file_location(f"solution_{modname}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_model(act_mod):
    acts = act_mod.build_activations(WIDTH)
    for i, a in enumerate(acts):
        if not isinstance(a, nn.Module):
            raise TypeError(f"activation {i} is not an nn.Module")
    return MLPNet(acts)


# ---------------------------------------------------------------- checks
def _numel_unique(tensors):
    seen, n = set(), 0
    for t in tensors:
        if id(t) in seen:
            continue
        seen.add(id(t))
        n += t.numel()
    return n


def param_report(model):
    """Activation 'parameters' = all parameters + all floating-point buffers in the activation modules."""
    lin_ids = {id(p) for p in model.linears.parameters()}
    per_layer = []
    for a in model.acts:
        ts = [p for p in a.parameters()] + [b for b in a.buffers() if b.is_floating_point()]
        per_layer.append(_numel_unique(ts))
    act_ts = [p for p in model.acts.parameters()] + [b for b in model.acts.buffers() if b.is_floating_point()]
    shares_linear = any(id(t) in lin_ids for t in act_ts)
    act = _numel_unique(act_ts)
    lin = _numel_unique(list(model.linears.parameters()))
    total = lin + act
    frac = act / total
    return {
        "linear_params": lin, "act_params": act, "act_params_per_layer": per_layer,
        "total_params": total, "act_fraction": round(frac, 6), "act_fraction_limit": ACT_FRACTION_LIMIT,
        "act_shares_linear_params": shares_linear,
        "params_ok": bool(frac <= ACT_FRACTION_LIMIT and not shares_linear),
    }


def structure_report(model):
    """The skeleton must be untouched: class, forward, Linear shapes, no hooks."""
    errs = []
    if type(model) is not MLPNet:
        errs.append(f"model class changed: {type(model)}")
    if "forward" in vars(model):
        errs.append("model.forward overridden on instance")
    dims = [IN_DIM] + [WIDTH] * NUM_HIDDEN + [NUM_CLASSES]
    if len(model.linears) != NUM_HIDDEN + 1 or len(model.acts) != NUM_HIDDEN:
        errs.append("number of layers changed")
    else:
        for i, lin in enumerate(model.linears):
            if type(lin) is not nn.Linear or "forward" in vars(lin):
                errs.append(f"linears[{i}] is not a plain nn.Linear")
            elif tuple(lin.weight.shape) != (dims[i + 1], dims[i]) or lin.bias is None:
                errs.append(f"linears[{i}] shape changed: {tuple(lin.weight.shape)}")
    names = {n for n, _ in model.named_children()}
    if names != {"linears", "acts"}:
        errs.append(f"top-level children changed: {sorted(names)}")
    for n, m in model.named_modules():
        if m._forward_hooks or m._forward_pre_hooks or getattr(m, "_backward_hooks", None):
            errs.append(f"hook registered on module '{n or 'model'}'")
    g = torch.nn.modules.module
    if g._global_forward_hooks or g._global_forward_pre_hooks:
        errs.append("global module hooks registered")
    return {"structure_ok": not errs, "structure_errors": errs}


@torch.no_grad()
def behaviour_report(model, xa, xb, tol=1e-3):
    """Eval-mode functional checks on the activations (model must be in eval mode, fp32).

    shape:  f_i maps (B, W) -> (B, W), finite.
    purity: f_i(z) depends only on z: re-applying f_i to the input it saw during a full forward pass,
            after an unrelated forward pass on other data, reproduces the same output.
    batch independence: per-sample outputs do not depend on the other samples in the batch.
    """
    errs = []
    cap = {}

    def mk(i):
        def hook(mod, inp, out):
            cap[i] = (inp[0].detach().clone(), out.detach().clone() if torch.is_tensor(out) else out)
        return hook

    hs = [a.register_forward_hook(mk(i)) for i, a in enumerate(model.acts)]
    try:
        logits_a = model(xa)
    finally:
        for h in hs:
            h.remove()
    for i in range(NUM_HIDDEN):
        if i not in cap:
            errs.append(f"activation {i} was not called in forward")
            continue
        z, out = cap[i]
        if not torch.is_tensor(out) or tuple(out.shape) != tuple(z.shape) or tuple(z.shape)[1:] != (WIDTH,):
            errs.append(f"activation {i}: output shape {getattr(out, 'shape', None)} != input shape {tuple(z.shape)}")
            continue
        if not torch.isfinite(out).all():
            errs.append(f"activation {i}: non-finite outputs")
    for i in reversed(range(NUM_HIDDEN)):
        if i not in cap or errs:
            continue
        z, out = cap[i]
        model(xb)  # unrelated forward right before; any hidden cross-call state is now different
        again = model.acts[i](z.clone())
        scale = max(1.0, out.abs().max().item())
        if (again - out).abs().max().item() > tol * scale:
            errs.append(f"activation {i}: output is not a function of its input Wx alone (purity check failed)")
    # batch independence: permuted batch and singletons
    perm = torch.randperm(xa.shape[0], generator=torch.Generator().manual_seed(0)).to(xa.device)
    lp = model(xa[perm])
    scale = max(1.0, logits_a.abs().max().item())
    batch_ok = (lp - logits_a[perm]).abs().max().item() <= tol * scale
    for j in range(4):
        l1 = model(xa[j:j + 1])
        batch_ok &= (l1 - logits_a[j:j + 1]).abs().max().item() <= tol * scale
    if not batch_ok:
        errs.append("eval-mode outputs depend on other samples in the batch")
    return {"behaviour_ok": not errs, "behaviour_errors": errs}
