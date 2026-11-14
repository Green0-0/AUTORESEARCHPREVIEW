"""Shared harness code for t02_self_reg: mode table, data loading, solution import, the training-input
guard, and the post-training model checks (train/eval equivalence, determinism, batch independence).
Depends only on torch/numpy and env vars (AR_DATA, AR_ROOT, AR_SEED)."""
import copy
import importlib.util
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

TASK = "t02_self_reg"
NUM_CLASSES = 100
# mode -> (train() budget s, n train images, n test images, eval batch, check batch)
MODES = {
    "full":      dict(train_budget_s=900.0, n_train=500, n_test=10000, eval_bs=500, check_bs=256),
    "smoke":     dict(train_budget_s=60.0,  n_train=500, n_test=10000, eval_bs=500, check_bs=256),
    "cpu-smoke": dict(train_budget_s=60.0,  n_train=100, n_test=500,   eval_bs=100, check_bs=64),
}
GRACE_S = 10.0
EVAL_SEED = 20020          # global RNG seed set once before the test pass (independent of AR_SEED)
CHECK_SEED = 20021         # seed used for the train/eval-equivalence and batch-independence checks
REL_TOL = 1e-3             # relative max-abs logit difference tolerated by the functional checks
FP_TOL = 1e-4              # fingerprint distance below which a training input row counts as a raw image


def data_dir() -> Path:
    return Path(os.environ.get("AR_DATA", "/usr/xtmp/kg349/ar/data")) / TASK


def private_dir() -> Path:
    root = os.environ.get("AR_ROOT") or str(Path(os.environ.get("AR_DATA", "/usr/xtmp/kg349/ar/data")).parent)
    return Path(root) / "private" / TASK


def to_float(x_u8: np.ndarray, device) -> torch.Tensor:
    return torch.from_numpy(x_u8).to(device).float().div_(255.0).contiguous()


def load_train(device, n=None):
    """The fixed 500-image subset (5/class). cpu-smoke uses the first image of every class (n=100)."""
    d = data_dir()
    x = np.load(d / "train_x.npy"); y = np.load(d / "train_y.npy")
    if n is not None and n < len(y):
        per = n // NUM_CLASSES
        keep = np.concatenate([np.flatnonzero(y == c)[:per] for c in range(NUM_CLASSES)])
        x, y = x[keep], y[keep]
    return to_float(x, device), torch.from_numpy(y).to(device)


def load_test(device, n=None):
    d = private_dir()
    x = np.load(d / "test_x.npy"); y = np.load(d / "test_y.npy")
    if n is not None:
        x, y = x[:n], y[:n]
    return to_float(x, device), torch.from_numpy(y).to(device)


def import_from(dirpath: Path, name: str):
    dirpath = Path(dirpath)
    if str(dirpath) not in sys.path:
        sys.path.insert(0, str(dirpath))
    spec = importlib.util.spec_from_file_location(f"sol_{name}", dirpath / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def seed_all(s: int):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)   # torch.manual_seed also seeds every CUDA device


# --------------------------------------------------------------------------- training-input guard
class InputGuard:
    """Watches every gradient-enabled forward during the solution's train() call.

    - Root forward pre-hook on `model`: each grad-enabled call must be `model(x)` with exactly one float32
      (B,3,32,32) tensor and no kwargs, and every row of x must be one of the raw training images
      (checked by a float64 random-projection fingerprint against the 500 raw images; asynchronous,
      no host sync per step).
    - Global module pre-hook: a grad-enabled call of a submodule of `model` while `model.forward` is not
      active is a bypass (e.g. `model.body(augment(x))`). A grad-enabled call of a parametrized module that
      is not part of `model` is recorded as a review flag ("foreign network", e.g. a separate teacher).
    """

    def __init__(self, model: torch.nn.Module, x_raw: torch.Tensor):
        self.root = model
        self.ids = {id(m) for m in model.modules()}
        g = torch.Generator().manual_seed(4242)
        self.W = torch.randn(x_raw[0].numel(), 4, generator=g, dtype=torch.float64).to(x_raw.device)
        self.fp = x_raw.double().flatten(1) @ self.W
        self.dev = x_raw.device
        self.active = 0
        self.n_bad_rows = torch.zeros((), dtype=torch.int64, device=self.dev)
        self.n_rows = torch.zeros((), dtype=torch.int64, device=self.dev)
        self.seen = torch.zeros(len(x_raw), dtype=torch.bool, device=self.dev)
        self.n_grad_calls = 0
        self.n_bad_sig = 0
        self.n_bypass = 0
        self.n_foreign = 0
        self.examples = []
        self._h = []

    def _note(self, msg):
        if len(self.examples) < 10:
            self.examples.append(msg)

    def _root_pre(self, module, args, kwargs):
        self.active += 1
        if not torch.is_grad_enabled():
            return None
        self.n_grad_calls += 1
        x = args[0] if len(args) == 1 else None
        if (kwargs or x is None or not torch.is_tensor(x) or x.dim() != 4 or tuple(x.shape[1:]) != (3, 32, 32)
                or x.dtype != torch.float32):
            self.n_bad_sig += 1
            desc = (f"args={[tuple(a.shape) if torch.is_tensor(a) else type(a).__name__ for a in args]}, "
                    f"kwargs={sorted(kwargs)}, dtype={getattr(x, 'dtype', None)}")
            self._note(f"model() called with a non-raw signature: {desc}")
            return None
        with torch.no_grad():
            fp = x.detach().double().flatten(1) @ self.W
            d = torch.cdist(fp, self.fp)
            dmin, arg = d.min(1)
            ok = dmin <= FP_TOL
            self.n_bad_rows += (~ok).sum()
            self.n_rows += x.shape[0]
            self.seen[arg[ok]] = True
        return None

    def _root_post(self, module, args, out):
        self.active -= 1
        return None

    def _global_pre(self, module, args):
        if self.active > 0 or module is self.root or not torch.is_grad_enabled():
            return None
        if id(module) in self.ids:
            self.n_bypass += 1
            self._note(f"submodule {type(module).__name__} called with grad outside model.forward")
        elif any(p.requires_grad for p in module.parameters(recurse=False)):
            self.n_foreign += 1
        return None

    def install(self):
        self._h.append(self.root.register_forward_pre_hook(self._root_pre, with_kwargs=True))
        self._h.append(self.root.register_forward_hook(self._root_post))
        self._h.append(torch.nn.modules.module.register_module_forward_pre_hook(self._global_pre))
        return self

    def remove(self):
        for h in self._h:
            h.remove()
        self._h = []

    def report(self):
        n_bad, n_rows = int(self.n_bad_rows), int(self.n_rows)
        rep = {"grad_forward_calls": self.n_grad_calls, "rows_checked": n_rows, "non_raw_rows": n_bad,
               "bad_signature_calls": self.n_bad_sig, "bypass_calls": self.n_bypass,
               "foreign_module_grad_calls": self.n_foreign,
               "train_images_seen": int(self.seen.sum()), "n_train": int(len(self.seen)),
               "examples": self.examples}
        errs = []
        if self.n_grad_calls == 0:
            errs.append("no gradient-enabled model(x) call observed: the loss must be computed on model(x_raw)")
        if n_bad:
            errs.append(f"{n_bad}/{n_rows} training inputs to model() were not raw training images "
                        "(augmentation/mixing/noise outside model.forward)")
        if self.n_bad_sig:
            errs.append(f"{self.n_bad_sig} grad-enabled model() calls with extra args/kwargs or non-float32 input")
        if self.n_bypass:
            errs.append(f"{self.n_bypass} grad-enabled submodule calls outside model.forward")
        rep["errors"] = errs
        rep["ok"] = not errs
        rep["review_flags"] = (["grad-enabled forward of a network that is not part of model "
                                f"({self.n_foreign} calls): teacher/auxiliary network? reviewer must check"]
                               if self.n_foreign else [])
        return rep


# --------------------------------------------------------------------------- post-training checks
def _rel(a, b):
    a, b = a.float(), b.float()
    return float((a - b).abs().max() / (b.abs().max() + 1e-12))


def _fwd(model, x, seed, train_mode, grad):
    model.train(train_mode)
    seed_all(seed)
    with torch.set_grad_enabled(grad):
        z = model(x)
    return z.detach().float()


def model_checks(model, x_check, x_other):
    """(a) train vs eval equivalence under the same seed, (c) determinism, stochasticity, batch independence,
    logits shape/finiteness. x_other: images not in x_check (used to replace half of the batch).
    Runs on a deepcopy so train-mode side effects (e.g. BN running stats) cannot leak into the scored
    evaluation; the train-mode pass runs last for the same reason. Returns (report, errors)."""
    m = copy.deepcopy(model)
    for p in m.parameters():
        p.requires_grad_(True)
    B = len(x_check); h = B // 2
    rep, errs = {}, []
    z_ev = _fwd(m, x_check, CHECK_SEED, False, False)
    rep["logits_shape"] = list(z_ev.shape)
    rep["logits_shape_ok"] = list(z_ev.shape) == [B, NUM_CLASSES]
    if not rep["logits_shape_ok"]:
        return rep, [f"model must return logits (N,{NUM_CLASSES}), got {list(z_ev.shape)}"]
    rep["logits_finite"] = bool(torch.isfinite(z_ev).all())
    if not rep["logits_finite"]:
        errs.append("non-finite logits")
    # (c) determinism under a fixed seed
    z_ev2 = _fwd(m, x_check, CHECK_SEED, False, False)
    rep["determinism_rel_diff"] = _rel(z_ev2, z_ev)
    if rep["determinism_rel_diff"] > REL_TOL:
        errs.append("forward is not reproducible under a fixed RNG seed")
    # stochasticity (informational): a different seed
    z_s = _fwd(m, x_check, CHECK_SEED + 1, False, False)
    rep["seed_change_rel_diff"] = _rel(z_s, z_ev)
    rep["forward_stochastic"] = rep["seed_change_rel_diff"] > 1e-6
    # batch independence: same seed and batch size, second half replaced by *different* images
    x_b = torch.cat([x_check[:h], x_other[:B - h]])
    z_b = _fwd(m, x_b, CHECK_SEED, False, False)
    rep["batch_independence_rel_diff"] = _rel(z_b[:h], z_ev[:h])
    if rep["batch_independence_rel_diff"] > REL_TOL:
        errs.append(f"outputs depend on other examples in the batch (rel {rep['batch_independence_rel_diff']:.3g}): "
                    "batch statistics or cross-example mixing in forward")
    # (a) train() + grad vs eval() + no_grad, same RNG seed (last: train mode may mutate buffers)
    z_tr = _fwd(m, x_check, CHECK_SEED, True, True)
    rep["train_eval_rel_diff"] = _rel(z_tr, z_ev)
    if rep["train_eval_rel_diff"] > REL_TOL:
        errs.append(f"train-mode and eval-mode outputs differ (rel {rep['train_eval_rel_diff']:.3g} > {REL_TOL}): "
                    "training-only behaviour (dropout, batch-stat BatchNorm, self.training branches, ...)")
    del m
    return rep, errs
