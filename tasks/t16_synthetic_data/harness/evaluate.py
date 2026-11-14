"""Validate a synthetic set and train/evaluate the three fixed classifiers on it.

Only this step reads CIFAR-100 test data. It never runs solution code.
"""
import importlib.util
import os
import pickle
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from archs import VISIBLE

TASK_ID = "t16_synthetic_data"
NUM_CLASSES = 100
MAX_PER_CLASS = 100
MEAN = torch.tensor([0.5071, 0.4865, 0.4409]).view(1, 3, 1, 1)
STD = torch.tensor([0.2673, 0.2564, 0.2762]).view(1, 3, 1, 1)
ALLOWED_X_DTYPES = (np.uint8, np.float16, np.float32)


class EvalError(RuntimeError):
    pass


# ------------------------------------------------------------------ validation
def validate(npz_path):
    """Returns (ok, report, x_uint8_or_float01 np.ndarray | None, y np.ndarray | None)."""
    rep = {"checks": {}, "errors": []}

    def fail(key, msg):
        rep["checks"][key] = False
        rep["errors"].append(msg)

    p = Path(npz_path)
    if not p.is_file():
        fail("artifact_exists", f"missing {p}")
        return False, rep, None, None
    rep["checks"]["artifact_exists"] = True
    rep["storage_bytes_npz"] = p.stat().st_size
    try:
        d = np.load(p, allow_pickle=False)
        keys = sorted(d.files)
    except Exception as e:  # noqa: BLE001
        fail("loadable", f"np.load failed: {e!r}")
        return False, rep, None, None
    rep["checks"]["loadable"] = True
    if keys != ["x", "y"]:
        fail("keys", f"npz keys must be exactly ['x','y'], got {keys}")
        return False, rep, None, None
    rep["checks"]["keys"] = True
    x, y = d["x"], d["y"]
    rep["x_shape"], rep["x_dtype"], rep["y_dtype"] = list(x.shape), str(x.dtype), str(y.dtype)
    rep["storage_bytes_raw"] = int(x.nbytes + y.nbytes)

    ok_shape = x.ndim == 4 and x.shape[1:] == (32, 32, 3) and x.shape[0] >= 1 and y.shape == (x.shape[0],)
    if not ok_shape:
        fail("shape", f"need x (N,32,32,3) and y (N,), got {x.shape} / {y.shape}")
    else:
        rep["checks"]["shape"] = True
    if x.dtype not in ALLOWED_X_DTYPES:
        fail("x_dtype", f"x dtype must be uint8, float16 or float32, got {x.dtype}")
    else:
        rep["checks"]["x_dtype"] = True
        if x.dtype != np.uint8:
            xf = x.astype(np.float32)
            if not np.all(np.isfinite(xf)):
                fail("x_range", "x has non-finite values")
            elif xf.size and (xf.min() < 0.0 or xf.max() > 1.0):
                fail("x_range", f"float x must lie in [0,1], got [{xf.min():.4g},{xf.max():.4g}]")
            else:
                rep["checks"]["x_range"] = True
        else:
            rep["checks"]["x_range"] = True
    if not np.issubdtype(y.dtype, np.integer):
        fail("y_dtype", f"y must be an integer array (hard labels), got {y.dtype}")
    else:
        rep["checks"]["y_dtype"] = True
        if y.size and (y.min() < 0 or y.max() >= NUM_CLASSES):
            fail("y_range", f"labels must be in [0,{NUM_CLASSES - 1}], got [{y.min()},{y.max()}]")
        else:
            rep["checks"]["y_range"] = True
            counts = np.bincount(y.astype(np.int64), minlength=NUM_CLASSES)
            rep["n_total"] = int(y.size)
            rep["per_class_max"] = int(counts.max())
            rep["per_class_min"] = int(counts.min())
            rep["n_classes_present"] = int((counts > 0).sum())
            if counts.max() > MAX_PER_CLASS:
                fail("per_class_limit", f"at most {MAX_PER_CLASS} examples per class; class "
                     f"{int(counts.argmax())} has {int(counts.max())}")
            else:
                rep["checks"]["per_class_limit"] = True
    ok = all(rep["checks"].values())
    return ok, rep, (x if ok else None), (y if ok else None)


# ------------------------------------------------------------------ data
def load_test(ar_data, limit=None):
    with open(Path(ar_data) / "cifar-100-python" / "test", "rb") as f:
        d = pickle.load(f, encoding="latin1")
    x = d["data"].reshape(-1, 3, 32, 32)
    y = np.asarray(d["fine_labels"], dtype=np.int64)
    if limit:
        x, y = x[:limit], y[:limit]
    return x, y


def to_tensor_nchw01(x):
    """x: (N,32,32,3) uint8 or float in [0,1] -> float32 NCHW in [0,1]."""
    t = torch.from_numpy(np.ascontiguousarray(x))
    t = t.float() / 255.0 if x.dtype == np.uint8 else t.float()
    return t.permute(0, 3, 1, 2).contiguous()


# ------------------------------------------------------------------ withheld arch
def private_dir():
    if os.environ.get("AR_PRIVATE_DIR"):
        return Path(os.environ["AR_PRIVATE_DIR"])
    root = os.environ.get("AR_ROOT") or str(Path(os.environ.get("AR_DATA", "/usr/xtmp/kg349/ar/data")).parent)
    return Path(root) / "private" / TASK_ID


def load_withheld():
    p = private_dir() / "withheld_arch.py"
    if not p.is_file():
        raise EvalError(f"withheld architecture file not found: {p}. Run tasks/{TASK_ID}/prepare_data.py "
                        f"(or set AR_PRIVATE_DIR). Refusing to evaluate on only the visible archs.")
    spec = importlib.util.spec_from_file_location("t16_withheld_arch", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for attr in ("RECIPE", "build_model"):
        if not hasattr(mod, attr):
            raise EvalError(f"{p} lacks {attr}")
    return dict(mod.RECIPE), mod.build_model


def all_archs():
    archs = [(name, dict(rec), fn) for name, (rec, fn) in VISIBLE.items()]
    rec, fn = load_withheld()
    archs.append(("withheld", rec, fn))
    return archs


# ------------------------------------------------------------------ fixed training
def _augment(xpad, idx, gen):
    """Random crop (4px reflect pad, pre-applied) + horizontal flip, all on device."""
    b = idx.numel()
    dev = xpad.device
    dy = torch.randint(0, 9, (b,), device=dev, generator=gen)
    dx = torch.randint(0, 9, (b,), device=dev, generator=gen)
    ar = torch.arange(32, device=dev)
    xb = xpad[idx]                                                       # (B,3,40,40)
    rows = (dy[:, None] + ar[None, :])                                   # (B,32)
    xb = torch.gather(xb, 2, rows[:, None, :, None].expand(b, 3, 32, 40))
    cols = (dx[:, None] + ar[None, :])
    xb = torch.gather(xb, 3, cols[:, None, None, :].expand(b, 3, 32, 32))
    flip = torch.rand(b, device=dev, generator=gen) < 0.5
    return torch.where(flip[:, None, None, None], xb.flip(3), xb)


def train_one(build_fn, recipe, x01, y, seed, device, steps=None, batch=None):
    torch.manual_seed(seed)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    steps = steps or recipe["steps"]
    batch = batch or recipe["batch"]
    model = build_fn(NUM_CLASSES).to(device).to(memory_format=torch.channels_last)
    xn = ((x01 - MEAN) / STD).to(device)
    xpad = F.pad(xn, (4, 4, 4, 4), mode="reflect")
    yd = y.to(device)
    opt = torch.optim.SGD(model.parameters(), lr=recipe["lr"], momentum=recipe["momentum"],
                          nesterov=True, weight_decay=recipe["wd"])
    warm = max(1, int(recipe["warmup_frac"] * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else max(0.0, (steps - s) / max(1, steps - warm)))
    use_amp = device.type == "cuda"
    model.train()
    n = xn.shape[0]
    for _ in range(steps):
        idx = torch.randint(0, n, (batch,), device=device, generator=gen)
        xb = _augment(xpad, idx, gen).contiguous(memory_format=torch.channels_last)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
            loss = F.cross_entropy(model(xb), yd[idx], label_smoothing=recipe["label_smoothing"])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
    return model, float(loss.detach())


@torch.no_grad()
def test_top1(model, xt01, yt, device, bs=2000):
    model.eval()
    correct = 0
    use_amp = device.type == "cuda"
    for i in range(0, xt01.shape[0], bs):
        xb = ((xt01[i:i + bs].to(device) - MEAN.to(device)) / STD.to(device)).contiguous(
            memory_format=torch.channels_last)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
            pred = model(xb).argmax(1)
        correct += (pred == yt[i:i + bs].to(device)).sum().item()
    return correct / xt01.shape[0]


def evaluate(x, y, ar_data, mode, seed, device, log=print):
    """Train the 3 fixed classifiers on (x, y) and return (mean_top1, per_arch, walls)."""
    archs = all_archs()   # raises EvalError if the withheld arch is missing
    if mode == "cpu-smoke":
        steps, batch, tlimit = 3, 64, 500
    elif mode == "smoke":
        steps, batch, tlimit = 60, None, None
    else:
        steps, batch, tlimit = None, None, None
    xt, yt = load_test(ar_data, tlimit)
    xt01 = torch.from_numpy(xt).float().div_(255.0)
    ytt = torch.from_numpy(yt)
    if device.type == "cuda":
        xt01, ytt = xt01.to(device), ytt.to(device)
        torch.backends.cudnn.benchmark = True
    x01, yy = to_tensor_nchw01(x), torch.from_numpy(y.astype(np.int64))
    per_arch, walls = {}, {}
    for k, (name, rec, fn) in enumerate(archs):
        t0 = time.time()
        model, last_loss = train_one(fn, rec, x01, yy, seed * 1000 + k, device, steps, batch)
        acc = test_top1(model, xt01, ytt, device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        walls[name] = round(time.time() - t0, 2)
        per_arch[name] = acc
        log(f"[eval] {name}: top1={acc:.4f} last_loss={last_loss:.3f} wall={walls[name]}s")
        del model
    return float(np.mean(list(per_arch.values()))), per_arch, walls
