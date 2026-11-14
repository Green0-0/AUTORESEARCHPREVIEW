"""t11_fewshot_seg harness library: mode table, the ONE allowed backbone (torchvision ResNet-50, ImageNet
IMAGENET1K_V2, loaded from a local file), data loading, structure checks, a torch.load audit, mIoU.
Pure torch + torchvision + numpy + stdlib.
"""
import importlib.util
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision

TASK = "t11_fewshot_seg"
NUM_CLASSES = 21
IGNORE = 255
RES = 320
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
GRACE_S = 10.0
BACKBONE_FILE = "resnet50_imagenet1k_v2.pth"

MODES = {
    #             train() budget   val images (None = all 1449)   eval batch
    "full":      {"budget_s": 900.0, "n_val": None, "eval_bs": 16},
    "smoke":     {"budget_s": 45.0,  "n_val": 256,  "eval_bs": 16},
    "cpu-smoke": {"budget_s": 40.0,  "n_val": 8,    "eval_bs": 4},
}


def data_dir() -> Path:
    return Path(os.environ.get("AR_DATA", "/usr/xtmp/kg349/ar/data")) / TASK


def private_dir() -> Path:
    return Path(os.environ.get("AR_ROOT", "/usr/xtmp/kg349/ar")) / "private" / TASK


def seed_all(s: int):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def import_from(dirpath, name):
    p = Path(dirpath) / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"sol_{name}", p)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(Path(dirpath)))
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.path.remove(str(Path(dirpath)))
    return mod


# ----------------------------------------------------------------------------------------------- backbone
_IMAGENET_SD = None
BUILT_BACKBONES = []          # every backbone handed to the solution (identity check after training)
ALLOWED_KW = {"replace_stride_with_dilation"}


def imagenet_state():
    global _IMAGENET_SD
    if _IMAGENET_SD is None:
        _IMAGENET_SD = torch.load(data_dir() / BACKBONE_FILE, map_location="cpu", weights_only=True)
    return _IMAGENET_SD


def _make_resnet50(pretrained: bool, **kw):
    bad = set(kw) - ALLOWED_KW
    if bad:
        raise ValueError(f"make_backbone: only {sorted(ALLOWED_KW)} may be passed, got {sorted(bad)}")
    m = torchvision.models.resnet50(weights=None, **kw)
    if pretrained:
        m.load_state_dict(imagenet_state(), strict=True)
        BUILT_BACKBONES.append(m)
    return m


def make_backbone_pretrained(**kw):
    """torchvision ResNet-50 with the ImageNet IMAGENET1K_V2 weights (train time)."""
    return _make_resnet50(True, **kw)


def make_backbone_skeleton(**kw):
    """Same architecture, no weights (eval time: the artifact's weights are loaded into it)."""
    return _make_resnet50(False, **kw)


def backbone_shapes():
    return {k: tuple(v.shape) for k, v in torchvision.models.resnet50(weights=None).state_dict().items()}


def structure_report(model, ref_shapes):
    """model.backbone must be a harness-built ImageNet ResNet-50; state_dict keys/shapes unchanged since
    build_model(); no hooks. Returns (report, errors)."""
    errs = []
    bb = getattr(model, "backbone", None)
    if not isinstance(bb, torchvision.models.ResNet):
        errs.append(f"model.backbone missing or not a torchvision ResNet ({type(bb)})")
    elif not any(bb is b for b in BUILT_BACKBONES):
        errs.append("model.backbone is not one of the backbones built by ctx/make_backbone (ImageNet weights)")
    else:
        got = {k: tuple(v.shape) for k, v in bb.state_dict().items()}
        if got != backbone_shapes():
            errs.append("model.backbone state_dict keys/shapes differ from torchvision resnet50")
    got = {k: tuple(v.shape) for k, v in model.state_dict().items()}
    if got != ref_shapes:
        errs.append(f"model state_dict changed during train(): extra {sorted(set(got) - set(ref_shapes))[:6]} "
                    f"missing {sorted(set(ref_shapes) - set(got))[:6]}")
    hooks = [n for n, m in model.named_modules()
             if m._forward_hooks or m._forward_pre_hooks or m._backward_hooks
             or getattr(m, "_backward_pre_hooks", None)]
    if hooks:
        errs.append(f"hooks left on modules: {hooks[:8]}")
    rep = {"n_state_tensors": len(got), "n_params": int(sum(p.numel() for p in model.parameters())),
           "n_backbone_params": int(sum(p.numel() for p in bb.parameters())) if isinstance(bb, nn.Module) else 0}
    return rep, errs


def backbone_drift(sd):
    """Relative change of a few backbone tensors from the ImageNet weights (diagnostic)."""
    ref = imagenet_state()
    out = {}
    for k in ("conv1.weight", "layer1.0.conv1.weight", "layer3.0.conv2.weight", "layer4.2.conv3.weight"):
        a = sd.get("backbone." + k)
        if a is not None:
            out[k] = round(float((a.float() - ref[k].float()).norm() / (ref[k].float().norm() + 1e-12)), 4)
    return out


# ----------------------------------------------------------------------------------------------- torch.load audit
class LoadAudit:
    """Records every torch.load / torch.hub weight fetch during train(); anything outside the scratch dir is a
    violation (the only allowed pretrained weights are the ones the harness already put into the backbone)."""

    def __init__(self, scratch):
        self.scratch = str(Path(scratch).resolve())
        self.events = []

    def __enter__(self):
        self._load = torch.load
        self._url = torch.hub.load_state_dict_from_url
        audit = self

        def load(f, *a, **k):
            p = str(Path(f).resolve()) if isinstance(f, (str, os.PathLike)) else repr(type(f))
            audit.events.append(p)
            return audit._load(f, *a, **k)

        def url(*a, **k):
            audit.events.append(f"hub:{a[:1]}")
            raise RuntimeError("downloading weights is not allowed in t11_fewshot_seg")

        import torchvision.models._api as tvapi
        self._tv = tvapi
        self._tvurl = getattr(tvapi, "load_state_dict_from_url", None)
        torch.load = load
        torch.hub.load_state_dict_from_url = url
        tvapi.load_state_dict_from_url = url      # torchvision `weights=...` path (would hit $TORCH_HOME cache)
        return self

    def __exit__(self, *exc):
        torch.load = self._load
        torch.hub.load_state_dict_from_url = self._url
        if self._tvurl is not None:
            self._tv.load_state_dict_from_url = self._tvurl

    def violations(self):
        ok = str((data_dir() / BACKBONE_FILE).resolve())
        return [e for e in self.events if not (e.startswith(self.scratch + os.sep) or e == ok)]


# ----------------------------------------------------------------------------------------------- data / metric
def load_train(device):
    D = data_dir()
    x = torch.from_numpy(np.load(D / "train_x.npy")).to(device)          # uint8 (100,320,320,3)
    y = torch.from_numpy(np.load(D / "train_y.npy")).to(device)          # uint8 (100,320,320)
    hw = torch.from_numpy(np.load(D / "train_hw.npy").astype(np.int64))
    meta = json.loads((D / "train_meta.json").read_text())
    return x, y, hw, meta


def load_val(n_val):
    P = private_dir()
    x = np.load(P / "val_x.npy", mmap_mode="r")
    y = np.load(P / "val_y.npy", mmap_mode="r")
    idx = np.arange(len(x)) if n_val is None or n_val >= len(x) else \
        np.linspace(0, len(x) - 1, n_val).round().astype(np.int64)      # evenly spread fixed subset
    return np.ascontiguousarray(x[idx]), np.ascontiguousarray(y[idx])


def confusion(pred, y, n=NUM_CLASSES):
    """pred, y: int64 tensors, same shape; labels outside 0..n-1 (255 void/padding) are ignored."""
    m = y < n
    return torch.bincount(y[m] * n + pred[m], minlength=n * n).view(n, n)


def miou_from_conf(conf):
    conf = conf.double()
    tp = conf.diag()
    union = conf.sum(0) + conf.sum(1) - tp
    present = union > 0
    iou = torch.where(present, tp / union.clamp_min(1), torch.full_like(tp, float("nan")))
    return float(iou[present].mean()), iou.tolist(), float(tp.sum() / conf.sum().clamp_min(1))
