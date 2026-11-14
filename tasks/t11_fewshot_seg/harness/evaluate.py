"""Evaluation step (process 2): rebuild the model from artifact/code/model.py with build_model(make_backbone_skeleton)
(no pretrained weights), load artifact/model.pt strictly, check backbone shapes / finiteness / determinism, then
score the VOC 2012 val set at the fixed 320x320 resolution in fp32 (TF32 off, model.eval()).

Protocol: input = float32 (B,3,320,320) RGB in [0,1] (the padded, long-side-320 val image); the model returns
logits (B,21,h,w); if (h,w) != (320,320) they are upsampled bilinearly (align_corners=False); prediction = argmax.
One confusion matrix accumulated over all val pixels whose label is 0..20 (255 = void boundary / padding ignored);
mIoU = mean over the 21 classes of TP / (TP + FP + FN).
usage: python evaluate.py <out_dir> <mode> <t_start_epoch_s>
"""
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import t11lib as L  # noqa: E402

METRIC = "voc12_val_miou"


@torch.no_grad()
def logits_of(model, xb):
    x = xb.permute(0, 3, 1, 2).float().div_(255)
    out = model(x)
    if not isinstance(out, torch.Tensor) or out.dim() != 4 or out.shape[:2] != (x.shape[0], L.NUM_CLASSES):
        raise RuntimeError(f"model output must be logits (B,{L.NUM_CLASSES},h,w), got "
                           f"{getattr(out, 'shape', type(out))}")
    out = out.float()
    if out.shape[-2:] != x.shape[-2:]:
        out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)
    return out


def main():
    out, mode, t_start = Path(sys.argv[1]).resolve(), sys.argv[2], float(sys.argv[3])
    t0 = time.time()
    cfg = L.MODES[mode]
    rep = {"mode": mode, "errors": [], "review_flags": []}
    valid, value = True, 0.0
    tr_p = out / "train_report.json"
    tr = json.loads(tr_p.read_text()) if tr_p.exists() else {"ok": False, "error": "no train_report.json"}
    rep["train"] = {k: tr.get(k) for k in ("ok", "error", "train_wall_s", "time_budget_s", "within_budget",
                                           "load_s", "n_train", "max_mem_gb", "structure", "backbone_rel_change",
                                           "torch_load_events")}
    if not tr.get("ok"):
        valid = False; rep["errors"].append(f"training failed: {tr.get('error')}")
    if tr.get("within_budget") is False:
        valid = False; rep["errors"].append("training exceeded the time budget")
    if tr.get("structure_errors"):
        valid = False; rep["errors"] += [f"structure: {e}" for e in tr["structure_errors"]]
    if tr.get("weight_load_violations"):
        valid = False; rep["errors"].append(f"torch.load / weight download outside scratch_dir during training: "
                                            f"{tr['weight_load_violations'][:3]}")
    scan_p = out / "scan.json"
    scan = json.loads(scan_p.read_text()) if scan_p.exists() else None
    rep["scan"] = scan
    if scan is None:
        rep["review_flags"].append("static scan missing")
    elif not scan.get("hard_ok", True):
        valid = False; rep["errors"].append("static scan: hard violation (see scan.hard_violations)")

    try:
        art = out / "artifact"
        if not (art / "model.pt").exists():
            raise RuntimeError("artifact/model.pt missing")
        dev = torch.device("cpu" if mode == "cpu-smoke" else "cuda")
        if mode == "cpu-smoke":
            torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", min(4, os.cpu_count() or 4))))
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        sd = torch.load(art / "model.pt", map_location="cpu", weights_only=True)
        bshape = {k[len("backbone."):]: tuple(v.shape) for k, v in sd.items() if k.startswith("backbone.")}
        arch_ok = bshape == L.backbone_shapes()
        rep["backbone_arch_ok"] = arch_ok
        if not arch_ok:
            valid = False; rep["errors"].append("artifact backbone.* keys/shapes differ from torchvision resnet50")
        model_mod = L.import_from(art / "code", "model")
        model = model_mod.build_model(L.make_backbone_skeleton)
        model.load_state_dict(sd, strict=True)
        model = model.float().to(dev).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        nonfinite = [k for k, v in model.state_dict().items() if v.is_floating_point() and not torch.isfinite(v).all()]
        if nonfinite:
            valid = False; rep["errors"].append(f"non-finite weights: {nonfinite[:5]}")
        rep["backbone_rel_change"] = L.backbone_drift(sd)

        X, Y = L.load_val(cfg["n_val"])
        bs = cfg["eval_bs"]
        conf = torch.zeros(L.NUM_CLASSES, L.NUM_CLASSES, dtype=torch.int64, device=dev)
        te = time.time()
        first = None
        for i in range(0, len(X), bs):
            xb = torch.from_numpy(X[i:i + bs]).to(dev)
            yb = torch.from_numpy(Y[i:i + bs]).to(dev).long()
            lo = logits_of(model, xb)
            if not torch.isfinite(lo).all():
                raise RuntimeError("non-finite logits on val images")
            if first is None:
                first = (xb, lo)
            conf += L.confusion(lo.argmax(1).reshape(-1), yb.reshape(-1))
        if dev.type == "cuda":
            torch.cuda.synchronize()
        rep["eval_forward_s"] = round(time.time() - te, 1)
        lo2 = logits_of(model, first[0])      # determinism: same batch again, same logits
        diff = float((lo2 - first[1]).abs().max())
        scale = float(first[1].abs().max()) + 1e-12
        det = diff <= 1e-5 * scale and torch.equal(lo2.argmax(1), first[1].argmax(1))
        rep["deterministic"] = bool(det); rep["determinism_max_abs_diff"] = diff
        if not det:
            valid = False; rep["errors"].append(f"eval not deterministic (max |dlogit| {diff:.3g})")
        miou, ious, pix_acc = L.miou_from_conf(conf.cpu())
        value = miou
        rep.update({"miou": miou, "pixel_acc": pix_acc, "n_val_images": len(X), "eval_res": L.RES,
                    "per_class_iou": {c: (None if not math.isfinite(v) else round(v, 4))
                                      for c, v in zip(json.loads((L.data_dir() / "train_meta.json").read_text())
                                                      ["classes"], ious)},
                    "n_scored_pixels": int(conf.sum())})
    except Exception as e:  # noqa: BLE001
        valid = False
        rep["errors"].append(f"eval failed: {type(e).__name__}: {e}")
        rep["traceback"] = traceback.format_exc()[-3000:]
    rep["eval_wall_s"] = round(time.time() - t0, 1)
    if not math.isfinite(value):
        value = 0.0
    m = {"metric": METRIC, "value": float(value), "valid": bool(valid), "constraint_report": rep,
         "wall_s": round(time.time() - t_start, 1), "mode": mode}
    (out / "metric.json").write_text(json.dumps(m, indent=1))
    print(json.dumps({k: m[k] for k in ("metric", "value", "valid", "wall_s", "mode")}))
    print(json.dumps({"pixel_acc": rep.get("pixel_acc"), "errors": rep["errors"]}))


if __name__ == "__main__":
    main()
