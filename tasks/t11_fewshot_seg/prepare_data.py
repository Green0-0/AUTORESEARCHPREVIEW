"""Idempotent data prep for t11_fewshot_seg (dataset problem #11, Few-Shot Segmentation, shortened).

Needs network (run once, outside the jail) and Pillow (in the main venv):
    source env.sh && python tasks/t11_fewshot_seg/prepare_data.py          (~3 min, 4 cores, ~8 GB RAM)

Steps
 1. Pascal VOC 2012 trainval (the genuine `VOCtrainval_11-May-2012.tar`, 1,999,639,040 bytes,
    md5 6cd6e144f989b92b3379bac3b3de84fd = the value torchvision's VOCSegmentation checks) from the official
    Oxford host, into a temporary dir under $AR_ROOT/tmp. Only the segmentation split files, the SegmentationClass
    PNGs and the JPEGs of the 2,913 segmentation images are read. The tar and every extracted file are DELETED at
    the end: the raw val masks must not stay anywhere the Claude jail can read (only $AR_ROOT/private is decoyed).
 2. Counts checked: train 1,464 / val 1,449 segmentation images.
 3. Few-shot train set: for each object class c = 1..20 in order, 5 images are drawn (rng seed SEL_SEED) from the
    VOC *train* segmentation images that contain class c and were not drawn for an earlier class -> 100 distinct
    images, each with its full 21-class annotation (other classes present in the image stay labelled).
 4. Fixed resolution: every image is resized so its LONGER side is 320 (PIL bilinear with antialiasing; masks
    nearest), then padded bottom/right to 320x320 with the ImageNet mean colour (124,116,104) and mask label 255.
    255 (VOC "void" boundaries + padding) is ignored in training and evaluation.
 5. ResNet-50 IMAGENET1K_V2 (torchvision `resnet50-11ad3fa6.pth`) -> $TORCH_HOME, copied to the task data dir.
Outputs
    $AR_DATA/t11_fewshot_seg/train_x.npy     uint8 (100,320,320,3)  RGB, NHWC
    $AR_DATA/t11_fewshot_seg/train_y.npy     uint8 (100,320,320)    0..20, 255 = ignore
    $AR_DATA/t11_fewshot_seg/train_hw.npy    int16 (100,2)          resized (h,w) before padding
    $AR_DATA/t11_fewshot_seg/train_meta.json VOC ids, the class each image was drawn for, class names
    $AR_DATA/t11_fewshot_seg/resnet50_imagenet1k_v2.pth             the ONE allowed pretrained backbone
    $AR_DATA/t11_fewshot_seg/meta.json       seeds, source md5, sha256 of every file
    $AR_ROOT/private/t11_fewshot_seg/val_x.npy  uint8 (1449,320,320,3)
    $AR_ROOT/private/t11_fewshot_seg/val_y.npy  uint8 (1449,320,320)
    $AR_ROOT/private/t11_fewshot_seg/val_hw.npy int16 (1449,2)
Private files are .npy only, so scripts/make_decoy_private.py can fake them. Re-running is a no-op when the
outputs exist and hashes match meta.json; a mismatch is an error (never silently re-drawn).
"""
import hashlib
import io
import json
import os
import shutil
import sys
import tarfile
import time
from pathlib import Path

import numpy as np

TASK = "t11_fewshot_seg"
URL = "https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"
MD5 = "6cd6e144f989b92b3379bac3b3de84fd"
SEL_SEED = 1111
SHOTS = 5
RES = 320
PAD_RGB = (124, 116, 104)
IGNORE = 255
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
           "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train",
           "tvmonitor"]
RESNET_FILE = "resnet50-11ad3fa6.pth"
PRE = "VOCdevkit/VOC2012/"


def log(*a):
    print(f"[prepare {time.strftime('%H:%M:%S')}]", *a, flush=True)


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def md5_file(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def save(arr, dst: Path):
    tmp = dst.with_name(dst.stem + ".tmp.npy")
    np.save(tmp, arr)
    os.replace(tmp, dst)


def to_fixed(img, mask):
    """PIL RGB image + PIL P-mode mask -> (uint8 (RES,RES,3), uint8 (RES,RES), (h,w)) long side RES, padded."""
    from PIL import Image
    w, h = img.size
    s = RES / max(w, h)
    nw, nh = max(1, round(w * s)), max(1, round(h * s))
    im = np.asarray(img.convert("RGB").resize((nw, nh), Image.BILINEAR, reducing_gap=None))
    mk = np.asarray(mask.resize((nw, nh), Image.NEAREST))
    x = np.empty((RES, RES, 3), np.uint8); x[:] = PAD_RGB
    y = np.full((RES, RES), IGNORE, np.uint8)
    x[:nh, :nw] = im; y[:nh, :nw] = mk
    return x, y, (nh, nw)


def main():
    ar_root = Path(os.environ.get("AR_ROOT", "/usr/xtmp/kg349/ar"))
    ar_data = Path(os.environ.get("AR_DATA", ar_root / "data"))
    pub, priv = ar_data / TASK, ar_root / "private" / TASK
    meta_p = pub / "meta.json"
    outs = {"train_x": pub / "train_x.npy", "train_y": pub / "train_y.npy", "train_hw": pub / "train_hw.npy",
            "train_meta": pub / "train_meta.json", "resnet50": pub / "resnet50_imagenet1k_v2.pth",
            "val_x": priv / "val_x.npy", "val_y": priv / "val_y.npy", "val_hw": priv / "val_hw.npy"}
    if meta_p.exists() and all(p.exists() for p in outs.values()):
        meta = json.loads(meta_p.read_text())
        bad = [k for k, p in outs.items() if sha_file(p) != meta["sha256"][k]]
        if bad:
            raise SystemExit(f"hash mismatch for {bad}: refusing to re-draw silently; delete the outputs to rebuild")
        log("all outputs present, hashes match: nothing to do"); return
    pub.mkdir(parents=True, exist_ok=True); priv.mkdir(parents=True, exist_ok=True)

    # ---- backbone
    import torch
    import torchvision
    w = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
    assert w.url.endswith(RESNET_FILE), w.url
    torchvision.models.resnet50(weights=w)  # downloads into $TORCH_HOME/hub/checkpoints, checks the hash prefix
    src = Path(torch.hub.get_dir()) / "checkpoints" / RESNET_FILE
    shutil.copy(src, outs["resnet50"])
    log("resnet50 IMAGENET1K_V2", outs["resnet50"], os.path.getsize(outs["resnet50"]))

    # ---- VOC download (temporary)
    tmpd = ar_root / "tmp" / "t11_voc_download"
    tmpd.mkdir(parents=True, exist_ok=True)
    tar_p = tmpd / "VOCtrainval_11-May-2012.tar"
    try:
        if not tar_p.exists() or md5_file(tar_p) != MD5:
            log("downloading", URL)
            import urllib.request
            urllib.request.urlretrieve(URL, tar_p)
        md5 = md5_file(tar_p)
        assert md5 == MD5, f"md5 {md5} != {MD5}"
        log("tar md5 ok", md5, os.path.getsize(tar_p))
        from PIL import Image
        files = {}
        with tarfile.open(tar_p) as tf:
            for m in tf:
                n = m.name
                if not m.isfile() or not n.startswith(PRE):
                    continue
                r = n[len(PRE):]
                if r in ("ImageSets/Segmentation/train.txt", "ImageSets/Segmentation/val.txt") or \
                        r.startswith("SegmentationClass/") or r.startswith("JPEGImages/"):
                    files[r] = m
            split = {s: tf.extractfile(files[f"ImageSets/Segmentation/{s}.txt"]).read().decode().split()
                     for s in ("train", "val")}
            log("counts", {k: len(v) for k, v in split.items()})
            assert len(split["train"]) == 1464 and len(split["val"]) == 1449, "unexpected VOC2012 seg counts"
            assert not set(split["train"]) & set(split["val"])

            def load(i):
                img = Image.open(io.BytesIO(tf.extractfile(files[f"JPEGImages/{i}.jpg"]).read()))
                mk = Image.open(io.BytesIO(tf.extractfile(files[f"SegmentationClass/{i}.png"]).read()))
                assert mk.mode == "P", mk.mode
                img.load(); mk.load()
                assert img.size == mk.size
                return img, mk

            # ---- few-shot selection on the train split (presence computed at full resolution)
            present = {}
            for i in split["train"]:
                a = np.asarray(load(i)[1])
                u = np.unique(a)
                assert set(u.tolist()) <= set(range(21)) | {IGNORE}, (i, u)
                present[i] = set(int(c) for c in u if 1 <= c <= 20)
            rng = np.random.default_rng(SEL_SEED)
            chosen, drawn_for = [], []
            for c in range(1, 21):
                cand = sorted(i for i in split["train"] if c in present[i] and i not in chosen)
                pick = rng.choice(len(cand), SHOTS, replace=False)
                for j in sorted(pick.tolist()):
                    chosen.append(cand[j]); drawn_for.append(c)
            assert len(set(chosen)) == 20 * SHOTS

            def build(ids):
                X = np.empty((len(ids), RES, RES, 3), np.uint8)
                Y = np.empty((len(ids), RES, RES), np.uint8)
                HW = np.empty((len(ids), 2), np.int16)
                for k, i in enumerate(ids):
                    X[k], Y[k], HW[k] = to_fixed(*load(i))
                return X, Y, HW

            tx, ty, thw = build(chosen)
            log("train built", tx.shape)
            vx, vy, vhw = build(split["val"])
            log("val built", vx.shape)
        # class coverage of the few-shot set (images containing each class)
        cover = {CLASSES[c]: int(sum(c in present[i] for i in chosen)) for c in range(1, 21)}
        save(tx, outs["train_x"]); save(ty, outs["train_y"]); save(thw, outs["train_hw"])
        save(vx, outs["val_x"]); save(vy, outs["val_y"]); save(vhw, outs["val_hw"])
        outs["train_meta"].write_text(json.dumps({"voc_ids": chosen, "drawn_for_class": drawn_for,
                                                  "classes": CLASSES, "images_containing_class": cover}, indent=1))
        vc = np.bincount(vy.reshape(-1), minlength=256)
        tc = np.bincount(ty.reshape(-1), minlength=256)
        meta = {"task": TASK, "source_url": URL, "source_md5": MD5, "sel_seed": SEL_SEED, "shots": SHOTS,
                "res": RES, "pad_rgb": PAD_RGB, "ignore": IGNORE, "classes": CLASSES,
                "resize": "longer side -> 320 (PIL bilinear, antialiased; masks nearest), pad bottom/right to 320x320",
                "n_train": len(chosen), "n_val": len(split["val"]),
                "train_pixels_per_class": tc[:21].tolist(), "val_pixels_per_class": vc[:21].tolist(),
                "train_ignore_pixels": int(tc[IGNORE]), "val_ignore_pixels": int(vc[IGNORE]),
                "resnet50_url": w.url, "sha256": {k: sha_file(p) for k, p in outs.items()}}
        meta_p.write_text(json.dumps(meta, indent=1))
        os.chmod(priv, 0o700)
        log("done", json.dumps({k: meta[k] for k in ("n_train", "n_val")}), "coverage", cover)
    finally:
        shutil.rmtree(tmpd, ignore_errors=True)   # raw VOC (incl. val masks) must not persist
        log("removed raw download", tmpd)


if __name__ == "__main__":
    sys.exit(main())
