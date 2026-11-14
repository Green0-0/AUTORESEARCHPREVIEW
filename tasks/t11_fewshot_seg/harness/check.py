"""Static scan of the solution for t11_fewshot_seg. Heuristic: hard violations invalidate the run, review flags
are for the reviewer. The real enforcement is structural (the harness builds the ImageNet backbone and checks
model.backbone is that object, torch.load calls during training are audited, the val data is never in the
training process, the jail has no network and a read-only filesystem).

Hard: network access; any pretrained weights other than the provided ResNet-50 (torchvision `weights=` /
      `pretrained=True`, torchvision segmentation/detection models, hub, timm, SAM, CLIP, DINO, ...);
      validation / private data; env-var or absolute paths into the data tree; other datasets; weight/data files
      shipped in the solution.
Review: file reads, torch.load, subprocesses, hooks.
usage: python check.py <solution_dir>   -> JSON on stdout
"""
import ast
import json
import re
import sys
from pathlib import Path

HARD = [
    (r"\burllib\b|\brequests\b|\bhttpx\b|\baiohttp\b|\bsocket\b|http\.client|\bwget\b|\bcurl\b|ftplib|"
     r"https?://", "network access"),
    (r"huggingface_hub|hf_hub_download|snapshot_download|from_pretrained|torch\.hub|load_state_dict_from_url|"
     r"\w+\([^)]*\bweights\s*=\s*(?!None\b)[\w\"']|pretrained\s*=\s*True|get_model_weights|_Weights\b|"
     r"IMAGENET1K|COCO_|\.get_state_dict\(|"
     r"models\.segmentation|models\.detection|deeplabv3|\bfcn_resnet|lraspp|maskrcnn|mask_rcnn|fasterrcnn|"
     r"retinanet|keypointrcnn|\btimm\b|segment_anything|\bSAM\b|sam_vit|open_clip|import\s+clip\b|"
     r"from\s+clip\b|\bdinov2\b|\bdino_|"
     r"transformers|mmseg|mmdet|detectron|segmentation_models_pytorch|\.safetensors\b|pytorch_model\.bin|"
     r"\bTORCH_HOME\b|hub/checkpoints", "pretrained weights other than the provided ImageNet ResNet-50"),
    (r"/private\b|\bprivate/|\bval_x\b|\bval_y\b|\bval_hw\b|decoy_", "validation / private data"),
    (r"\bAR_ROOT\b|\bAR_DATA\b|\bHF_HOME\b|/usr/xtmp|/usr/project|\.cache/|\bxtmp\b",
     "path into the data tree (use only ctx)"),
    (r"\bload_dataset\b|^\s*(import|from)\s+datasets\b|torchvision\.datasets|\bVOCSegmentation\b|"
     r"\bVOCdevkit\b|JPEGImages|SegmentationClass|\bSBD\b|SBDataset|\bcoco\b|cityscapes|\bade20k\b|"
     r"ImageNet\(|ILSVRC|imagenet_root|cifar", "other datasets"),
]
REVIEW = [
    (r"\bopen\(|np\.load|np\.fromfile|\bglob\b|os\.listdir|os\.walk|Path\([^)]*\)\.(read|iterdir|glob)",
     "file access (allowed only for ctx['scratch_dir'])"),
    (r"\bsubprocess\b|os\.system|os\.popen|\bexec\(|\beval\(\s*[\"'a-z]", "subprocess / dynamic code"),
    (r"torch\.load|pickle", "loading a serialized object (only allowed for files the solution wrote itself)"),
    (r"register_\w*hook|_forward_hooks|_forward_pre_hooks", "module hooks (must be removed before train returns)"),
    (r"torchvision\.models\.\w+\(|models\.(resnet|vgg|efficientnet|convnext|vit|mobilenet|regnet|swin)\w*\(",
     "builds a torchvision model itself (randomly initialised only; pretrained backbone = ctx/make_backbone)"),
]


def _doc_lines(src):
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
    hard, flags, files = [], [], []
    for f in sorted(list(root.rglob("*.py")) + list(root.rglob("*.sh"))):
        if "__pycache__" in f.parts:
            continue
        rel = str(f.relative_to(root))
        files.append(rel)
        try:
            src = f.read_text(errors="replace")
        except OSError:
            continue
        doc = _doc_lines(src) if f.suffix == ".py" else set()
        for n, line in enumerate(src.splitlines(), 1):
            s = line.strip()
            if s.startswith("#") or n in doc:
                continue
            code = line.split("#", 1)[0] if "#" in line and "'" not in line and '"' not in line else line
            rec = {"file": rel, "line": n, "text": s[:200]}
            for pat, why in HARD:
                if re.search(pat, code):
                    hard.append({**rec, "why": why})
            for pat, why in REVIEW:
                if re.search(pat, code):
                    flags.append({**rec, "why": why})
    other = [str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()
             and p.suffix in (".pt", ".pth", ".bin", ".safetensors", ".npy", ".npz", ".ckpt", ".onnx")]
    if other:
        hard.append({"file": other[:5], "line": 0, "text": "", "why": "weight / data files shipped in the solution"})
    return {"files": files, "hard_violations": hard, "hard_ok": not hard,
            "review_flags": flags[:60], "n_review_flags": len(flags)}


if __name__ == "__main__":
    print(json.dumps(scan(sys.argv[1]), indent=1))
