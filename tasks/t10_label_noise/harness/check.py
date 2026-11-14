"""Mechanical constraint checks for t10_label_noise.
static_check(solution_dir) scans the solution source for pretrained-weight use, network
downloads and access to data the solution must not read; and flags bundled weight files.
sandbox_check(sandbox_dir) checks that nothing was downloaded into the (empty, offline) caches.
usage (standalone): check.py <solution_dir>"""
import json, re, sys
from pathlib import Path

CODE_EXT = {".py", ".sh", ".ipynb", ".yaml", ".yml", ".json", ".cfg", ".toml", ".txt"}
WEIGHT_EXT = {".pt", ".pth", ".ckpt", ".safetensors", ".bin", ".onnx", ".h5", ".npz", ".npy", ".pkl", ".pickle", ".tar", ".gz", ".zip", ".msgpack"}
WEIGHT_FILE_MAX = 64 * 1024

RULES = {
    "pretrained_weights": [
        r"pretrained\s*=\s*(True|1|['\"])",
        r"\bweights\s*=\s*['\"]?[\w.]*(IMAGENET|DEFAULT|_Weights|SWAG|LVD|_V1|_V2)",
        r"\b\w+_Weights\b",
        r"torch\.hub|\bhub\.load\b|load_state_dict_from_url|model_zoo|get_model_weights|get_weight\(",
        r"from_pretrained|hf_hub_download|snapshot_download|huggingface_hub|\btransformers\b|\btimm\b|open_clip|\bclip\.load",
    ],
    "network": [
        r"urlretrieve|urllib\.request|\brequests\.(get|post)|download_url|download_and_extract|\bwget\b|\bcurl\b|https?://",
        r"\bsocket\b|http\.client",
    ],
    "forbidden_data": [
        r"cifar-10-batches-py|cifar-100-python|data_batch|test_batch|\bCIFAR10\b|\bCIFAR100\b|torchvision\.datasets|tv\.datasets",
        r"private|train_clean|noise_mask|test\.npz|train_noisy|AR_ROOT|AR_DATA|AR_REPO|/usr/xtmp",
    ],
}


def static_check(sol_dir):
    sol = Path(sol_dir)
    hits, big = [], []
    if not (sol / "solution.py").is_file():
        hits.append({"rule": "interface", "file": "solution.py", "line": 0, "text": "missing solution.py"})
    for f in sorted(sol.rglob("*")):
        if not f.is_file() or "__pycache__" in f.parts:
            continue
        rel = str(f.relative_to(sol))
        suf = f.suffix.lower()
        if suf in WEIGHT_EXT or (suf not in CODE_EXT and f.stat().st_size > WEIGHT_FILE_MAX):
            if f.stat().st_size > WEIGHT_FILE_MAX or suf in {".pt", ".pth", ".ckpt", ".safetensors", ".onnx", ".h5"}:
                big.append({"rule": "bundled_weights", "file": rel, "bytes": f.stat().st_size})
            continue
        if suf not in CODE_EXT:
            continue
        for i, line in enumerate(f.read_text(errors="replace").splitlines(), 1):
            for rule, pats in RULES.items():
                for p in pats:
                    if re.search(p, line):
                        hits.append({"rule": rule, "file": rel, "line": i, "text": line.strip()[:200]})
                        break
    viol = hits + big
    return {"ok": not viol, "violations": viol[:50], "n_violations": len(viol)}


def sandbox_check(sandbox_dir):
    files = [str(p.relative_to(sandbox_dir)) for p in Path(sandbox_dir).rglob("*")
             if p.is_file() and p.suffix.lower() in WEIGHT_EXT and p.stat().st_size > WEIGHT_FILE_MAX]
    return {"ok": not files, "downloaded_weight_like_files": files[:50]}


if __name__ == "__main__":
    r = static_check(sys.argv[1]); print(json.dumps(r, indent=1)); sys.exit(0 if r["ok"] else 1)
