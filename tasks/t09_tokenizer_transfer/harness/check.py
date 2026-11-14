"""Static scan of the solution for t09_tokenizer_transfer. Heuristic: hard violations invalidate the run,
review flags are for the reviewer. The real enforcement is structural (the harness owns the model, the eval
data is never in the adaptation process, the jail has no network and a read-only filesystem).

Hard: network access, Hugging Face hub / from_pretrained / other pretrained models, held-out / private data,
      other datasets, env-var or absolute paths into the data tree, touching the tokenizer.
Review: file reads (should only use the ctx paths), transformers imports, subprocesses, torch.load.
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
     r"\bAutoModel\w*|GPT2LMHeadModel|\bpipeline\(|\btimm\b|sentence_transformers|fasttext|gensim|"
     r"\.safetensors\b|pytorch_model\.bin", "pretrained model other than the provided GPT-2 / model hub"),
    (r"/private\b|\bprivate/|heldout_ids|heldout_wiki|\beval_windows\b|decoy_", "held-out / private data"),
    (r"\bAR_ROOT\b|\bAR_DATA\b|\bHF_HOME\b|/usr/xtmp|/usr/project|\.cache/huggingface|\bxtmp\b",
     "path into the data tree (use only the paths passed in ctx)"),
    (r"\bload_dataset\b|^\s*(import|from)\s+datasets\b|wikimedia|\.parquet\b|fineweb|cifar|\bmc4\b|\bcc100\b|"
     r"\boscar\b", "other datasets"),
    (r"(tokenizer_path|gpt2_tokenizer_path)[^\n]*['\"]\s*,\s*['\"][wa]|\.save\(\s*ctx\[",
     "writing to the provided tokenizer files"),
]
REVIEW = [
    (r"\bopen\(|np\.load|np\.fromfile|\bglob\b|os\.listdir|os\.walk|Path\([^)]*\)\.(read|iterdir|glob)",
     "file access (allowed only for the ctx paths and ctx['scratch_dir'])"),
    (r"^\s*(import|from)\s+transformers\b", "transformers imported (tokenizers / torch suffice; no hub loading)"),
    (r"\bsubprocess\b|os\.system|os\.popen|\bexec\(|\beval\(", "subprocess / dynamic code"),
    (r"torch\.load|pickle", "loading a serialized object (only allowed for files the solution wrote itself)"),
    (r"register_\w*hook|_forward_hooks|_forward_pre_hooks", "module hooks (must be removed before adapt returns)"),
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
