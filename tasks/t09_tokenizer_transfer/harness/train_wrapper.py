"""Adaptation step (process 1): build the harness GPT-2 skeleton (pretrained body, fresh 32k wte/lm_head), load
the pre-tokenized training stream onto the device, call the solution's adapt(model, ctx) under a timer, check
structure + tokenizer hash, save the artifact. Held-out data is never loaded in this process.

usage: python train_wrapper.py <solution_dir> <out_dir> <mode: full|smoke|cpu-smoke>
writes <out_dir>/artifact/{model.pt, code/} and <out_dir>/train_report.json
"""
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import t09lib as L  # noqa: E402


def main():
    sol, out, mode = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve(), sys.argv[3]
    art = out / "artifact"
    if art.exists():
        shutil.rmtree(art)
    art.mkdir(parents=True)
    scratch = out / "scratch"
    scratch.mkdir(exist_ok=True)
    rep = {"mode": mode, "ok": False}
    seed = int(os.environ.get("AR_SEED", "0"))
    cfg = L.MODES[mode]
    D = L.data_dir()
    try:
        meta = json.loads((D / "meta.json").read_text())
        tok_path = D / "tokenizer.json"
        tok_sha0 = L.sha_file(tok_path)
        rep["tokenizer_sha_ok_before"] = tok_sha0 == meta["tokenizer_sha256"]
        if mode == "cpu-smoke":
            device = torch.device("cpu")
            torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", min(4, os.cpu_count() or 4))))
        else:
            if not torch.cuda.is_available():
                raise RuntimeError("GPU mode requested but CUDA is unavailable")
            device = torch.device("cuda")
            torch.zeros(1, device=device).sum().item()  # CUDA init outside the timer
        t_load = time.time()
        train_ids = L.load_train_ids(device, cfg["n_train_tokens"])
        off = np.load(D / "train_doc_offsets.npy")
        if cfg["n_train_tokens"] is not None:
            off = off[off <= cfg["n_train_tokens"]]
        model, gpt2_wte = L.build_adapt_model(device, seed)
        gpt2_wte = gpt2_wte.to(device)
        rep["load_s"] = round(time.time() - t_load, 1)
        rep["n_train_tokens"] = int(train_ids.numel())

        shutil.copytree(sol, art / "code", ignore=shutil.ignore_patterns(
            "__pycache__", "*.pyc", "*.pt", "*.pth", "*.npy", "*.npz", "*.safetensors", "data", "out*", "runs"))
        adapt_mod = L.import_from(sol, "adapt")
        L.seed_all(seed)
        if device.type == "cuda":
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        budget = cfg["budget_s"]
        t0 = time.time()
        ctx = {"mode": mode, "device": device, "time_budget_s": budget, "deadline": t0 + budget, "seed": seed,
               "vocab_size": L.VOCAB, "n_ctx": L.N_CTX, "eos_id": L.EOS_ID,
               "train_ids": train_ids, "train_doc_offsets": torch.from_numpy(off),
               "gpt2_wte": gpt2_wte, "build_gpt2_reference": lambda: L.build_gpt2_reference(device),
               "tokenizer_path": str(tok_path), "gpt2_tokenizer_path": str(D / "gpt2" / "tokenizer.json"),
               "train_docs_path": str(D / "train_docs.jsonl"), "scratch_dir": str(scratch)}
        model.train()
        adapt_mod.adapt(model, ctx)
        if device.type == "cuda":
            torch.cuda.synchronize()
        wall = time.time() - t0
        rep.update({"adapt_wall_s": round(wall, 2), "time_budget_s": budget,
                    "within_budget": bool(wall <= budget + L.GRACE_S)})
        srep, serrs = L.structure_report(model)
        rep["structure"] = srep; rep["structure_errors"] = serrs
        rep["tokenizer_sha_ok_after"] = L.sha_file(tok_path) == meta["tokenizer_sha256"]
        sd = {k: v.detach().to("cpu", torch.float32).clone() for k, v in model.state_dict().items()}
        torch.save(sd, art / "model.pt")
        rep["ok"] = True
        if device.type == "cuda":
            rep["max_mem_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    except Exception as e:  # noqa: BLE001
        rep["error"] = f"{type(e).__name__}: {e}"
        rep["traceback"] = traceback.format_exc()
        print(rep["traceback"], file=sys.stderr)
    (out / "train_report.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps({k: v for k, v in rep.items() if k != "traceback"}, indent=2))
    sys.exit(0 if rep["ok"] else 1)


if __name__ == "__main__":
    main()
