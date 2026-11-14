"""Evaluation step (process 2): rebuild the harness GPT-2 skeleton (32k vocab), load artifact/model.pt strictly,
check shapes / finiteness / tokenizer hash, then compute held-out perplexity per token on the fixed eval windows
in fp32 (TF32 off, model.eval()). Also reports bits per byte / per character (tokenizer-independent).

Scoring: each window is 1025 held-out tokens w; input w[:1024], targets w[1:]; only the last 512 targets are
scored, so every scored token sees >= 512 tokens of context (context 1024, stride 512).
usage: python evaluate.py <out_dir> <mode> <t_start_epoch_s>
"""
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import t09lib as L  # noqa: E402

METRIC = "heldout_ppl_per_token"


@torch.no_grad()
def nll_batch(model, w):
    """w (B, 1025) int64 on device -> (sum NLL over scored targets [fp64], per-target NLL (B, 512))."""
    x, y = w[:, :-1], w[:, 1:]
    h = model.hidden(x)[:, L.CTX - L.SCORE:]
    logits = model.lm_head(h).float()
    nll = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y[:, L.CTX - L.SCORE:].reshape(-1),
                          reduction="none").view(w.shape[0], L.SCORE)
    return nll.double().sum().item(), nll


def main():
    out, mode, t_start = Path(sys.argv[1]).resolve(), sys.argv[2], float(sys.argv[3])
    t0 = time.time()
    cfg = L.MODES[mode]
    rep = {"mode": mode, "errors": [], "review_flags": []}
    valid, value = True, float("inf")
    tr_p = out / "train_report.json"
    tr = json.loads(tr_p.read_text()) if tr_p.exists() else {"ok": False, "error": "no train_report.json"}
    rep["train"] = {k: tr.get(k) for k in ("ok", "error", "adapt_wall_s", "time_budget_s", "within_budget",
                                           "load_s", "n_train_tokens", "max_mem_gb", "structure")}
    if not tr.get("ok"):
        valid = False; rep["errors"].append(f"adaptation failed: {tr.get('error')}")
    if tr.get("within_budget") is False:
        valid = False; rep["errors"].append("adaptation exceeded the time budget")
    if tr.get("structure_errors"):
        valid = False; rep["errors"] += [f"structure: {e}" for e in tr["structure_errors"]]
    for k in ("tokenizer_sha_ok_before", "tokenizer_sha_ok_after"):
        if tr.get(k) is False:
            valid = False; rep["errors"].append(f"{k}: tokenizer.json hash differs from meta.json")
    scan_p = out / "scan.json"
    scan = json.loads(scan_p.read_text()) if scan_p.exists() else None
    rep["scan"] = scan
    if scan is None:
        rep["review_flags"].append("static scan missing")
    elif not scan.get("hard_ok", True):
        valid = False; rep["errors"].append("static scan: hard violation (see scan.hard_violations)")

    try:
        D = L.data_dir()
        meta = json.loads((D / "meta.json").read_text())
        tok_ok = L.sha_file(D / "tokenizer.json") == meta["tokenizer_sha256"]
        rep["tokenizer_sha_ok"] = tok_ok
        if not tok_ok:
            valid = False; rep["errors"].append("tokenizer.json hash differs from meta.json")
        art = out / "artifact"
        if not (art / "model.pt").exists():
            raise RuntimeError("artifact/model.pt missing")
        dev = torch.device("cpu" if mode == "cpu-smoke" else "cuda")
        if mode == "cpu-smoke":
            torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", min(4, os.cpu_count() or 4))))
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        sd = torch.load(art / "model.pt", map_location="cpu", weights_only=True)
        exp = L.expected_shapes()
        got = {k: tuple(v.shape) for k, v in sd.items()}
        shape_ok = got == exp
        rep["arch_ok"] = shape_ok
        if not shape_ok:
            valid = False
            rep["errors"].append(f"artifact keys/shapes differ from GPT-2 small (32k vocab): extra "
                                 f"{sorted(set(got) - set(exp))[:5]} missing {sorted(set(exp) - set(got))[:5]} "
                                 f"changed {[k for k in exp if k in got and got[k] != exp[k]][:5]}")
        model = L.GPT2(L.VOCAB)
        model.load_state_dict(sd, strict=True)
        model = model.float().to(dev).eval()
        nonfinite = [k for k, v in model.state_dict().items() if not torch.isfinite(v).all()]
        if nonfinite:
            valid = False; rep["errors"].append(f"non-finite weights: {nonfinite[:5]}")
        body_ref = L.load_gpt2_state()
        moved = {k: float((sd[k].float() - body_ref[k].float()).norm() / (body_ref[k].float().norm() + 1e-12))
                 for k in ("wpe.weight", "h.0.attn.c_attn.weight", "h.11.mlp.c_proj.weight", "ln_f.weight")}
        rep["rel_change_from_gpt2"] = {k: round(v, 4) for k, v in moved.items()}
        rep["tied_head_in_artifact"] = bool(torch.equal(sd["wte.weight"], sd["lm_head.weight"]))

        W = L.load_eval_windows(cfg["n_win"])
        nb_tab, nc_tab = L.token_byte_tables(D / "tokenizer.json")
        sc = W[:, 1:][:, L.CTX - L.SCORE:].numpy()
        n_bytes, n_chars = int(nb_tab[sc].sum()), int(nc_tab[sc].sum())
        bs = cfg["eval_bs"]
        te = time.time()
        tot, n_tok, first = 0.0, 0, None
        for i in range(0, len(W), bs):
            w = W[i:i + bs].to(dev)
            s, nll = nll_batch(model, w)
            if not math.isfinite(s):
                raise RuntimeError("non-finite held-out loss")
            if first is None:
                first = (w, s)
            tot += s; n_tok += nll.numel()
        rep["eval_forward_s"] = round(time.time() - te, 1)
        s2, _ = nll_batch(model, first[0])        # determinism: same batch again, same result
        det = abs(s2 - first[1]) <= 1e-6 * abs(first[1])
        rep["deterministic"] = bool(det)
        if not det:
            valid = False; rep["errors"].append(f"eval not deterministic: {first[1]} vs {s2}")
        mean_nll = tot / n_tok
        value = math.exp(mean_nll)
        rep.update({"ppl": value, "nll_per_token": mean_nll, "n_scored_tokens": n_tok, "n_windows": len(W),
                    "context": L.CTX, "stride": L.SCORE,
                    "bits_per_byte": tot / math.log(2) / n_bytes, "bits_per_char": tot / math.log(2) / n_chars,
                    "n_scored_bytes": n_bytes, "n_scored_chars": n_chars,
                    "chars_per_token": n_chars / n_tok})
    except Exception as e:  # noqa: BLE001
        valid = False
        rep["errors"].append(f"eval failed: {type(e).__name__}: {e}")
        rep["traceback"] = traceback.format_exc()[-3000:]
    rep["eval_wall_s"] = round(time.time() - t0, 1)
    if not math.isfinite(value):
        value = 1e9          # crash / divergence: huge perplexity, valid=false
    m = {"metric": METRIC, "value": float(value), "valid": bool(valid), "constraint_report": rep,
         "wall_s": round(time.time() - t_start, 1), "mode": mode}
    (out / "metric.json").write_text(json.dumps(m, indent=1))
    print(json.dumps({k: m[k] for k in ("metric", "value", "valid", "wall_s", "mode")}))
    print(json.dumps({"bits_per_char": rep.get("bits_per_char"), "errors": rep["errors"]}))


if __name__ == "__main__":
    main()
