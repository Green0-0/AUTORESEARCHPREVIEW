"""t09_tokenizer_transfer harness library: the fixed GPT-2-small skeleton, data loading, mode table and checks.
Pure torch + numpy + stdlib (no transformers / tokenizers needed by the harness itself).
"""
import hashlib
import importlib.util
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

TASK = "t09_tokenizer_transfer"
VOCAB = 32_000            # the fixed task tokenizer
GPT2_VOCAB = 50_257
N_CTX, N_EMBD, N_LAYER, N_HEAD = 1024, 768, 12, 12
EOS_ID = 0
CTX, SCORE = 1024, 512    # eval: 1024-token windows, last 512 positions scored (stride 512)
GRACE_S = 10.0

MODES = {
    #             adapt() budget  scored windows  eval batch  train tokens given (None = all)
    "full":      {"budget_s": 900.0, "n_win": 2048, "eval_bs": 16, "n_train_tokens": None},
    "smoke":     {"budget_s": 45.0,  "n_win": 128,  "eval_bs": 16, "n_train_tokens": None},
    "cpu-smoke": {"budget_s": 40.0,  "n_win": 6,    "eval_bs": 3,  "n_train_tokens": 2_000_000},
}


def data_dir() -> Path:
    return Path(os.environ.get("AR_DATA", "/usr/xtmp/kg349/ar/data")) / TASK


def private_dir() -> Path:
    root = Path(os.environ.get("AR_ROOT", "/usr/xtmp/kg349/ar"))
    return root / "private" / TASK


def seed_all(s: int):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def sha_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


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


# ----------------------------------------------------------------------------------------------- model
class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.c_attn = nn.Linear(N_EMBD, 3 * N_EMBD)
        self.c_proj = nn.Linear(N_EMBD, N_EMBD)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(C, dim=2)
        q, k, v = (t.view(B, T, N_HEAD, C // N_HEAD).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.c_proj(y.transpose(1, 2).reshape(B, T, C))


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.c_fc = nn.Linear(N_EMBD, 4 * N_EMBD)
        self.c_proj = nn.Linear(4 * N_EMBD, N_EMBD)

    def forward(self, x):
        return self.c_proj(F.gelu(self.c_fc(x), approximate="tanh"))


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln_1 = nn.LayerNorm(N_EMBD, eps=1e-5)
        self.attn = Attention()
        self.ln_2 = nn.LayerNorm(N_EMBD, eps=1e-5)
        self.mlp = MLP()

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


class GPT2(nn.Module):
    """GPT-2 small (124M), numerically equivalent to HF GPT2LMHeadModel (no dropout). Only `wte` and `lm_head`
    depend on the vocabulary. `lm_head` is a separate Linear; tie it with `model.tie_weights()` if wanted."""

    def __init__(self, vocab_size: int = VOCAB):
        super().__init__()
        self.vocab_size = vocab_size
        self.wte = nn.Embedding(vocab_size, N_EMBD)
        self.wpe = nn.Embedding(N_CTX, N_EMBD)
        self.h = nn.ModuleList([Block() for _ in range(N_LAYER)])
        self.ln_f = nn.LayerNorm(N_EMBD, eps=1e-5)
        self.lm_head = nn.Linear(N_EMBD, vocab_size, bias=False)

    def tie_weights(self):
        self.lm_head.weight = self.wte.weight

    def hidden(self, idx):
        """idx (B, T) int64, T <= 1024 -> final hidden states after ln_f, (B, T, 768)."""
        T = idx.shape[1]
        x = self.wte(idx) + self.wpe(torch.arange(T, device=idx.device))
        for b in self.h:
            x = b(x)
        return self.ln_f(x)

    def forward(self, idx):
        """idx (B, T) int64 -> logits (B, T, vocab_size)."""
        return self.lm_head(self.hidden(idx))


def load_gpt2_state():
    return torch.load(data_dir() / "gpt2" / "gpt2_state.pt", map_location="cpu", weights_only=True)


def build_adapt_model(device, seed: int):
    """GPT-2 small with pretrained wpe / blocks / ln_f; new wte and lm_head (32k) drawn N(0, 0.02)."""
    sd = load_gpt2_state()
    m = GPT2(VOCAB)
    body = {k: v for k, v in sd.items() if not k.startswith("wte.")}
    missing, unexpected = m.load_state_dict(body, strict=False)
    assert set(missing) == {"wte.weight", "lm_head.weight"} and not unexpected, (missing, unexpected)
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        m.wte.weight.copy_(torch.randn(m.wte.weight.shape, generator=g) * 0.02)
        m.lm_head.weight.copy_(torch.randn(m.lm_head.weight.shape, generator=g) * 0.02)
    return m.to(device), sd["wte.weight"]


def build_gpt2_reference(device):
    """The original pretrained GPT-2 small with its own 50,257-token vocabulary (tied head)."""
    m = GPT2(GPT2_VOCAB)
    sd = load_gpt2_state()
    sd = dict(sd); sd["lm_head.weight"] = sd["wte.weight"]
    m.load_state_dict(sd, strict=True)
    m.tie_weights()
    return m.to(device)


def expected_shapes(vocab=VOCAB):
    return {k: tuple(v.shape) for k, v in GPT2(vocab).state_dict().items()}


def structure_report(model):
    """Class, state_dict keys/shapes, no hooks, no extra modules. Returns (report, errors)."""
    errs = []
    if type(model) is not GPT2:
        errs.append(f"model class changed: {type(model)}")
    exp = expected_shapes()
    got = {k: tuple(v.shape) for k, v in model.state_dict().items()}
    if set(got) != set(exp):
        errs.append(f"state_dict keys differ: extra {sorted(set(got) - set(exp))[:8]}, "
                    f"missing {sorted(set(exp) - set(got))[:8]}")
    bad = [k for k in exp if k in got and got[k] != exp[k]]
    if bad:
        errs.append(f"shape changed: {[(k, got[k], exp[k]) for k in bad[:8]]}")
    hooks = [n for n, m in model.named_modules()
             if m._forward_hooks or m._forward_pre_hooks or m._backward_hooks
             or getattr(m, "_backward_pre_hooks", None)]
    if hooks:
        errs.append(f"hooks left on modules: {hooks[:8]}")
    n_mod = sum(1 for _ in model.modules())
    n_exp = sum(1 for _ in GPT2(VOCAB).modules())
    if n_mod != n_exp:
        errs.append(f"module count {n_mod} != {n_exp}")
    tied = model.lm_head.weight.data_ptr() == model.wte.weight.data_ptr()
    rep = {"n_state_tensors": len(got), "tied_head": bool(tied),
           "n_params_unique": int(sum(p.numel() for p in model.parameters()))}
    return rep, errs


# ----------------------------------------------------------------------------------------------- data
def load_train_ids(device, n_tokens=None):
    a = np.load(data_dir() / "train_ids.npy", mmap_mode="r")
    if n_tokens is not None:
        a = a[:n_tokens]
    return torch.from_numpy(np.asarray(a, dtype=np.int32)).to(device)


def load_eval_windows(n_win):
    w = np.load(private_dir() / "eval_windows.npy", mmap_mode="r")
    assert w.shape[1] == CTX + 1, w.shape
    if n_win < len(w):
        w = w[np.linspace(0, len(w) - 1, n_win).round().astype(np.int64)]   # evenly spread subset
    return torch.from_numpy(np.ascontiguousarray(w, dtype=np.int64))


_BYTE_DECODER = None


def _byte_decoder():
    """Inverse of GPT-2's bytes_to_unicode map (byte-level BPE token strings -> raw bytes)."""
    global _BYTE_DECODER
    if _BYTE_DECODER is None:
        bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
        cs = bs[:]
        n = 0
        for b in range(256):
            if b not in bs:
                bs.append(b); cs.append(256 + n); n += 1
        _BYTE_DECODER = {chr(c): b for b, c in zip(bs, cs)}
    return _BYTE_DECODER


def token_byte_tables(tokenizer_json):
    """Per-token (UTF-8 byte count, UTF-8 lead-byte count = characters started). Special tokens count 0
    (the separator EOS is not text). Used for bits-per-byte / bits-per-character."""
    vocab = json.loads(Path(tokenizer_json).read_text())["model"]["vocab"]
    special = {t["content"] for t in json.loads(Path(tokenizer_json).read_text()).get("added_tokens", [])}
    dec = _byte_decoder()
    nb = np.zeros(VOCAB, np.int64); nc = np.zeros(VOCAB, np.int64)
    for s, i in vocab.items():
        if s in special:
            continue
        b = bytes(dec[ch] for ch in s)
        nb[i] = len(b); nc[i] = sum(1 for x in b if (x & 0xC0) != 0x80)
    return nb, nc
