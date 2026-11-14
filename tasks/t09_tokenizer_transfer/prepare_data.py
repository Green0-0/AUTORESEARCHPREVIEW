"""Idempotent data prep for t09_tokenizer_transfer (dataset problem #9, Tokenizer Transfer).

Needs network (run once, outside the jail) and pyarrow, which the main venv lacks:
    source env.sh
    uv run --no-project --python $AR_ROOT/venv/bin/python --with pyarrow python tasks/t09_tokenizer_transfer/prepare_data.py
(~16 cores / 64 GB RAM recommended; about 20-30 min, dominated by tokenizer training and encoding.)

Steps
 1. GPT-2 small (HF `openai-community/gpt2` = `gpt2`: config, model.safetensors, tokenizer files) -> $HF_HOME, then converted to the
    harness's own parameter naming (Conv1D weights transposed to nn.Linear layout) and written as
        $AR_DATA/t09_tokenizer_transfer/gpt2/gpt2_state.pt   + tokenizer.json, vocab.json, merges.txt, config.json
 2. Japanese Wikipedia (`wikimedia/wikipedia`, config 20231101.ja): the first N_SHARDS parquet shards only
    (no full-dump download). Rows are pooled in shard order, permuted once with SPLIT_SEED; the first 5,000 rows
    of the permutation are the held-out set, the next 200,000 the training set (disjoint by construction).
 3. ONE fixed 32k-vocab byte-level BPE tokenizer is trained on a fixed 50k-row sample of the training rows
    (training on all 200k needs >110 GB RAM in `tokenizers`) ->
        $AR_DATA/t09_tokenizer_transfer/tokenizer.json         (solutions may not change it; sha256 in meta.json)
 4. Pre-tokenized data (doc tokens followed by <|endoftext|> = id 0, concatenated):
        $AR_DATA/.../train_ids.npy          uint16 (N_train_tokens,)
        $AR_DATA/.../train_doc_offsets.npy  int64  (200001,)  doc i = train_ids[off[i]:off[i+1]] (incl. its EOS)
        $AR_DATA/.../train_docs.jsonl       raw training text, one {"id","title","text"} per line (same order)
        $AR_DATA/.../meta.json              seeds, shard files, hashes, sizes
        $AR_ROOT/private/t09_tokenizer_transfer/heldout_ids.npy     uint16 held-out token stream (same format)
        $AR_ROOT/private/t09_tokenizer_transfer/eval_windows.npy    int32 (N_WIN, CTX+1) fixed scoring windows
        $AR_ROOT/private/t09_tokenizer_transfer/heldout_wiki_ids.npy int64 wiki ids of the held-out rows
    Private files are .npy only (no raw text), so scripts/make_decoy_private.py can fake them.
Re-running is a no-op when files exist and hashes match; a mismatch is an error (never silently re-drawn).
"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

TASK = "t09_tokenizer_transfer"
DATASET, CONFIG = "wikimedia/wikipedia", "20231101.ja"
N_SHARDS = 3                     # first 3 of 15 shards (~1.25 GB parquet) hold enough rows for 205k
SPLIT_SEED = 909
N_TRAIN, N_HELD = 200_000, 5_000
VOCAB = 32_000
TOK_ROWS, TOK_SEED = 50_000, 9091   # BPE trained on a fixed 50k-row sample of the training rows (all 200k OOMs >110 GB)
EOS = "<|endoftext|>"
CTX, SCORE = 1024, 512           # eval windows: 1024-token context, last 512 positions scored (stride 512)
N_WIN = 2048                     # 2048 windows x 512 = 1,048,576 scored held-out tokens
WIN_SEED = 9090


def _sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def _sha_arr(*arrs):
    h = hashlib.sha256()
    for a in arrs:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def _save(arr, dst: Path):
    tmp = dst.with_name(dst.stem + ".tmp.npy")
    np.save(tmp, arr)
    os.replace(tmp, dst)


def log(*a):
    print(f"[prepare {time.strftime('%H:%M:%S')}]", *a, flush=True)


# ----------------------------------------------------------------------------------------------- GPT-2
def prep_gpt2(pub: Path):
    import torch
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file
    out = pub / "gpt2"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "gpt2_state.pt").exists() and (out / "tokenizer.json").exists():
        log("gpt2: exists"); return
    src = Path(snapshot_download("openai-community/gpt2", allow_patterns=["config.json", "model.safetensors", "tokenizer.json",
                                                          "vocab.json", "merges.txt"]))
    hf = load_file(str(src / "model.safetensors"))
    sd = {}
    for k, v in hf.items():
        k = k.removeprefix("transformer.")
        if k.endswith(".attn.bias") or k.endswith(".attn.masked_bias"):
            continue
        if any(k.endswith(s) for s in ("c_attn.weight", "c_proj.weight", "c_fc.weight")):
            v = v.t().contiguous()            # HF Conv1D (in, out) -> nn.Linear (out, in)
        sd[k] = v.float().contiguous()
    assert sd["wte.weight"].shape == (50257, 768) and len([k for k in sd if k.startswith("h.")]) == 12 * 12
    torch.save(sd, out / "gpt2_state.tmp")
    os.replace(out / "gpt2_state.tmp", out / "gpt2_state.pt")
    for f in ("config.json", "tokenizer.json", "vocab.json", "merges.txt"):
        (out / f).write_bytes((src / f).read_bytes())
    log(f"gpt2: wrote {out} ({len(sd)} tensors)")


# ----------------------------------------------------------------------------------------------- data
def load_rows():
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    ids, titles, texts, files = [], [], [], []
    for s in range(N_SHARDS):
        fn = f"{CONFIG}/train-{s:05d}-of-00015.parquet"
        p = hf_hub_download(DATASET, fn, repo_type="dataset")
        t = pq.read_table(p, columns=["id", "title", "text"])
        ids += [int(x) for x in t.column("id").to_pylist()]
        titles += t.column("title").to_pylist()
        texts += t.column("text").to_pylist()
        files.append(fn)
        log(f"shard {fn}: {t.num_rows} rows (pool {len(ids)})")
    assert len(ids) >= N_TRAIN + N_HELD, f"only {len(ids)} rows in {N_SHARDS} shards"
    return np.asarray(ids, dtype=np.int64), titles, texts, files


def build_tokenizer(train_texts):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    tok = Tokenizer(models.BPE())
    # split by Unicode script (kanji / hiragana / katakana / latin runs), then GPT-2 byte-level regex + bytes
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.UnicodeScripts(),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)])
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=VOCAB, min_frequency=2, special_tokens=[EOS],
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
    tok.train_from_iterator(train_texts, trainer=trainer, length=len(train_texts))
    assert tok.get_vocab_size() == VOCAB and tok.token_to_id(EOS) == 0, tok.get_vocab_size()
    return tok


def encode_stream(tok, texts, bs=2000):
    eos = tok.token_to_id(EOS)
    parts, lens = [], []
    for i in range(0, len(texts), bs):
        for e in tok.encode_batch(texts[i:i + bs], add_special_tokens=False):
            a = np.asarray(e.ids + [eos], dtype=np.uint16)
            parts.append(a); lens.append(len(a))
    off = np.zeros(len(lens) + 1, dtype=np.int64)
    off[1:] = np.cumsum(lens)
    return np.concatenate(parts), off


def eval_windows(held):
    """N_WIN windows of CTX+1 tokens; starts on a 512-grid, drawn without replacement (fixed seed), so the
    scored halves never overlap. The scored part of window w is targets w[1:][SCORE-1+1:] ... see harness."""
    n_slots = (len(held) - CTX - 1) // SCORE
    assert n_slots >= N_WIN, (len(held), n_slots)
    rng = np.random.default_rng(WIN_SEED)
    starts = np.sort(rng.choice(n_slots, N_WIN, replace=False)) * SCORE
    return np.stack([held[s:s + CTX + 1] for s in starts]).astype(np.int32), starts


def main():
    ar_data = Path(os.environ["AR_DATA"])
    ar_root = Path(os.environ.get("AR_ROOT", ar_data.parent))
    pub, priv = ar_data / TASK, ar_root / "private" / TASK
    pub.mkdir(parents=True, exist_ok=True); priv.mkdir(parents=True, exist_ok=True)
    prep_gpt2(pub)

    meta_p = pub / "meta.json"
    need = ["tokenizer.json", "train_ids.npy", "train_doc_offsets.npy", "train_docs.jsonl"]
    need_priv = ["heldout_ids.npy", "eval_windows.npy", "heldout_wiki_ids.npy"]
    if meta_p.exists() and all((pub / f).exists() for f in need) and all((priv / f).exists() for f in need_priv):
        meta = json.loads(meta_p.read_text())
        ok = (_sha_file(pub / "tokenizer.json") == meta["tokenizer_sha256"]
              and _sha_file(pub / "train_ids.npy") == meta["train_ids_file_sha256"])
        if not ok:
            sys.exit(f"[prepare] hash mismatch in {pub}: refusing to overwrite (delete it to rebuild)")
        log(f"data: exists, hashes ok ({pub})")
        os.chmod(priv, 0o700)
        return

    wiki_ids, titles, texts, files = load_rows()
    perm = np.random.default_rng(SPLIT_SEED).permutation(len(wiki_ids))
    held_i, train_i = perm[:N_HELD], perm[N_HELD:N_HELD + N_TRAIN]
    held_i = np.sort(held_i); train_i = np.sort(train_i)          # keep wiki order inside each split
    assert not set(held_i.tolist()) & set(train_i.tolist())
    assert not set(wiki_ids[held_i].tolist()) & set(wiki_ids[train_i].tolist())
    tr_texts = [texts[i] for i in train_i]
    he_texts = [texts[i] for i in held_i]
    log(f"split: train {len(tr_texts)} rows / {sum(map(len, tr_texts)):,} chars; "
        f"held-out {len(he_texts)} rows / {sum(map(len, he_texts)):,} chars")

    t0 = time.time()
    tok_i = np.sort(np.random.default_rng(TOK_SEED).choice(N_TRAIN, TOK_ROWS, replace=False))
    tok = build_tokenizer([tr_texts[i] for i in tok_i])
    tok.save(str(pub / "tokenizer.tmp.json"))
    os.replace(pub / "tokenizer.tmp.json", pub / "tokenizer.json")
    log(f"tokenizer: vocab {tok.get_vocab_size()} trained in {time.time() - t0:.0f}s")

    t0 = time.time()
    tr_ids, tr_off = encode_stream(tok, tr_texts)
    he_ids, he_off = encode_stream(tok, he_texts)
    log(f"encode: train {len(tr_ids):,} tokens, held-out {len(he_ids):,} tokens ({time.time() - t0:.0f}s)")
    win, starts = eval_windows(he_ids)

    with open(pub / "train_docs.tmp", "w", encoding="utf-8") as f:
        for i in train_i:
            f.write(json.dumps({"id": int(wiki_ids[i]), "title": titles[i], "text": texts[i]},
                               ensure_ascii=False) + "\n")
    os.replace(pub / "train_docs.tmp", pub / "train_docs.jsonl")
    _save(tr_ids, pub / "train_ids.npy"); _save(tr_off, pub / "train_doc_offsets.npy")
    _save(he_ids, priv / "heldout_ids.npy"); _save(win, priv / "eval_windows.npy")
    _save(wiki_ids[held_i], priv / "heldout_wiki_ids.npy")
    tr_chars = sum(map(len, tr_texts))
    meta = {
        "task": TASK, "source": f"{DATASET} / {CONFIG}", "shards": files, "pool_rows": int(len(wiki_ids)),
        "split_seed": SPLIT_SEED, "n_train_rows": N_TRAIN, "n_heldout_rows": N_HELD,
        "train_wiki_ids_sha256": _sha_arr(wiki_ids[train_i]), "heldout_wiki_ids_sha256": _sha_arr(wiki_ids[held_i]),
        "tokenizer": {"type": "byte-level BPE (UnicodeScripts + ByteLevel pre-tokenizer)", "vocab_size": VOCAB,
                      "eos": EOS, "eos_id": 0, "trained_on": f"{TOK_ROWS} training rows (seed {TOK_SEED}) of the 200k; never held-out rows",
                      "tok_rows_sha256": _sha_arr(tok_i)},
        "tokenizer_sha256": _sha_file(pub / "tokenizer.json"),
        "gpt2_state_sha256": _sha_file(pub / "gpt2" / "gpt2_state.pt"),
        "n_train_tokens": int(len(tr_ids)), "n_train_chars": int(tr_chars),
        "n_heldout_tokens": int(len(he_ids)), "n_heldout_chars": int(sum(map(len, he_texts))),
        "train_ids_sha256": _sha_arr(tr_ids), "train_ids_file_sha256": _sha_file(pub / "train_ids.npy"),
        "heldout_ids_sha256": _sha_arr(he_ids), "eval_windows_sha256": _sha_arr(win),
        "eval": {"ctx": CTX, "scored_per_window": SCORE, "n_windows": N_WIN, "window_seed": WIN_SEED,
                 "n_scored_tokens": N_WIN * SCORE},
    }
    meta_p.write_text(json.dumps(meta, indent=1))
    os.chmod(priv, 0o700)
    log("done", json.dumps({k: meta[k] for k in ("n_train_tokens", "n_heldout_tokens", "n_train_chars")}))


if __name__ == "__main__":
    main()
