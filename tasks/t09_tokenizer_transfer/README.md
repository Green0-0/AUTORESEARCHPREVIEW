# t09_tokenizer_transfer: Tokenizer Transfer (dataset problem #9, Type I, test split)

Shortened proxy of problem #9 (original: 4 h on 1 A5000). GPT-2 small (124M, English) is adapted to ONE fixed
32k-vocab byte-level BPE tokenizer trained on Japanese Wikipedia. The metric is held-out per-token perplexity
under that tokenizer, so **lower is better** (`higher_is_better: false`). Budget: a 900 s `adapt()` call on 1 A5000,
covering everything the solution does. Replaces t02 (self-regularization), which had no headroom.

## Layout
- `TASK.md`: the prompt the pipeline sees. `task.yaml`: metadata (floor measured, ref null).
- `prepare_data.py`: idempotent, needs network and pyarrow (`uv run --no-project --python $AR_ROOT/venv/bin/python --with pyarrow python prepare_data.py`). It takes about 7 min on 16 cores and peaks at 38 GB RAM. It writes:
  - `$AR_DATA/t09_tokenizer_transfer/`:
    - `gpt2/`: `gpt2_state.pt`, which is HF `openai-community/gpt2` converted to the harness naming with Conv1D weights transposed, plus GPT-2's `tokenizer.json`, `vocab.json`, `merges.txt` and `config.json`;
    - `tokenizer.json`: the fixed 32k tokenizer;
    - `train_ids.npy`: uint16, 331,015,675 tokens;
    - `train_doc_offsets.npy`;
    - `train_docs.jsonl`: the raw training text, 1.6 GB;
    - `meta.json`: seeds, shard names, every sha256.
  - `$AR_ROOT/private/t09_tokenizer_transfer/`: `heldout_ids.npy` (8,261,164 tokens), `eval_windows.npy` (int32, 2048×1025) and `heldout_wiki_ids.npy`. These are **.npy only, no raw text**, so `scripts/make_decoy_private.py` can fake them. A hash mismatch on re-run aborts the script; it never re-draws the data silently.
- `harness/t09lib.py`: holds the following.
  - The harness-owned `GPT2` skeleton. It is pure torch with SDPA and matches HF `GPT2LMHeadModel` logits to 7e-5 (checked on CPU).
  - The mode table, data loading, structure checks, and the byte/char tables used for bits-per-byte and bits-per-char.
- `harness/train_wrapper.py`: process 1. It loads the training stream onto the GPU as int32 and builds the skeleton with the pretrained body and fresh N(0, 0.02) 32k `wte`/`lm_head`. It then runs `adapt(model, ctx)` under a timer, checks the structure and the tokenizer hash, and saves `artifact/model.pt` in fp32.
- `harness/evaluate.py`: process 2. It rebuilds the skeleton, loads the weights with `strict=True` and runs these checks: shapes, finiteness, tokenizer sha and determinism (re-scores one batch). It then scores the held-out windows in fp32 with TF32 off.
- `harness/check.py`: static scan with hard violations and review flags.
- `harness/run.sh <solution_dir> <out_dir> [--smoke|--cpu-smoke]`: runs offline (`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`).
  - Hard caps: the adaptation process gets 1080 s (full), 150 s (smoke) or 120 s (cpu-smoke). The eval gets 300 s or 120 s.
  - `TMPDIR`, `TRITON_CACHE_DIR`, `TORCHINDUCTOR_CACHE_DIR` and `XDG_CACHE_HOME` point into `out/scratch`, which is writable in the jail and is deleted after adaptation.
- `harness/validate_gpu.sbatch`: GPU validation **inside `realjudge/jail.sh` harness mode** (read-only filesystem, no network), with `nvidia-smi dmon` logging.
- `starter/adapt.py`: naive transfer.
  - New embeddings are random N(0, 0.02) and the head is tied. The full model is trained on random 1024-token windows.
  - AdamW: lr 6e-4, betas (0.9, 0.95), wd 0.1 on 2-D weights, clip 1.0, batch 16×1024, bf16 autocast.
  - The LR schedule is time-based: 3% warmup, then cosine decay to 10%. It stops 15 s before the deadline.

## Data
- **Source:** `wikimedia/wikipedia` `20231101.ja`. Only the first 3 of 15 parquet shards are downloaded, 1.25 GB in total (612 + 383 + 305 MB). Shard 0 alone has only 92,632 rows, too few for a 205k-row pool, so the first-shard-only option was not possible.
- **Split:** the pool has 277,895 rows. One permutation with seed 909 gives 5,000 held-out rows, then 200,000 training rows. The two sets are disjoint by index and by wiki id. The training set holds 626M characters and the held-out set 15.6M.
- **Tokenizer:**
  - Type: byte-level BPE, vocab 32,000 (id 0 = `<|endoftext|>`). The pre-tokenizer is `UnicodeScripts` (which splits kanji, kana and latin runs) followed by GPT-2's ByteLevel regex.
  - It is trained on a fixed sample of 50,000 of the 200k training rows (seed 9091). Training on all 200k rows was OOM-killed at 110 GB in `tokenizers`.
  - It averages 1.89 characters per token. It uses the same byte-to-unicode alphabet as GPT-2, so each new token string can be re-tokenized with GPT-2's BPE directly; on average a new token splits into 4.08 GPT-2 tokens.
- **GPT-2 download:** `model.safetensors` (548 MB) plus the tokenizer files.
- **Eval windows:**
  - 2048 windows of 1,025 tokens, starting on a 512-token grid sampled without replacement (seed 9090).
  - The model sees 1,024 tokens and the last 512 predictions are scored, which gives a context of 1024, a stride of 512 and 1,048,576 scored tokens.
  - Smoke uses an evenly spread subset of 128 windows and cpu-smoke a subset of 6.
  - Bits per byte and bits per character divide the total NLL by the scored tokens' UTF-8 byte counts and character-start counts. EOS adds loss but no bytes.

## Interface (details in harness/CONTRACT.md)
`adapt.py: adapt(model, ctx)` trains the harness `GPT2(32000)` in place. When it is called, `wpe`, `h.*` and `ln_f` hold the pretrained GPT-2 weights.

`ctx` gives:
- `train_ids`: int32 on the GPU, 1.27 GB;
- `train_doc_offsets`;
- `gpt2_wte`: the original 50257×768 embedding;
- `build_gpt2_reference()`: the full original GPT-2, usable as a teacher;
- `tokenizer_path`, `gpt2_tokenizer_path`, `train_docs_path` and `scratch_dir`;
- `deadline`, `seed` and `mode`.

The solution may use torch, numpy, `tokenizers` and the standard library.

## Measured
All runs were on RTX A5000s (fitz-09/10/11/18), through `validate_gpu.sbatch`, inside the harness jail. Outputs are in `$AR_ROOT/harness_runs/t09_tokenizer_transfer/`.

| run | ppl (metric) | bits/char | adapt() wall | total wall (load / eval) | GPU SM util |
| --- | --- | --- | --- | --- | --- |
| **full, starter, seed 0** | **51.616** | 3.023 | 885.5 s (3470 steps, 56.9M tok) | 940.9 s (2.4 / 43.9) | 99.5% mean non-idle; 185/189 dmon samples ≥ 80% |
| **full, starter, seed 1** | **53.940** | 3.056 | 885.3 s | 940.8 s | 99.7%; 184/189 |
| **full, starter, seed 2** | **52.010** | 3.028 | 885.4 s | 943.1 s | 100%; 184/190 |
| full, probe, seed 0 | 41.304 | 2.852 | 885.6 s (649 + 2928 steps) | 943.4 s | 99.9%; 185/190 |
| full, probe, seed 1 | 41.224 | 2.850 | 885.5 s | 994.3 s (53.7 s NFS load) | 99.2%; 182/200 |
| full, probe, seed 2 | 41.400 | 2.854 | 885.5 s | 995.3 s | 99.9%; 183/200 |
| smoke, starter | 3887.6 | 6.37 | 30.5 s (106 steps) | 54.1 s | 100% during training |
| smoke, probe | 334.2 | 4.48 | 30.4 s | 54.5 s | 100% during training |
| cpu-smoke, starter / probe (4 cores, harness jail) | 7826 / 7584 | | 35 s | 64 s / 60 s | n/a |

- The dmon samples below 80% are CUDA start-up and eval.
- Throughput is about 65k tokens/s (bf16, no compile), and peak memory is 14.8 GB including the 1.27 GB token stream.
- **Floor = 52.52 ± 1.24** (mean ± sd over starter seeds 0-2), recorded in `task.yaml`.

**Headroom probe** (`$AR_ROOT/t09_dev/probe_solution/adapt.py`, outside starter/). It uses the starter's recipe with two changes:
- FVT init: each new embedding is the mean of the GPT-2 embeddings of its GPT-2 re-tokenization. The init takes 1.1 s.
- The first 15% of the time trains only the tied embedding/head, at lr 2e-3 with the body frozen. The rest is a full fine-tune.

Result: **41.31 ± 0.09**, so it lowers perplexity by 11.2 (−21%). That gap is 9.0× the starter's seed sd and 12.7× the pooled sd, so the task has headroom. Bits per char drop from 3.04 to 2.85.

The starter's seed sd (1.24) is much larger than the probe's (0.09). The random embedding init seems to make the starter less stable, so the floor is the noisier anchor.

## Constraint-check tests (cpu-smoke, harness jail)
| deliberately bad / borderline solution | result |
| --- | --- |
| adds `model.extra = Linear` | **valid=false**: structure (extra keys, module count) + strict load fails |
| replaces `wte` with a 32,001-row embedding | **valid=false**: shape check + strict load fails |
| `AutoModelForCausalLM.from_pretrained(...)` | **valid=false**: scan hard violation. At runtime the hub call also fails (no network in the jail) |
| `adapt()` sleeps past the budget | **valid=false**: within_budget (55 s > 40 + 10) |
| `np.load($AR_ROOT/private/.../eval_windows.npy)` | **valid=false**: scan hard violation (the read itself succeeds in harness mode, see caveats) |
| writes a NaN into one block weight | **valid=false**: non-finite weights / loss |
| untied head, no training (control) | valid=true (no false positive) |

The starter and the probe also run valid under `JAIL_MODE=claude` (decoy private data, so the ppl there is meaningless).

## Caveats
- The tokenizer was trained on 50k of the 200k training rows, not all of them, because of memory. It is still training-data-only and fixed for every solution.
- **Private data is visible in harness mode.** The harness eval needs it, so this applies to all tasks. A solution that reads `$AR_ROOT/private` through an obfuscated path would escape the static scan. Only the reviewer would catch it.
- The held-out token windows exist only as `.npy` files, and no raw held-out text is stored anywhere.
- The data loading before the timer (1.27 GB train stream + 0.5 GB GPT-2 weights from NFS) took between 2 and 54 s, depending on the node's cache. The hard cap of 1080 s leaves room for that.
- `artifact/model.pt` is about 540 MB per run (fp32, head stored untied).
- Seed noise is dominated by training randomness. The eval is deterministic, and every run sees the same windows.
- Methods that change the tokenizer cannot help, by construction: the eval uses pre-tokenized ids, and the sha is checked.
