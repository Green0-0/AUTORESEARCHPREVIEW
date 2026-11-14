# t09_tokenizer_transfer — solution contract

The harness owns the network (GPT-2 small, `harness/t09lib.py` `GPT2`) and the evaluation. You supply one
function that adapts the model to the fixed 32k Japanese tokenizer, in place.

## Fixed network (harness/t09lib.py `GPT2`, do not copy or modify)
```
idx (B, T) int64, T <= 1024
x = wte(idx) + wpe(arange(T))          # wte: Embedding(32000, 768)  <- new, vocab-dependent
x = h[0..11](x)                        # 12 pre-LN blocks: ln_1, attn.c_attn (768->2304), attn.c_proj, ln_2,
                                       #   mlp.c_fc (768->3072), mlp.c_proj (3072->768); GELU(tanh); causal SDPA
logits = lm_head(ln_f(x))              # lm_head: Linear(768, 32000, bias=False)  <- new, vocab-dependent
```
`model(idx)` returns logits (B, T, 32000); `model.hidden(idx)` returns the post-`ln_f` hidden states.
`model.tie_weights()` ties `lm_head.weight` to `wte.weight` (optional). No dropout.

When `adapt` is called, `wpe`, `h.*` and `ln_f` hold the **pretrained GPT-2 small weights**; `wte` and
`lm_head` are fresh N(0, 0.02) draws (untied). Reinitialize them however you like.

## The file you must provide in `solution/`
### `adapt.py`
```python
def adapt(model: GPT2, ctx: dict) -> None
```
Train `model` **in place**. The harness saves `model.state_dict()` when `adapt` returns, and evaluation rebuilds a
fresh `GPT2(32000)` in another process and loads it with `strict=True`. Only the weights carry over, so all
learned state must live in the model's parameters/buffers (no new modules, no forward changes).

`ctx` keys:
| key | value |
| --- | --- |
| `mode` | `"full"`, `"smoke"` or `"cpu-smoke"` |
| `device` | `torch.device` (cuda, or cpu in cpu-smoke) |
| `time_budget_s`, `deadline` | 900 s (full), 45 s (smoke), 40 s (cpu-smoke); `deadline` is a `time.time()` value |
| `seed` | int (`AR_SEED`) |
| `vocab_size`, `n_ctx`, `eos_id` | 32000, 1024, 0 (`<|endoftext|>`) |
| `train_ids` | int32 tensor **on the device**: the 200k training rows tokenized with the fixed tokenizer, each doc followed by EOS, concatenated (331,015,675 tokens, 1.27 GB as int32; the first 2M in cpu-smoke) |
| `train_doc_offsets` | int64 CPU tensor (n_docs+1,): doc i = `train_ids[off[i]:off[i+1]]` |
| `gpt2_wte` | float32 (50257, 768) tensor on the device: the original GPT-2 token embedding (= its tied LM head) |
| `build_gpt2_reference` | callable `() -> GPT2(50257)`: a fresh copy of the full original GPT-2 small on the device (e.g. as a teacher) |
| `tokenizer_path` | the fixed 32k tokenizer (`tokenizers` JSON, byte-level BPE, same byte-to-unicode alphabet as GPT-2). **Read only.** |
| `gpt2_tokenizer_path` | GPT-2's own `tokenizer.json` (read only) |
| `train_docs_path` | raw training text, JSONL with `{"id","title","text"}` per line, same order as `train_ids` (1.6 GB, 626M characters; read it only if you need it, it counts toward the budget) |
| `scratch_dir` | a writable directory for temporary files (deleted after adaptation). `TMPDIR`, `TRITON_CACHE_DIR` and `TORCHINDUCTOR_CACHE_DIR` point into it, so `torch.compile` works. |

Libraries available: `torch`, `numpy`, `tokenizers` (Hugging Face tokenizers; load files with
`Tokenizer.from_file(ctx[...])`), and the Python standard library. Your `solution/` dir is on `sys.path`, so helper
modules there can be imported (don't name one `t09lib`).

### Budget
The whole `adapt()` call — initialization, preprocessing, every training step — must finish within
`time_budget_s` (+10 s grace), otherwise the run is **invalid**. The process (data load + adapt + save) is hard-killed at 18 min. Stop on
`ctx["deadline"]` with a safety margin. Loading the data/weights into `ctx` happens before the timer starts.
Smoke modes only need to show the code runs end to end (a few steps).

### Allowed
Any embedding / LM-head initialization computed from the provided GPT-2 weights, tokenizers and training text;
untying or tying the head; freezing / unfreezing schedules; any optimizer, schedule, batch size, sequence length,
loss (including distillation from `build_gpt2_reference()` on the training text, re-tokenized with GPT-2's
tokenizer); AMP; `torch.compile`; a validation split carved from `train_ids`; temporary extra modules or copies
during training, as long as only `model`'s final weights are what you hand back.

### Not allowed
- Changing the architecture: adding/removing/replacing submodules of `model`, changing any shape, hooks left on
  the model after `adapt` returns. (The evaluator only loads weights into its own skeleton anyway.)
- Changing the tokenizer or evaluating under another one. The tokenizer file's sha256 is checked.
- Any data other than the provided training set; any pretrained weights other than the provided GPT-2 small
  (`ctx["gpt2_wte"]`, `ctx["build_gpt2_reference"]`, the initial `model`); network access (there is none);
  reading any file other than the `ctx` paths and your own `scratch_dir`; writing anywhere except `scratch_dir`.

## What the harness checks (`metric.json` → `constraint_report`)
- `train.within_budget`: `adapt()` wall time ≤ budget + 10 s.
- `structure` / `arch_ok`: after `adapt`, class `GPT2`, identical state_dict keys and shapes (GPT-2 small with a
  32k vocab), same module count, no hooks; the artifact loads strictly into a fresh skeleton.
- `tokenizer_sha_ok`: `tokenizer.json` unchanged (before/after adaptation and at eval).
- all weights finite; held-out loss finite; `deterministic`: re-scoring a batch gives the same loss.
- `scan`: static scan of the solution (`harness/check.py`). Hard violations (network, hub / `from_pretrained`,
  other models or datasets, paths into the data tree or private data, weight files shipped in `solution/`)
  invalidate the run. Review flags (file reads, `transformers` import, subprocess) go to the reviewer.

**Metric:** `heldout_ppl_per_token` (**lower is better**) = exp(mean NLL) over 2,048 fixed windows of the held-out
token stream (1,048,576 scored tokens): each window is 1,025 tokens, the model sees the first 1,024, and the
last 512 next-token predictions are scored (so every scored token has ≥ 512 tokens of context). fp32, TF32 off,
`model.eval()`. `constraint_report` also gives `bits_per_byte` and `bits_per_char` (tokenizer-independent; EOS
contributes loss but no bytes) and `rel_change_from_gpt2` (how far the body moved). Any violation makes the run
`valid: false`; `value` is still reported (1e9 if the eval itself failed).

## Running
```
harness/run.sh <solution_dir> <out_dir> [--smoke | --cpu-smoke]
```
`--cpu-smoke`: CPU, 2M training tokens, 40 s adaptation, 6 eval windows, < 3 min (numbers are meaningless).
`--smoke`: GPU, 45 s adaptation, 128 eval windows. No flag: full mode.
