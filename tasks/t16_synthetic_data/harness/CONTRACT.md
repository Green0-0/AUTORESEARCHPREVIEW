# t16_synthetic_data — solution contract

## What you provide
A directory `solution/` containing **`synthesize.py`** (plus any helper modules it imports). The harness runs:

```
python solution/synthesize.py --real <path>/real10.npz --out <out_dir>/artifact \
       --seed <int> --budget-s <seconds> --mode full|smoke|cpu-smoke
```
with the working directory set to `solution/`. Environment: `AR_SEED`, `T16_MODE`, `T16_BUDGET_S`; `AR_DATA` points to a sandbox dir that contains only `real10.npz`. In `cpu-smoke` mode no GPU is visible (`CUDA_VISIBLE_DEVICES=""`), so use `torch.cuda.is_available()` and keep the work tiny in `smoke`/`cpu-smoke` modes.

**Input** `real10.npz`: `x` uint8 `(1000, 32, 32, 3)` (HWC, RGB), `y` int64 `(1000,)` with exactly 10 images per class 0..99, `idx` (original CIFAR train indices; informational).

**Output** — exactly one file, `<out>/synthetic.npz` (`np.savez` or `np.savez_compressed`), with exactly two arrays:
- `x`: shape `(N, 32, 32, 3)` HWC RGB, dtype `uint8` (0..255) **or** `float16`/`float32` with all values finite and in `[0, 1]`.
- `y`: shape `(N,)`, integer dtype, values in `0..99` (**hard labels only**; no soft labels, no per-example weights).
- At most **100 examples per class**. Classes may have fewer (or zero) examples.

Anything else in the artifact dir is ignored for training but counted in `storage_bytes_artifact_dir`. Storage is reported as `storage_bytes_npz` (file size on disk — compression counts in your favour) and `storage_bytes_raw` (`x.nbytes + y.nbytes`).

## Budget
- `full`: 15 min (900 s) wall-clock for `synthesize.py`, on one A5000, hard-killed at the limit. Overrun or a non-zero exit = invalid.
- `smoke`: 60 s, `cpu-smoke`: 90 s. The eval in those modes uses a few training steps, so the metric is meaningless; they only check the plumbing.

## Allowed / not allowed
- Allowed: anything computed from the 1000 given images: augmentation, generative models trained from scratch, dataset distillation / gradient or distribution matching with networks trained from scratch, mixing, selection, etc. You may train your own proxy networks of any architecture (including copies of the visible ones in `harness/archs.py`).
- Not allowed: reading any other data (the rest of CIFAR-100 train, the CIFAR-100 test set, CIFAR-10, anything under `$AR_ROOT`), pretrained weights, network access, reading or importing `$AR_ROOT/private/`, modifying `harness/`.

## Evaluation (fixed; not under your control)
After synthesis, a separate step validates the artifact and trains **three classifiers from scratch** on it: `convnet` and `resnet` (definitions and recipes in `harness/archs.py`) and a third **withheld** architecture you cannot see. Each is trained for a fixed number of SGD steps (batch 512, sampling uniformly with replacement from your set, so the number of steps does not depend on N), with fixed on-GPU augmentation (4-px reflect-pad random crop + horizontal flip), CIFAR-100 mean/std normalisation, bf16 autocast, fixed seeds. Then each is tested on the 10,000 CIFAR-100 test images.

Metric `mean_top1_3arch` = mean test top-1 (fraction in [0,1]) of the three. `constraint_report` also has `per_arch_top1`, `synthesis_wall_s`, the storage fields and the validation `checks`.

## Running
```
harness/run.sh solution <out_dir> --cpu-smoke   # CPU, ~30 s
harness/run.sh solution <out_dir> --smoke       # GPU, < 2 min
harness/run.sh solution <out_dir>               # full
```
Result: `<out_dir>/metric.json`; synthesis stdout/stderr in `<out_dir>/synthesis.log`. `starter/synthesize.py` is a minimal valid example.
