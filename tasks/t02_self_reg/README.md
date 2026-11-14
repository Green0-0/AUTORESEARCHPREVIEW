# t02_self_reg: Self-Regularizing Network (dataset problem #2, Type II, test split)

Shortened CIFAR-100 proxy of problem #2 (original: ImageNet-1k, 5 imgs/class, 4 h). Training data is a fixed
500-image subset of CIFAR-100 train (5 per class, seed 2002). The metric is top-5 accuracy on the full 10k
CIFAR-100 test set (top-1 is reported in `constraint_report`). **Training-only regularization is banned**: the
model must process data at training exactly as at test. Budget: a 900 s `train()` call on 1 A5000.
Replaces t19 (data ordering), which had no headroom.

## Layout
- `TASK.md`: the prompt the pipeline sees. `task.yaml`: metadata (floor measured, ref null).
- `prepare_data.py`: idempotent. Writes `$AR_DATA/t02_self_reg/train_{x,y,idx}.npy` (500 images, uint8 NCHW, fine labels,
  original indices) and `meta.json` (seed and sha256; a hash mismatch aborts instead of re-drawing). It also writes
  `$AR_ROOT/private/t02_self_reg/test_{x,y}.npy` (full 10k test set, read only by `evaluate.py`).
- `harness/CONTRACT.md`: the solution interface. Claude reads it.
- `harness/t02lib.py`: mode table, data loading, the training-input guard and the post-training checks.
- `harness/train_wrapper.py`: process 1. Loads the 500 images onto the GPU as float32 [0,1], builds `build_model()`, installs the guard,
  calls the solution's `train(model, x, y, ctx)` under a timer, saves `artifact/model.pt` (state_dict) and `artifact/code/` (copy of the solution).
- `harness/evaluate.py`: process 2. Rebuilds the model from `artifact/code/model.py`, loads it strictly, runs the checks, then computes top-5/top-1 in fp32 (TF32 off). Writes `metric.json`.
- `harness/check.py`: static scan (hard violations plus review flags).
- `harness/run.sh <solution_dir> <out_dir> [--smoke|--cpu-smoke]`. Caps: training process 960 s (full) / 120 s (smoke, cpu-smoke); eval 300 s / 120 s.
- `harness/validate_gpu.sbatch`: GPU validation with `nvidia-smi dmon` (smoke, or `--array=0-2` for full seeds).
- `starter/`: ResNet-9-style CNN at 2× width (128-256-512-1024), **GroupNorm** (groups of 16 channels), no dropout/BN/augmentation, 26.4M params. Plain CE,
  SGD nesterov lr 0.1, wd 5e-4 (not on norm/bias), 10% linear warmup then linear decay, batch 250, 3000 steps (1500 epochs), bf16 autocast, data on GPU.

## Interface and how the rule is enforced
The solution provides `model.py: build_model()` and `train.py: train(model, x, y, ctx)`. **The harness owns the
training process** and hands `train()` the model and the raw images, so it can watch every forward pass:

1. **Runtime guard during `train()`** (hard; `t02lib.InputGuard`):
   - a forward pre-hook on the root model checks that every **grad-enabled** call is `model(x)` with one float32 (B,3,32,32) tensor and no kwargs;
   - every row of `x` must be a raw training image. This is checked with a float64 random-projection fingerprint against the 500 images: one tiny matmul + cdist per call, no host sync;
   - a global module pre-hook counts grad-enabled calls of `model`'s submodules made outside `model.forward` (bypass, e.g. `model.net(aug(x))`).
     It also counts grad-enabled calls of foreign parametrized modules (a separate teacher), which are a review flag only.
   - So augmentation, mixup or noise in `train.py`, extra `train=True` args and submodule bypasses are all caught mechanically.
     `no_grad` forwards (monitoring, EMA teachers, pseudo-targets) are not restricted.
2. **Post-training checks** (hard, eval process, fp32, on a deepcopy, 256 test images):
   - determinism under a fixed seed;
   - batch independence: same seed and batch size, second half replaced by other images, first-half logits must not change;
   - **train-mode + grad vs eval-mode + no_grad under the same RNG seed** must match within rel 1e-3. This runs last, because train mode can mutate BN buffers;
   - logits shape (B,100) and finite.
3. **Static scan** (`check.py`):
   - Hard: test / private / raw-CIFAR paths, `AR_DATA`/`AR_ROOT`, torchvision datasets / hub / pretrained / network, hook-registry access, augmentation libraries imported in `train.py`.
   - Review flags in `train.py`: augmentation-like ops, mixing identifiers, random tensors, dropout/BN, functional forwards / `.forward(`, file access, deepcopy/EMA/teacher.
   - Review flags in `model.py`: `self.training` / `is_grad_enabled` branches, dropout/BN-like layers, randomness, progress-dependent state.
   - Docstring lines are skipped.

**Randomness at test time:** the evaluator seeds `random`/`numpy`/`torch` once (EVAL_SEED 20020, independent of
`AR_SEED`), then makes one forward per batch of 500 in a fixed order. Stochastic in-forward augmentation is thus
applied at test exactly as in training, and is reproducible. Averaging over K views has to be built into `forward`,
so the same K is used in training by construction. `constraint_report.checks.forward_stochastic` tells the reviewer whether the forward is random.

**Line drawn (stated in TASK.md):** banned = anything that makes the train-time forward differ from the test-time
forward (dropout, stochastic depth, batch-stat BN, train-only augmentation/mixup/cutout/noise, mode/grad branches).
Free = architecture, init, optimizer, schedule, weight decay, loss (incl. label smoothing), EMA, self-distillation on
raw images. Every loss/optimizer regularizer is "training-only" in a trivial sense, so the task's own definition
("the manner in which the model processes data") is taken as the operative one.

## Measured
All runs on RTX A5000 (fitz-14/15/16), sbatch, `nvidia-smi dmon -s um -d 5`. Outputs are in `$AR_ROOT/harness_runs/t02_self_reg/`.

| run | top-5 | top-1 | train() wall | total wall | GPU SM util |
| --- | --- | --- | --- | --- | --- |
| **full, starter, seed 0** | **0.3095** | 0.1246 | 218.2 s | 227.6 s (eval 3.9 s) | 99.4% mean over non-idle samples; 91% of all samples ≥ 80% (the rest are CUDA startup/eval) |
| **full, starter, seed 1** | **0.3098** | 0.1205 | 218.8 s | 228.2 s | 97.9% non-idle mean, 91% ≥ 80% |
| **full, starter, seed 2** | **0.3133** | 0.1272 | 217.4 s | 227.3 s | 97.6% non-idle mean, 91% ≥ 80% |
| smoke, starter, seed 0 | 0.2701 | 0.104 | 22.2 s | 31.9 s | 100% during training |
| cpu-smoke, starter (4 cores, 100 train / 500 test images) | 0.05 | 0.008 | 18.4 s | 38 s | n/a |

**Floor = 0.3109 ± 0.0021 (sd over seeds 0-2)**, recorded in `task.yaml`.

Calibration (job 13474288, single runs, bf16 eval, top-5):
| config | top-5 |
| --- | --- |
| w64, 1000 steps | 0.290 |
| w64, 3000 steps | 0.295 |
| w64, bs 100 | 0.300 |
| w64, wd 5e-3 | 0.236 |
| w64, lr .05 + wd 5e-3 | 0.279 |
| w64, bs 500 + lr .2 | 0.140 |
| w64, 8000 steps + wd 5e-3 | 0.208 |
| **w128, 3000 steps** | **0.312** |

Without augmentation, training loss hits ~0 within ~1000 steps, and more steps or stronger decay don't help.
That is why the starter stops at ~3.6 min of the 15. The remaining budget is headroom for in-forward augmentation,
multi-view averaging, ensembles inside the model and so on.

## Constraint-check tests (cpu-smoke, job 13480634)
| deliberately bad / borderline solution | result |
| --- | --- |
| `nn.Dropout(0.3)` before the classifier | **valid=false**: train/eval rel diff 1.19 (scan also flags the Dropout) |
| `train.py` random h-flip of `x[idx]` before `model()` | **valid=false**: guard, 159/300 training inputs not raw images (scan flags flip + rand) |
| `train.py` calls `model.net(flipped)` directly | **valid=false**: guard, 258 submodule calls outside forward, no `model(x)` call |
| BatchNorm2d (running stats) | **valid=false**: train/eval rel diff 40.9 |
| BatchNorm2d(track_running_stats=False), batch stats at train *and* test | **valid=false**: batch independence rel 0.045 |
| mixup inside forward across the batch (`x.roll(1,0)`) | **valid=false**: batch independence rel 0.066 |
| noise only when `torch.is_grad_enabled()` | **valid=false**: train/eval rel diff 0.146 (scan flags the branch) |
| `model(x, train=True)` adds noise | **valid=false**: guard, 6 calls with extra args/kwargs |
| random h-flip *inside* forward (applied at test too) | **valid=true** (no false positive); `forward_stochastic` review flag |

## Caveats
- **Isolation is by convention.** The guard relies on module hooks, and solution code runs in the same process. A malicious `train.py` could bypass the hooks without touching the registries
  (e.g. `torch.func.functional_call`, or manual `F.conv2d` on `model`'s parameters with augmented inputs). The scan flags these patterns, and the reviewer must read `train.py`.
- `torch.compile` of the top-level model or wrapping it is disallowed by the contract (it would hide calls from the root hook); compiling submodules inside `build_model()` is fine.
- The fingerprint check requires float32 inputs equal to the raw rows; a solution that feeds `x.half()` is rejected by design (autocast is fine).
- The test-time RNG seed is fixed and public, so in principle a model could special-case it. That would be absurd; the reviewer would catch it.
- cpu-smoke uses 1 image/class and 500 test images, so its numbers mean nothing.
- Wall time is measured on whichever A5000 node runs the job; node-to-node throughput varies by a few percent.
