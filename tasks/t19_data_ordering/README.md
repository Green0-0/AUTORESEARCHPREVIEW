# t19_data_ordering: harness notes

Dataset problem #19, "Data Ordering as the Only Control" (Type I, **test** split). Proxy budget: 15 min on 1×A5000 for the whole training process, scheduling included.

## Layout
- `prepare_data.py`: idempotent. Writes `$AR_DATA/t19_data_ordering/train_{x,y}.npy` (50k, uint8 NCHW, fine labels) and `$AR_ROOT/private/t19_data_ordering/test_{x,y}.npy` (10k). Only `eval.py` reads the private test set.
- `harness/common.py`: the fixed recipe (model, augmentation, normalization, mode table).
- `harness/train.py`: the fixed training loop. It calls the solution's `Scheduler` and validates the indices.
- `harness/eval.py`: separate test top-1 eval. Writes `metric.json`.
- `harness/check.py`: static source checks (errors only for a missing `Scheduler`; everything else is reported as warnings for the reviewer).
- `harness/run.sh`: `run.sh <solution_dir> <out_dir> [--smoke|--cpu-smoke]`.
- `starter/scheduler.py`: a uniform random reshuffle each epoch, on the GPU.

## Fixed training (owned by the harness)
| | |
| --- | --- |
| model | ResNet-9 style at 2× width (128-256-512-1024, 2 residual blocks, max-pool head, logits × 0.125), 26.4M params, channels_last, init via `torch.manual_seed(AR_SEED)` |
| optimizer | SGD, nesterov, momentum 0.9, peak lr 0.4, wd 5e-4 (not on BN/bias) |
| schedule | linear warmup over 15% of steps, then linear decay to 0, as a function of step only |
| loss | CE with label smoothing 0.2, mean over batch |
| augmentation | GPU random flip + reflect-pad-4 random crop, own `torch.Generator` (seed+1) |
| precision | bf16 autocast |
| batch / updates | **512 × 6000 steps = 3,072,000 presentations (≈61.4 epochs of 50k)** |
| budget | 900 s wall from `run.sh` start until the model is saved (includes python/CUDA startup and all scheduler time); hard `timeout` at 915 s |

Modes:
- `smoke`: 512 × 300 steps, 110 s budget, about 37 s.
- `cpu-smoke`: 2048 examples, batch 64, 12 steps, 512 test images, 170 s budget, about 71 s on 4 CPU cores (`-p compsci -c 4`).

Interface: see `harness/CONTRACT.md`. In short:
- `Scheduler(ctx)`.
- `next_batch(step)` returns exactly B integer indices in [0, N). Duplicates are allowed.
- Optional `observe(step, indices, per_example_losses, logits)`.
- `ctx.probe(indices, augment=False)` gives eval-mode forward passes of the current model. They are timed and counted.

Total presentations are fixed by construction. The per-example presentation-count histogram (with min/max/gini/never-seen) goes into `train_report.json` and `metric.json` (`constraint_report.histogram`), and the raw counts go into `artifact/presentation_counts.npy`.

Rejection (valid = false, value 0), all tested on cpu-smoke:
- out-of-range numpy index → `ScheduleError: index out of range`;
- batch of 63 instead of 64 → `ScheduleError: batch must have exactly 64 indices`;
- float indices → `ScheduleError: indices must be an integer tensor`;
- a scheduler that sleeps past the budget → `BudgetExceeded`.

Out-of-range GPU indices are clamped for the gather and counted asynchronously, which raises within 50 steps (tested on GPU smoke, see below).

## Measured
All runs were on an RTX A5000 (fitz-12), sbatch, with `nvidia-smi dmon` logged. Logs are in `$AR_ROOT/harness_runs/t19_data_ordering/`.

| run | test top-1 | wall (train process) | notes |
| --- | --- | --- | --- |
| **full, starter, seed 0** | **0.7990** | 517.7 s (8.6 min). Training loop 512 s, about 85 ms/step | valid. Every example shown 61–62×. 382 s headroom left for scheduling |
| smoke, starter, seed 0 | 0.5750 | 36.9 s | |
| smoke, starter, seed 1 | 0.5789 | 37.0 s | smoke seed noise ≈ 0.4 pt |
| smoke, probe-test scheduler (5k-example probe every 100 steps) | 0.5719 | 39.9 s | 15k probed examples cost 3.4 s, about 0.23 ms/example (a full 50k probe ≈ 4.8 s) |
| smoke, GPU out-of-range index at step 7 | 0 | 12.5 s | rejected: `ScheduleError: 1 out-of-range indices returned` |
| cpu-smoke, starter | ~0.02 | 71 s | 4 CPU cores |

GPU util during the full run (dmon at 5 s): 96% mean SM over the whole train process, 99.9% over non-zero samples, and 96% of samples at ≥ 80%. The only 0% samples are CUDA startup and eval.

Superseded calibration: the 1× width ResNet-9 (6.6M params) ran 9800 steps in 303 s and reached 0.7615, which used too little of the budget, so the model was widened to 2×. Benchmark at batch 512 on an A5000: 1×: 29 ms/step, 1.5×: 55 ms/step, 2×: 84 ms/step.

## Caveats
- **Isolation is by convention.** Python can't sandbox the scheduler. It gets copies of the train images and labels and no model handle, but a malicious solution could still reach harness objects via `gc`. `check.py` flags such patterns (`gc.get_objects`, `_h`, `state_dict`, `param_groups`, test/private paths) as warnings, and the Claude review must enforce them.
- Test files under `$AR_ROOT/private` are readable by the same Unix user (chmod 700 only stops other users).
- The per-step `observe` feedback is the training-mode loss on the augmented view, from the forward pass *before* that step's update.
- The `probe` uses eval-mode BN (running stats) and its own augmentation RNG, so probing never perturbs the training RNG stream or BN stats. With the same scheduler and seed, training is reproducible up to cuDNN nondeterminism.
- About 61 epochs is a fairly converged regime. Ordering gains will likely be small (tenths of a point up to 1–2 points), so seed noise matters when comparing ideas. Full-run seed noise has not been measured yet; smoke noise is about 0.4 pt.
- The time budget is wall-clock on whatever A5000 node runs the job. Throughput varies a few percent across nodes, and fixed training uses about 8–9 of the 15 minutes.
