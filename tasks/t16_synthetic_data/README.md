# t16_synthetic_data — Useful Synthetic Data (dataset problem #16, Type II, test split)

## How it works
1. `prepare_data.py` (idempotent; `--force` to rewrite)
   - `$AR_DATA/t16_synthetic_data/real10.npz`: 10 CIFAR-100 train images per class, chosen with fixed seed 16 (independent of `AR_SEED`). The idx sha1 prefix is `fcee41e14102`.
   - `$AR_ROOT/private/t16_synthetic_data/withheld_arch.py`: the withheld third classifier and its recipe. The source is embedded in `prepare_data.py`, never in `harness/`.
2. `harness/run.sh <solution_dir> <out_dir> [--smoke|--cpu-smoke]` calls `harness/run.py`, which:
   - **Synthesis (solution code):** a subprocess `solution/synthesize.py --real … --out <out>/artifact …` with a hard wall-clock kill at the budget (900 s full / 60 s smoke / 90 s cpu-smoke). `AR_DATA` points to a sandbox that contains only `real10.npz`. Timeout or non-zero exit means `valid: false`.
   - **Validation:** `synthetic.npz` must hold exactly the keys `x` and `y`. `x` is `(N,32,32,3)` uint8, or float16/32 that is finite and in [0,1]. `y` is integer in 0..99. At most 100 per class. Storage is reported as the npz file size, the raw nbytes and the size of the whole artifact dir.
   - **Fixed classifiers (harness cost, not budgeted):** `convnet` (ConvNet-3, width 128, instance norm), `resnet` (narrow ResNet-9-style, BN), and `withheld` (loaded via importlib from the private dir; the run **fails with exit 3 and `valid: false`** if that file is missing). Each one:
     - trains 1500 SGD-nesterov steps at batch 512, sampling with replacement from the GPU-resident set;
     - uses on-GPU reflect-pad crop + flip, bf16 autocast, channels_last, and a warmup-then-linear-decay LR;
     - uses seed `AR_SEED*1000+k`;
     - is then tested on the 10k CIFAR-100 test images (read only in this step).
3. The output is `metric.json`. `metric = mean_top1_3arch` (fraction). `constraint_report` holds `per_arch_top1`, `per_arch_wall_s`, `synthesis_wall_s`, `classifier_wall_s`, `storage_bytes_npz`, `storage_bytes_raw`, `per_class_max`/`min`, `checks`, and `errors`.

Labels are **hard labels only**. This is a deliberate simplification: soft and learned labels are a big lever in the dataset-distillation literature, and here they are out of scope.

## Starter
`starter/synthesize.py` keeps each real image once and adds 9 random crop/flip copies of it, giving 100/class and 10,000 uint8 images. It runs on CPU in about 1.5 s and writes a 25.5 MB compressed npz (30.8 MB raw).

## Measured results (1x A5000, fitz)
| run | mean | convnet | resnet | withheld | synthesis wall | classifier wall | total |
| --- | --- | --- | --- | --- | --- | --- | --- |
| starter, full, AR_SEED=0 (job 13373471) | **0.1701** | 0.1818 | 0.1878 | 0.1406 | 1.3 s | 150 s (40/42/67) | 153 s |
| starter, full, AR_SEED=1 (job 13373486) | 0.1667 | 0.1689 | 0.1929 | 0.1382 | 1.5 s | 151 s | 158 s |
| identity only (1000 real images, eval-only, seed 0) | 0.1735 | 0.1826 | 0.1970 | 0.1410 | — | 148 s | 150 s |
| starter, `--smoke` (60 steps) | 0.081 | | | | 1.8 s | 36 s | 48 s |
| starter, `--cpu-smoke` (srun compsci, 3 steps, 500 test imgs) | ~0.01 | | | | | 15 s | 18 s (19 s wall incl. srun) |

GPU utilization (`nvidia-smi dmon -s um -d 5`): **SM 100%** during all three classifier trainings (job 13373486: mean 93% over all non-idle samples). The only gaps are about 5 s at each model switch and the Python/CUDA start-up. Memory use is under 4 GB.

Rejection tests (`run.py --eval-only` on hand-made artifacts) all gave `valid: false` with a clear error:
- 101/class;
- float values of 1.5;
- labels 95..104;
- float (soft) labels;
- NCHW shape;
- a missing artifact;
- a missing private withheld file (exit 3).

## Caveats
- The harness itself already applies crop+flip, so augmentation copies add nothing beyond identity: the starter scores 0.170/0.167 against 0.174 for identity, within seed noise of about 0.004 on the mean. Gains have to come from genuinely new information or better-conditioned examples.
- The classifiers train for a fixed 1500 steps whatever N is, and they reach near-zero training loss. Small sets are not penalised by having fewer steps.
- The "no other data" rule is enforced only by the sandboxed `AR_DATA` and by review. The CIFAR files are still readable by absolute path, so `check`/review must look for reads outside `--real`.
- The withheld architecture is hidden by convention only: it sits in `$AR_ROOT/private/` and inside `prepare_data.py`. Claude sessions should not be pointed at `tasks/t16_synthetic_data/prepare_data.py`. Copy only `harness/`, `TASK.md` and `starter/` into idea dirs.
- `floor`/`ref` in `task.yaml` are still null. The starter (about 0.17) is the natural floor, and a ref still needs a stronger solution (e.g. distribution matching or a small generative model).
