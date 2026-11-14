# t10_label_noise: Label-Noise Survivor (dataset problem #10, Type II, train split)

CIFAR-10 with 40% symmetric label noise. The solution trains on noisy labels, and the score is clean test top-1.
The proxy budget is 15 min of training on 1 A5000 (the original task allows 2 h).

## Data (`prepare_data.py`, idempotent, sha-checked)
- **Noise convention: symmetric flip.** Exactly 20,000 of the 50,000 train indices are chosen uniformly without
  replacement (seed 20261110). Each one gets a new label drawn uniformly from the **9 other** classes, so a
  "replaced" label never equals the true one and the realised noise rate is exactly 40%. This is the Co-teaching
  style transition matrix, with 0.6 on the diagonal and 0.4/9 off it. Note that the DivideMix "sym" convention
  instead samples over all 10 classes, which gives about 36% real noise.
- `$AR_DATA/t10_label_noise/train_noisy.npz` holds `x` uint8 [50000,3,32,32] and `y` (noisy labels). It is the
  only file the training step reads. `meta.json` stores the seed and the label sha256
  `15a6589f…3838`.
- `$AR_ROOT/private/t10_label_noise/` holds `test.npz` (clean test set, read only by the eval step),
  `train_clean_labels.npy` and `noise_mask.npy` (for analysis only, never read by the harness).

## Harness (`harness/run.sh <solution_dir> <out_dir> [--smoke|--cpu-smoke]`)
`run.py` runs these steps:
1. `check.py` scans the solution statically. It looks for pretrained-weight APIs, network/download code and
   forbidden data paths or env vars, and flags bundled weight or data files.
2. `train_entry.py` runs in its own process with a kill timeout. It loads the noisy npz to the device, removes
   `AR_DATA/AR_ROOT/AR_REPO` from the environment, and calls `solution.train(images_u8, noisy_labels, artifact_dir, info)`.
   The sandbox env sets HF/transformers offline vars, uses empty per-run `TORCH_HOME`/`HF_HOME`/`XDG_CACHE_HOME`/triton
   dirs, and points `http(s)_proxy` at a dead port, so downloads fail.
3. `eval_entry.py` runs in a separate process. It calls `solution.load_predictor(artifact_dir, device)` and scores
   the test set in batches of 1000. In full mode this is the clean CIFAR-10 test set. In the smoke modes it is the
   last 1000 train images against their *noisy* labels, so Claude gets no test feedback while iterating.
4. `metric.json` is valid only if all of these hold: static check passes, training exits 0 within the budget
   (the `train()` call is ≤ 900 s, the process is killed at 960 s), the artifact is between 0 B and 512 MiB,
   eval finishes within 180 s, and the sandbox caches contain no downloaded weight files.

Solution interface: see `harness/CONTRACT.md`. `harness/validate_gpu.sbatch` is the validation job used below.

## Starter (`starter/solution.py`)
The starter is plain CE (label smoothing 0.1) with no noise handling. The network is an airbench-like ResNet
(widths 64/128/256/512, about 6.5M params). Data stays on the GPU, flip and crop augmentation run on the GPU,
it trains in bf16 autocast with channels_last at batch 1024, using SGD-Nesterov with lr 0.4 and a warmup/linear
schedule for 40 epochs. Eval adds flip TTA.

## Measured (job 13373482, fitz-12, 1x RTX A5000, AR_SEED=0)
| mode | metric | value | train_s | wall_s | valid |
| --- | --- | --- | --- | --- | --- |
| full | clean_test_top1 | **0.6844** | 99.4 | 105.8 | yes |
| --smoke | noisy_holdout_top1 | 0.441 | – | 31.7 | yes |
| --cpu-smoke (srun compsci, 4 cores) | noisy_holdout_top1 | 0.146 | ~24 | 36 | yes |

- GPU utilization (`nvidia-smi dmon -s um -d 5`): SM was **100%** in every sample during training
  (19/19 samples). It was 0% only during process start-up, data load and eval, which is a few seconds.
- About 2.2 s/epoch. The starter uses about 100 s of the 900 s budget.
- The training loss fell from 1.95 to 0.78 over the last epochs, which shows the CE model memorizing the noisy
  labels. That memorization is why clean accuracy stays at about 68%. It leaves a wide gap for noise-robust
  methods, which typically reach the high 80s or 90s.

## Constraint tests (cpu-smoke)
- `torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)` in the solution is caught by the static
  check (`pretrained_weights` rule), so `valid=false`.
- An obfuscated download (`getattr(torch, "h"+"ub")`, URL built from `chr()` codes) passes the static scan.
  It fails at runtime because the dead proxy refuses the connection, so training crashes and `valid=false`.

## Caveats
- The same Unix user owns everything, so the private files are protected only by convention, the static scan
  and the Claude adherence review. They are not protected by permissions. A determined solution could read the
  CIFAR pickles via obfuscated paths, and the scan can miss that.
- The static scan is regex-based. Comments containing e.g. `timm`, `https://` or the word `private` give false
  positives, which make the run invalid. CONTRACT.md tells solutions to avoid them.
- There is no network namespace (the proxy-env block only). Raw sockets that ignore the proxy are only caught by
  the static `socket` rule.
- Eval-time training is forbidden by contract and limited by the 180 s eval limit, but it is not otherwise detected.
