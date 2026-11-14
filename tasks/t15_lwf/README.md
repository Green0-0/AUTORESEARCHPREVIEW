# t15_lwf: Learning Without Forgetting (class-incremental Split CIFAR-100)

Dataset problem #15, Type I, **train** split. Proxy budget: 15 min of training on 1×A5000 for the whole 10-task sequence.

## Layout
- `TASK.md`: the prompt the pipeline sees.
- `task.yaml`: task metadata (`floor`/`ref` are still null).
- `prepare_data.py`: idempotent data prep.
  - Fixes the class order with `CLASS_ORDER_SEED=1993` and relabels so that task t owns labels 10t..10t+9.
  - Writes `$AR_DATA/t15_lwf/{train.npz,class_order.json}` and `$AR_ROOT/private/t15_lwf/test.npz` (test is never in the harness/idea dir).
- `harness/CONTRACT.md`: the solution interface and rules (what Claude reads).
- `harness/run.sh <sol> <out> [--smoke|--cpu-smoke]`: runs `train.py` (under `timeout`), then `eval.py` (separate process), and writes `metric.json`.
- `harness/common.py`: modes, data loading, and the reachability/aux-state accounting.
- `harness/validate_gpu.sbatch`: harness-prep GPU validation with dmon (`sbatch --export=ALL,MODE=full,SOL=..,OUT=.. -t 00:25:00`).
- `starter/learner.py`: plain fine-tuning of a ResNet-9-style CNN (6.6M params). Data is GPU-resident, augmentation runs on the GPU, uses bf16 AMP and channels_last, 30 epochs per task, bs 256, SGD with OneCycle, and no aux state.

## How it works
- **Data stream.** The harness owns the stream.
  - Train data stays in CPU memory. On each task's turn, that task's uint8 images and labels are moved to the GPU and passed to `Learner.learn_task` once.
  - Afterwards the harness zeroes them in place and drops them, so any alias the learner kept sees zeros.
  - `AR_DATA`/`AR_ROOT` are unset before the solution is imported.
- **Audit (after each task, off the clock).** Any failed check means `valid=false`.
  - (a) The `torch.save` size of `aux_state()` must be ≤ 1 MiB.
  - (b) An object-graph scan from the learner and from the solution modules' globals (including `__builtins__`) finds every tensor, ndarray, bytes object or long numeric list. Non-exempt bytes, deduplicated by storage, must be ≤ 1 MiB.
    - Exempt: the model's Parameters, plus tensors shaped like a model parameter (snapshots, optimizer moments, grads, Fisher, BN stats), capped at 8× the model's parameter bytes.
    - Image-shaped tensors are never exempt.
  - (c) Live CUDA memory not reached by the scan must be ≤ 8 MiB (after clearing cuBLAS workspaces).
  - (d) No files may be written into the solution dir.
  - (e) All 10 tasks must be done within budget.
- **Artifact.** The harness itself saves `model.state_dict()` and `aux_state()` as the artifact. Eval rebuilds `build_model()` (strict load) and calls `predict(model, aux, x)` to get 100-way logits on the 10k test images, with no task id.
- **Metric.** `final_top1_100way`, in percent. The report also includes `per_task_acc` and `pred_task_hist`.

## Measured results
Starter, seed 0, on an RTX A5000 (job 13373513):

| mode | value (top-1 %) | valid | stream s | wall s | notes |
| --- | --- | --- | --- | --- | --- |
| full | **7.64** | true | 94.0 | 104.7 | per-task acc = 0 for tasks 0–8 and 76.4 for task 9 (pure forgetting); **SM 100%** in every dmon sample during training (~52% mem util); eval 0.5 s |
| --smoke | 1.06 | true | 3.9 | 13 | |
| --cpu-smoke (srun -p compsci -c 4) | 0.8 | true | 25.8 | 47 | |

The starter uses only ~10% of the budget. That is deliberate: it is a floor, and longer fine-tuning doesn't help it.

**Constraint tests.** Cheat solutions are in `$AR_ROOT/t15_tests/sol_*`, all built on the starter. Every one was flagged `valid=false` while the starter stayed valid:
- `sol_bigaux`: `aux_state()` returns 1.6 MB of prototypes. It fails `aux_state_le_1MiB` and `scanned_aux_le_1MiB`.
- `sol_stash`: keeps a replay buffer on `self` that it doesn't declare. It fails `scanned_aux_le_1MiB`.
- `sol_global`: stashes numpy copies in a module global. It fails `scanned_aux_le_1MiB`, and the path is reported in `top_aux`.
- `sol_hidden`: hides GPU copies in `builtins`. The scan finds it through `__builtins__`, so it fails `scanned_aux_le_1MiB`.
- `sol_hidden2`: hides a GPU copy in `torch.nn.init.__dict__`, which the scan can't reach. It fails `no_hidden_cuda_memory` with 123 MB unaccounted.

## Caveats
- **Hidden-CUDA tolerance.** The legit starter shows up to 1.2 MB of unaccounted CUDA memory (allocator/cuDNN internals), so the threshold is 8 MiB. A copy hidden outside the scanned graph that is smaller than ~7 MiB (≈2k uint8 images) on the GPU, or any copy kept in CPU RAM outside the graph, is not caught mechanically. Neither is disk or environment smuggling outside the solution dir, or reading the dataset files by guessed path. All of these are forbidden in CONTRACT.md and left to the review step.
- **Parameter-shaped exemption.** This is an interpretation of "excluding model parameters": frozen teacher copies (classic LwF), EWC Fisher and optimizer state are free, up to 8× the model. Data could in principle be reshaped into a parameter-shaped tensor. That is forbidden by rule and left to review.
- **Model size.** It is unconstrained apart from the time budget. Eval also has no time limit beyond a 300 s timeout.
- **Smoke-mode results.** These are meaningless as accuracy numbers; they only check that the pipeline runs end to end.
- **Overhead.** The budget clock counts only `build_model` + `Learner.__init__` + `learn_task`. Process startup, data loading and the audits add ~10 s, and the hard timeout is 1080 s.
