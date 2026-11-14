# t07_activation_tuning — Activation Tuning (dataset problem #7, Type I, train split)

CIFAR-100 with a fixed MLP, 3072 → 1024 → 1024 → 1024 → 100 (5,348,452 Linear params). Only the three hidden activations `f_i(W_i h + b_i)` and the training procedure are up to the solution. Activations may hold at most 20% of the total params, counting activation params and float buffers (at most 1,337,113). Budget: a 900 s `train()` call on 1 A5000.

## Layout
- `TASK.md`: the ideation prompt. `task.yaml`: metadata (floor and ref are null).
- `prepare_data.py`: idempotent. Writes `$AR_DATA/t07_activation_tuning/train_{x,y}.npy` (uint8 NCHW, fine labels) and `$AR_ROOT/private/t07_activation_tuning/test_{x,y}.npy`.
- `harness/CONTRACT.md`: the solution interface. Claude reads it.
- `harness/t07lib.py`: the `MLPNet` skeleton, data loading and the checks (params, structure, purity and batch independence).
- `harness/train_wrapper.py`: process 1. Loads the train data onto the GPU, builds `MLPNet(build_activations(1024))`, calls the solution's `train(model, x, y, ctx)` under a timer, checks the structure, then saves `artifact/model.pt` (state_dict) and `artifact/code/` (a copy of the solution).
- `harness/evaluate.py`: process 2. Rebuilds the skeleton from `artifact/code/activations.py`, loads the state_dict strictly, computes fp32 test top-1 (TF32 off) and runs the checks. Writes `metric.json`.
- `harness/run.sh <solution_dir> <out_dir> [--smoke|--cpu-smoke]`: hard caps are 1080 s (full), 150 s (smoke) and 180 s (cpu-smoke).
- `starter/`: ReLU on all 3 layers. AdamW (lr 3e-3, wd 0.05) with one-cycle, label smoothing 0.1, GPU flip + 4px reflect-crop, batch 2048, bf16 autocast, 1500 epochs.

## Design decisions
- The harness owns `forward`, so the activation can only ever see its own layer's `Wx+b`. Eval rebuilds the net in a fresh process from the state_dict, so changes made during training don't carry over, and strict key matching catches extra modules.
- Mechanical checks:
  - activation param fraction ≤ 0.20, with no sharing of Linear params;
  - class, children, Linear shapes and no hooks, after training and after reload;
  - shape and finiteness of each activation's output;
  - **purity**: re-apply `f_i` to the exact `z` it saw in a forward pass, after an unrelated forward pass, and require the same output (this catches hidden skip connections through global state);
  - **batch independence** in eval mode: same logits under batch permutation and for single-example batches;
  - the `train()` wall time stays within budget plus 10 s.
- Training freedom: the optimizer, schedule, loss, augmentation, init, epochs and train-only stochasticity inside activations are all free. Teachers, ensembles, external data and file reads are not allowed. Per the letter of the task, normalization-style activations (BN/LN on `Wx`, with params counted) are allowed if they pass the checks above.

## Measured results (1 A5000, fitz node, job 13373523, AR_SEED=0)
| mode | top-1 | train() wall | total wall | GPU SM util |
| --- | --- | --- | --- | --- |
| cpu-smoke (login/compsci CPU, 2k train / 1k test) | 0.029 | 4 s | 13 s | n/a |
| smoke | 0.183 | 2.1 s | ~9 s | n/a |
| **full (starter)** | **0.3213** | **77.9 s** | **83.4 s** | **100% in every dmon sample during training** (mean 79% over the whole job, including startup and eval) |

An earlier version (batch 1024, 300 epochs) reached 0.3131 in 28 s, with only 80–81% util. Bigger batches fixed the util.

## Constraint-check tests (cpu-smoke, all handled as expected)
| deliberately bad solution | result |
| --- | --- |
| a Linear(1024,1024) inside each activation (act fraction 0.37) | valid=false (params) |
| hidden skip connection via module-global state | valid=false (purity check) |
| batch-statistics normalization in eval mode | valid=false (batch independence) |
| `train()` adds a submodule `model.extra` | valid=false (structure + unexpected state_dict keys) |
| `train()` sleeps past the budget | valid=false (within_budget) |
| BatchNorm1d + per-channel PReLU (0.29% params) | valid=true (no false positive) |

## Caveats
- The starter uses only about 80 s of the 900 s budget. A 3×1024 MLP trains at about 0.05 s/epoch, so the budget is generous, and ideas can spend it on more epochs or heavier per-unit activations. The floor and ref anchors still need to be set (floor ≈ starter 0.32).
- Raw CIFAR-100 test pickles also exist in `$AR_DATA/cifar-100-python/`. The contract forbids reading files, and the review step has to enforce that.
- The purity and batch-independence checks are functional tests on 512 test images. They can't prove the absence of all side channels (e.g. code that only misbehaves on certain inputs). The read-only review covers the rest.
- Eval runs in fp32. An activation that only works under bf16 autocast would be evaluated in fp32 anyway.
