# t13_robust_noadv: Adversarial robustness without adversarial training (dataset #13, Type II, test split)

CIFAR-10. The goal is to maximize white-box L∞ PGD-20 robust accuracy at ε=8/255, with no
adversarial examples generated during training and clean accuracy ≥ 85%. The proxy budget is 20 min
of training on 1 A5000.

## Layout
- `TASK.md`: the prompt the pipeline sees.
- `task.yaml`: metadata. `floor`/`ref` are the real_score anchors, still null. `clean_acc_floor` is the validity gate.
- `prepare_data.py`: idempotent. Writes `$AR_DATA/t13_robust_noadv/{train,train_tiny}.npz` (uint8 NHWC + int64 labels)
  and `$AR_ROOT/private/t13_robust_noadv/test.npz` (the harness reads this at eval only).
- `harness/run.sh <solution_dir> <out_dir> [--smoke|--cpu-smoke]` runs these steps:
  1. `check.py`: static scan of the solution.
  2. `timeout <budget> python solution/train.py --train … --out <out>/artifact …`: the train step sees only train data.
  3. A rescan that also covers the artifact.
  4. `evaluate.py` in a separate process, which writes `metric.json`. If eval crashes or times out, run.sh writes a fallback `metric.json` with `valid: false`.
- `harness/CONTRACT.md`: what solutions must provide and what they may and may not do.
- `starter/`: plain-CE ResNet-9 (6.6M params). Data stays on the GPU, flip and crop augmentation run on the GPU, bf16 autocast, batch 512, 30 epochs, OneCycle SGD.

## Interface
`train.py` writes `artifact/model.py` (containing `build_model()`) and `artifact/weights.pt` (a state_dict).
The harness then:
- builds the model, loads the weights with `strict=True` and `weights_only=True`, and calls `.float().eval()`;
- feeds inputs in [0,1], so normalization has to live inside the model;
- evaluates in fp32 with TF32 off and cudnn deterministic.

## Attack and metric
- **Clean accuracy** is measured on all 10k test images.
- **PGD**:
  - L∞, ε=8/255, step 2/255, 20 steps, uniform random start, fixed seed 13013 (independent of `AR_SEED`), projection to ε-ball ∩ [0,1].
  - It runs on the first 2000 test images, in batches of 500.
  - It runs twice: once with **CE loss** and once with **margin loss** (max wrong logit − true logit).
  - `value` = the fraction of images that are correct clean, at every iterate and at the final point of **both** attacks.
  - The margin attack is there because it is invariant to logit scaling, so it defeats the cheapest gradient-masking trick (saturated softmax, which zeroes CE gradients).
  - Counting a sample as robust only if it survives every iterate guards against oscillating, noisy gradients.
- **Masking diagnostics** (`constraint_report.masking_diagnostics.flags`) are reported for the reviewer and do not invalidate a run by themselves:
  - the fraction of exactly-zero input gradients;
  - non-finite gradients;
  - a gap between CE and margin robust accuracy;
  - a gap between final-iterate and any-iterate robust accuracy;
  - accuracy under uniform noise at ε;
  - a sanity attack at ε=16/255 on 500 images;
  - robust accuracy above 0.5, which is implausible without adversarial training.
- **Determinism:** the harness runs two forward passes. If they differ by a relative amount above 1e-3, the run is invalid, because randomized defenses would need EOT.
- **Validity**: valid is false when any of the following holds:
  - training rc ≠ 0 (rc 124/137 means the budget ran out);
  - the scan finds a hard violation;
  - the model fails to load or eval;
  - the forward pass is nondeterministic;
  - clean accuracy is below 0.85 (full mode only; smoke modes report it without enforcing it).
- **Static scan** (`check.py`):
  - *Hard violations*: imports of attack or robust-weight libraries (torchattacks, foolbox, ART, advertorch, autoattack, cleverhans, robustbench), and any reference to test or raw CIFAR files or `private/`.
  - *Review flags* only: `autograd.grad`, `requires_grad` on tensors, `.sign()`, attack-like identifiers, `torch.func`/jacobian, `.grad` on input-like names, hub or pretrained downloads.
  - *Model files only*: no_grad, detach, round, quantize, JPEG, argmax, randomness and dropout.
  - Input-gradient-norm regularization (double backprop) is a legitimate non-adversarial method, so `autograd.grad` is only flagged, never invalidated.

## Clean-accuracy floor: 0.85
The original task's 85% is kept as the floor. In 97 s the starter reaches 93.8%, which is about 9 points
above the floor, so a standard fast ResNet clears 85% easily inside the 20-min budget. A stricter floor,
such as 90–91%, would rule out most noise-augmentation and smoothness-regularization ideas. Those ideas
typically cost 5–10 clean points, and trading clean accuracy for robustness under a fixed floor is
exactly what the task is about. To change the floor, edit `CLEAN_FLOOR` in `harness/evaluate.py` and
`clean_acc_floor` in `task.yaml`.

## Measured starter results (job 13373506, fitz-12, RTX A5000, 2026-11-08)
| mode | clean acc | robust acc (CE / margin / both) | train wall | eval wall | total |
| --- | --- | --- | --- | --- | --- |
| full | **0.9381** | 0.000 / 0.000 / **0.000** | 97.4 s (30 epochs, ~2.9 s/epoch) | 14.2 s (PGD 11.1 s) | 113 s |
| smoke | 0.798 (2 epochs) | 0.000 | 19.8 s | ~5 s | 25 s |
| cpu-smoke (loaded CPU node, 4 cores) | 0.20 | n/a | 17 s | ~58 s | 75 s |

- GPU utilization (`nvidia-smi dmon -s um -d 5`): SM ≈ 100% for every sample during training (2 idle samples were startup and the smoke-to-full switch).
- PGD sanity check:
  - uniform noise at ε=8/255 leaves accuracy at 0.90, while PGD takes it to 0.00 (also 0.00 under the final-iterate-only criterion);
  - zero-gradient fraction 0, no diagnostic flags;
  - scan: 0 flags on the starter.
- Masking-detection sanity: the same weights with logits multiplied by 1e6 give CE-PGD "robust" accuracy 0.94 (zero-gradient fraction 1.0), while margin PGD gives 0.00. The value stays 0 and two flags fire.
- So the starter's metric is 0.0. The real_score `floor` anchor should probably be 0 (or a weak baseline such as Gaussian-noise augmentation), and `ref` will come from a stronger reference run.

## Caveats
- **Gradient masking is the main gaming risk.** The margin loss plus the diagnostics catch logit saturation and exact-zero gradients, but not every obfuscation; examples are BPDA-requiring non-differentiable layers that still pass some gradient, and shattered gradients via sharp activations. The Claude reviewer must read `artifact/model.py` and the flags. A masking defense should get adherence 0.
- PGD-20 is a weak attack compared with AutoAttack, so robust accuracies are upper bounds. This follows the task text.
- The eval process imports solution code (`model.py`) with the test data in memory. A malicious model.py could read the test file. The static scan of the artifact flags references to test files, and the reviewer checks the rest.
- "No adversarial examples during training" is checked by heuristic plus review, not proven. The scan cannot tell input-gradient regularization apart from FGSM written with unusual names.
- The starter uses about 2 of the 20 budget minutes. The budget leaves room for e.g. bigger models, longer noise-augmented training or self-distillation.
- In cpu-smoke the eval uses only 128 clean and 16 PGD images, so its numbers mean nothing.

## Change 2026-11-08: scored ε lowered to 4/255
PGD probes (step = ε/4, CE+margin, every-iterate) on the starter and on starter + Gaussian noise aug σ=0.1 (clean 86.7%):

| ε | starter | noise aug |
| --- | --- | --- |
| 2/255 | 0.085 | 0.61 |
| 4/255 | 0.0003 | 0.30 |
| 8/255 | 0 | 0.027 |

At 8/255 honest methods were compressed into 0–3%, so the scored metric is now ε=4/255 with step 1/255. 8/255 is still reported as `robust_acc_eps8_secondary` and is not scored. The sanity attack is ε=16/255 against the scored 4/255; the uniform-noise diagnostic uses ε=4/255. The 85% clean floor is unchanged.
