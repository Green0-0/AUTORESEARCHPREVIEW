# Harness contract: t13_robust_noadv (robustness without adversarial training)

`harness/` is read-only. Write your solution in `solution/` and test it with
`harness/run.sh solution <out_dir> --cpu-smoke` (CPU node) or `--smoke` (GPU). The full run is
`harness/run.sh solution <out_dir>`.

## What the solution must provide
`solution/train.py`, run as
```
cd solution && python train.py --train <train.npz> --out <artifact_dir> \
    --mode full|smoke|cpu-smoke --budget-s <seconds> --device cuda|cpu --seed <int>
```
- `<train.npz>` holds `x`: uint8 `(N,32,32,3)` (NHWC, RGB) and `y`: int64 `(N,)`. This is the CIFAR-10
  training set: 50,000 images in full/smoke mode and 512 in cpu-smoke. It is the **only** data you may
  read. Do not read test data, other CIFAR files or anything under `$AR_ROOT/private`, and do not
  download anything. The static scan invalidates any reference to them.
- Before exiting, train.py must write into `<artifact_dir>`:
  - `model.py`, defining `build_model() -> torch.nn.Module`. It may import torch, torchvision, numpy
    and other `.py` files you copy into `<artifact_dir>`. Copy the file there; the harness does not use
    `solution/` at eval time.
  - `weights.pt`, which the harness loads with `model.load_state_dict(torch.load(..., weights_only=True), strict=True)`.
- Modes. In `full`, train for real: the hard budget is `--budget-s` = 1200 s, wall clock, enforced
  with `timeout`, and it includes startup and data loading, so leave a margin. In `smoke`, train for a
  minute or less (e.g. 1–2 epochs); the harness kills training at 75 s. In `cpu-smoke`, run a few
  steps on CPU (killed at 90 s) just to exercise the code path.
- Seed from `--seed` (or `AR_SEED`).
- GPU efficiency: keep the data on the GPU and do augmentation there, use AMP and large batches, and
  don't use a CPU DataLoader. `starter/` shows the pattern.

## The model at evaluation (what the harness does)
- `model = build_model(); load weights; model.float().eval()`; parameters are frozen.
- Input: float32 NCHW in **[0,1]** (no normalization applied by the harness; put your normalization
  inside the model). Output: logits `(N,10)`.
- The model is evaluated in **fp32** (TF32 off). It must return the same output for the same input
  (determinism is checked; randomness at test time makes the run invalid).
- Clean accuracy is measured on all 10,000 CIFAR-10 test images and must be **≥ 0.85** in full mode,
  or the run is invalid.
- Robust accuracy is the metric `pgd20_robust_acc`. It is measured on the first 2000 test images
  with an L∞ PGD-20 attack: ε=4/255, step 1/255, uniform random start in the ε-ball, fixed seed,
  projection onto the ε-ball ∩ [0,1]. The attack runs twice, once with a cross-entropy loss and once
  with a margin loss (max wrong logit − true logit), on the gradients of your model's logits. A
  sample is robust only if it is classified correctly clean, at every PGD iterate, and at the final
  point of **both** attacks.

## Rules
1. **No adversarial examples during training.** You may not run FGSM, PGD or any procedure that
   optimizes perturbations of the inputs to increase the loss, not even one step, not even
   "virtual" adversarial perturbations (VAT) or adversarial weight/feature-space perturbations that
   are computed from input gradients. Also forbidden: attack libraries (torchattacks, foolbox, ART,
   advertorch, autoattack, cleverhans) and robust or pretrained weights (robustbench, torch.hub, …).
   Allowed: random noise or augmentation, regularizers on the model, input-gradient *norm* penalties
   (double backprop) that do not construct perturbed inputs, architecture choices, label smoothing,
   distillation from a model you train yourself within the budget, and so on. A static scan flags
   `torch.autograd.grad`, `requires_grad` on tensors, `.sign()`, attack-like names and functional
   autodiff for human review. Clear cases make the run invalid.
2. **No gradient masking or obfuscation.** The defense must be a genuinely robust, ordinary
   differentiable network. Not allowed in `forward`: `no_grad` or `detach` on the input path,
   rounding, quantization or JPEG of inputs, non-differentiable or randomized preprocessing,
   input-dependent stochasticity, deliberately saturated logits (e.g. multiplying logits by a huge
   constant), detection or rejection branches, or using `self.training` / gradient-enabled state to
   behave differently under attack. The harness reports diagnostics (`constraint_report.masking_diagnostics`:
   zero-gradient fraction, CE-vs-margin gap, uniform-noise accuracy, ε=16/255 sanity check) and a
   reviewer reads `model.py`. A masking defense gets adherence 0.
3. Training wall time must stay within the budget; overrunning makes the run invalid.
4. Don't modify `harness/`.

## Output
`<out_dir>/metric.json`: `{"metric":"pgd20_robust_acc","value":<robust acc in [0,1]>,"valid":bool,
"constraint_report":{clean_acc, robust_acc_ce, robust_acc_margin, masking_diagnostics, scan, errors, ...},
"wall_s", "mode"}`. Logs: `<out_dir>/train.log` and `<out_dir>/eval.log`.
