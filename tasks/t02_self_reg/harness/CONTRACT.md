# Harness contract: t02_self_reg (self-regularizing network, CIFAR-100, 5 images per class)

`harness/` is read-only. Write your solution in `solution/` and test it with
`harness/run.sh solution <out_dir> --cpu-smoke` (CPU node, about 20 s) or `--smoke` (GPU). The full run is
`harness/run.sh solution <out_dir>`.

## What the solution must provide
Two files in `solution/` (more `.py` helpers are fine; the whole dir is copied into the artifact):

1. `model.py` with `build_model() -> torch.nn.Module`.
   - Input: float32 NCHW `(B,3,32,32)` RGB in **[0,1]**, exactly as given (no normalization is applied
     by the harness; put it inside the model). Output: logits `(B,100)`.
   - **All input processing lives in `forward`**: normalization, and any augmentation, noise, cropping,
     multi-view averaging and so on. Whatever `forward` does at training it also does at test.
   - `build_model()` must not read any files. The eval process calls `build_model()`, then
     `load_state_dict(torch.load(model.pt, weights_only=True), strict=True)`, so everything the model
     needs at test time must be in its parameters/buffers.
2. `train.py` with `train(model, x, y, ctx) -> None`. It trains `model` in place.
   - `model` is the object returned by your `build_model()`, already on `ctx["device"]`.
   - `x`: float32 `(N,3,32,32)` in [0,1] on the device; `y`: int64 `(N,)` fine labels. N = 500 (5 per class) in
     full/smoke mode and 100 (1 per class) in cpu-smoke. These are the **only** data you may use.
   - `ctx`: `{"mode": "full"|"smoke"|"cpu-smoke", "device", "time_budget_s", "deadline" (time.time() value),
     "seed", "num_classes": 100}`. The harness seeds `random`, `numpy` and `torch` with `seed` before
     `build_model()`.
   - After `train()` returns, the harness saves `model.state_dict()`.

## Rule: no training-only regularization (how it is checked)
The model must process data at training exactly as at test. Allowed vs. not:
- Not allowed: dropout, stochastic depth, BatchNorm with batch statistics (in train *or* eval mode), any
  augmentation / mixup / cutmix / cutout / noise applied to the training data outside `forward`,
  `self.training` or `torch.is_grad_enabled()` branches, extra forward arguments (`model(x, train=True)`),
  cross-example mixing inside `forward`, augmentation whose strength depends on training progress.
- Allowed: any augmentation or randomness *inside* `forward` (it then also runs at test; e.g. a random flip, or
  averaging K random views where K is fixed inside the model), GroupNorm/LayerNorm/no norm, any architecture,
  init, optimizer, LR schedule, weight decay, loss function, label smoothing, EMA of weights, self-distillation
  from the model's own outputs on raw images, any number of epochs.

Checks:
1. **Runtime guard during `train()`** (hard). Every forward with gradients enabled must:
   - be a call `model(x)` with exactly one float32 `(B,3,32,32)` tensor and no other args or kwargs;
   - have every row of `x` equal to one of the raw training images (a float64 fingerprint is checked against
     the 500 images). Do not cast the input to bf16 yourself; autocast inside the call is fine;
   - not call submodules of `model` from outside `model.forward` (e.g. `model.body(aug(x))`).
   Forwards under `torch.no_grad()` are not checked. A grad-enabled forward of a network that is not part of `model`
   (e.g. a separate teacher) is flagged for review. Do not wrap `model` in another object or `torch.compile`
   the top-level module, and do not call `model.forward(...)` directly (submodules may be compiled inside
   `build_model()`). Do not touch module hook registries.
2. **Post-training functional checks** (hard, in the eval process, on a copy of the trained model, fp32, 256 test images):
   - (a) `model.train()` + grad enabled vs. `model.eval()` + `no_grad`, same RNG seed: logits must match
     (relative max difference ≤ 1e-3);
   - determinism: the same seed twice gives the same logits;
   - batch independence: same seed and batch size, with the second half of the batch replaced by other images.
     The logits of the first half must not change;
   - logits shape `(B,100)` and finite.
3. **Static scan** (`check.py`). Hard: references to test / private / raw CIFAR files, `AR_DATA`/`AR_ROOT`,
   torchvision datasets, pretrained weights or network access, hook registries, augmentation libraries
   imported in `train.py`. Everything else is a review flag: augmentation-like ops, random tensors, dropout/BN,
   functional forwards or file access in `train.py`; mode/grad branches, dropout/BN and randomness in `model.py`.
   A reviewer reads the code: circumventing the rule by other means gets adherence 0.

## Evaluation
- Rebuild from `artifact/code/model.py`, load the state_dict strictly, `.float()`, `model.eval()`,
  fp32 with TF32 off and cudnn deterministic.
- The global RNG (`random`, `numpy`, `torch` CPU+CUDA) is seeded **once** with a fixed seed (20020, independent
  of `AR_SEED`). The 10,000 test images are then fed in a fixed order in batches of 500 under `no_grad`, one
  forward per batch. Randomness inside `forward` is therefore used at test exactly as at training.
- The metric `cifar100_test_top5` is top-5 accuracy on the full test set; `top1` is in `constraint_report`.
  Eval has a 300 s cap, so keep the forward reasonably cheap (e.g. K-view averaging with modest K).

## Budget and modes
- full: `train()` wall time ≤ 900 s (+10 s grace), measured by the harness. The process is killed at 960 s.
  Plan on staying well below 900 s, e.g. check `ctx["deadline"]`.
- smoke: `train()` ≤ 60 s on GPU (aim for ~20 s). cpu-smoke: `train()` ≤ 60 s on 4 CPU cores with 100 training images
  and 500 test images; just run a few steps.
- GPU efficiency: the data is already on the GPU. Use large batches and bf16 autocast, and do in-forward
  augmentation on the GPU; don't use a CPU DataLoader. `starter/` shows the pattern.
- Seed from `ctx["seed"]` (= `AR_SEED`).

## Output
`<out_dir>/metric.json`: `{"metric": "cifar100_test_top5", "value": <top-5 in [0,1]>, "valid": bool,
"constraint_report": {top1, top5, train, guard, checks, scan, errors, review_flags, ...}, "wall_s", "mode"}`.
Logs: `<out_dir>/train.log`, `<out_dir>/eval.log`, `<out_dir>/train_report.json`, `<out_dir>/scan.json`.
