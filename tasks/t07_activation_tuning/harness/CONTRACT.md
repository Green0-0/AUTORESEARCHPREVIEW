# t07_activation_tuning — solution contract

The harness owns the network. You supply the **activation functions** and the **training procedure**.

## Fixed network (harness/t07lib.py `MLPNet`, do not copy or modify)
```
x: float images in [0,1], (B,3,32,32)
h0 = flatten((x - mean) / std)                # fixed CIFAR-100 channel mean/std, (B,3072)
h1 = f1(L1(h0))   L1 = nn.Linear(3072, 1024)
h2 = f2(L2(h1))   L2 = nn.Linear(1024, 1024)
h3 = f3(L3(h2))   L3 = nn.Linear(1024, 1024)
logits = L4(h3)   L4 = nn.Linear(1024, 100)   # no output activation; prediction = argmax
```
Linear params: 5,348,452. Width is fixed at 1024.

## Files you must provide in `solution/`
### `activations.py`
```python
def build_activations(width: int) -> list[torch.nn.Module]   # exactly 3 modules: [f1, f2, f3]
```
- Each `f_i` is an `nn.Module` with `forward(z) -> tensor`, mapping `z` of shape (B, 1024) to the same shape. `z` is that layer's `Wx + b`. It gets nothing else.
- `f_i` may be anything that is a function of `z` alone: elementwise or not, with or without learnable parameters, different per layer. It may hold parameters and buffers, which are saved in the state_dict.
- **Parameter limit:** `act_params <= 0.20 * (linear_params + act_params)`, so at most 1,337,113 in total across the three activations. `act_params` counts every parameter and every floating-point buffer in the activation modules. An activation may not reference the Linear weights.
- **In eval mode** (`model.eval()`), `f_i` must be deterministic and per-sample. Its output for one example must not depend on other examples in the batch, on earlier calls, on other layers, or on any global or hidden state.
- `build_activations` is called again in a fresh process at evaluation time, followed by `load_state_dict(strict=True)` on the saved weights. All learned state must therefore live in parameters or buffers. Nothing may be read from files.
- The module is imported from a copy of your `solution/` dir, so helper modules there may be imported. Don't name any module `t07lib`.

### `train.py`
```python
def train(model: MLPNet, x: torch.Tensor, y: torch.Tensor, ctx: dict) -> None
```
- `model` is the harness `MLPNet` already built with your activations, on `ctx["device"]` and in train mode. Train it **in place**. The harness saves `model.state_dict()` after `train` returns.
- `x`: uint8 (N,3,32,32) training images, already on the device. Divide by 255 before calling `model`. `y`: int64 (N,) fine labels. N = 50,000 in full and smoke modes and 2,048 in cpu-smoke.
- `ctx` = `{"mode": "full"|"smoke"|"cpu-smoke", "device": torch.device, "time_budget_s": float, "deadline": float (time.time()), "seed": int, "width": 1024, "num_classes": 100}`.
- The time budget for the `train()` call is 900 s in full mode, 60 s in smoke and 45 s in cpu-smoke. The harness allows 10 s of grace, and anything beyond that makes the run **invalid**. The process is hard-killed at about 18 min. Keep a safety margin, for example by stopping on `ctx["deadline"]`. Smoke modes only need to show that the code runs (about 1 to 3 epochs).
- **Allowed:** any optimizer, LR schedule, loss, regularization, weight init, data augmentation (do it on the GPU), batch size, epochs, AMP or `torch.compile(model)`, use of a validation split carved from `x`, and train-mode-only behaviour of your activations (e.g. noise or dropout inside `f_i`). Wrappers such as `torch.compile(model)` are fine as long as the original `model` object is what gets trained.
- **Not allowed:**
  - modifying the skeleton: replacing, adding or removing submodules of `model` (including replacing `model.acts[i]` after construction), changing Linear shapes, monkey-patching `forward`, or registering hooks that remain after `train` returns;
  - training or using any other network (teachers, distillation, ensembles). Averaging `model`'s own weights (EMA/SWA) and copying the result back into `model` is fine. Only `model`'s final state_dict is evaluated;
  - pretrained weights or external data;
  - reading any files (in particular test data);
  - writing anything into the artifact.

## What the harness checks (`metric.json` → `constraint_report`)
- `params`: per-layer and total activation params, and the fraction (must be ≤ 0.20).
- `structure_ok`: class, children, Linear shapes and no hooks, checked after training and again after reload.
- `state_dict_ok`: a fresh `build_activations` + skeleton loads the saved state_dict with no missing or unexpected keys.
- `behaviour_ok` (eval mode, fp32): each `f_i` preserves shape and gives finite outputs. Re-applying `f_i` to the exact `z` it saw in a forward pass, after unrelated forward passes, reproduces its output (purity). Logits are unchanged under batch permutation and for single-example batches (batch independence).
- `within_budget`: the `train()` wall time.
- **Metric:** CIFAR-100 test top-1 of the reloaded model, in eval mode, fp32 (TF32 off), with no test-time augmentation. Any violation makes the run `valid: false`, and `value` is still reported.

## Running
```
harness/run.sh <solution_dir> <out_dir> [--smoke | --cpu-smoke]
```
`--cpu-smoke` runs on CPU with a 2,048-image train slice and 1,000 test images, in under 3 min. `--smoke` runs on GPU with a 60 s training budget. With no flag, it runs in full mode.
