# t19_data_ordering: solution contract

The harness owns the whole training loop. Your solution only decides **which training examples form each batch**.

## What you provide
`solution/scheduler.py` must define a top-level `class Scheduler`:

```python
class Scheduler:
    def __init__(self, ctx): ...            # called once, after the model has been built
    def next_batch(self, step: int):         # step = 0 .. ctx.total_steps-1, called in order
        """Return exactly ctx.batch_size integer indices in [0, ctx.n_train).
        A torch integer tensor (CPU or ctx.device, 1-D), a numpy integer array or a list of ints.
        Duplicates are allowed, inside a batch and across batches."""
    def observe(self, step, indices, losses, logits):   # OPTIONAL; called right after each update
        """indices: int64 (B,) on device, the batch just trained on (after validation).
        losses: float32 (B,) per-example training loss (label-smoothed CE, augmented inputs, train-mode BN),
                computed in the forward pass before this step's update.
        logits: float32 (B, 100) of that same forward pass. All detached. Don't modify them."""
```
You may add other `.py` files to `solution/`. The solution dir is on `sys.path`.

## `ctx` (read-only)
| field | meaning |
| --- | --- |
| `ctx.n_train` | number of training examples (50000 in full/smoke, 2048 in cpu-smoke) |
| `ctx.batch_size`, `ctx.total_steps`, `ctx.total_presentations` | fixed: 512 × 6000 in full mode (≈61 epochs); differ in smoke modes, so **read them, don't hard-code** |
| `ctx.images` | uint8 tensor (N, 3, 32, 32) of training images on `ctx.device` (a copy) |
| `ctx.labels` | int64 tensor (N,) of training labels on `ctx.device` (a copy) |
| `ctx.num_classes` | 100 |
| `ctx.device`, `ctx.seed`, `ctx.mode` | torch device, `AR_SEED`, `"full"`/`"smoke"`/`"cpu-smoke"` |
| `ctx.budget_s`, `ctx.time_left()` | wall budget of the whole training process, and the seconds left in it |
| `ctx.recipe` | dict describing the fixed optimizer/schedule (for information only) |
| `ctx.normalize(x_uint8)` | the harness's input normalization, if you build your own scoring model |
| `ctx.probe(indices, augment=False, batch_size=2048)` | forward pass of the **current** training model on training examples: eval-mode BN, no grad, nothing in the model changes. It returns float32 logits (len, 100). With `augment=True` it uses the fixed flip+crop with a separate RNG, so the training augmentation stream is untouched. Time is charged to the budget, and the number of examples probed is logged. |

## Rules
- **Allowed:** any ordering, curriculum, repetition or subsetting policy, including ones computed from training images and labels, the feedback above, `ctx.probe`, or a separate scoring model that you train yourself on the training set inside the budget.
- **Not allowed:**
  - changing or reaching the training model, its weights, BN statistics, the optimizer, the LR schedule, the loss or the augmentation (no introspection via `gc`, frames or harness internals);
  - reading test data or anything under `$AR_ROOT/private`;
  - external data or pretrained weights;
  - changing global torch seeds to influence the harness (the harness uses its own generators);
  - editing `harness/`.
- **Budget:** the whole training process, from python start through scheduler init, every `next_batch`/`observe`/`probe` call and all training steps until the model is saved, must finish within `ctx.budget_s` (900 s in full mode). Fixed training alone takes about 8–9 min on an A5000, so keep scheduling cheap. Overruns abort the run, and the run is invalid.
- **Throughput:** `next_batch`/`observe` run on the critical path every step. Keep the work on the GPU and avoid `.item()`, `.cpu()` and `.tolist()` per step where you can, because each one stalls the pipeline. Amortize (for example, plan a whole epoch at once).
- **Rejected (run invalid):** a wrong batch length, a non-integer dtype, a non-1-D shape, or any index outside `[0, n_train)`. GPU-tensor indices are range-checked asynchronously every 50 steps, and any bad index invalidates the run.

## Running
`harness/run.sh <solution_dir> <out_dir> [--smoke | --cpu-smoke]` writes `<out_dir>/metric.json`. `--cpu-smoke` (CPU, 2048 examples, batch 64, 12 steps, about 70 s) checks that your code runs end to end. `--smoke` is a short GPU run. Training writes `<out_dir>/artifact/` (model + per-example presentation counts) and `train_report.json` (histogram of presentation counts, scheduler/probe timing). A separate step then evaluates test top-1.
