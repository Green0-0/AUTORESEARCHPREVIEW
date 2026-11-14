# t15_lwf solution contract

Do not edit anything under `harness/`. Your solution is a directory (`solution/`) whose entry point is
`solution/learner.py`. You may add other `.py` files next to it and import them.

Run it with `harness/run.sh <solution_dir> <out_dir> [--smoke | --cpu-smoke]`. This writes `<out_dir>/metric.json`.
Start by copying `starter/` to `solution/` and editing that.

## What `learner.py` must define

```python
def build_model() -> torch.nn.Module
    # The deployed classifier. Takes no arguments and must be deterministic in structure, because eval
    # rebuilds it and loads the saved state_dict with strict=True. If you grow heads, pre-allocate
    # all 100 outputs (or every slot you will need).

class Learner:
    def __init__(self, model: torch.nn.Module, info: dict): ...
        # model = build_model() already moved to info["device"]. Keep it as self.model; you may replace
        # self.model with another instance of build_model() later.
        # info: num_tasks=10, classes_per_task=10, num_classes=100, device ("cuda" or "cpu"),
        #       budget_s (total training seconds), mode ("full"|"smoke"|"cpu-smoke"), seed,
        #       image_shape=(3,32,32), image_dtype="uint8 0..255 NCHW"
    def learn_task(self, task_id: int, x: torch.Tensor, y: torch.Tensor, info: dict) -> None: ...
        # Called once per task, for task_id = 0..9 in order.
        # x: uint8 [N,3,32,32] and y: int64 [N], both already on the device (N = 5000 in full mode).
        # Labels are global: task t owns labels 10t..10t+9. info = {"classes": [...], "time_left_s": float}
    def aux_state(self) -> dict: ...
        # EVERYTHING your learner carries between tasks or into eval, apart from model parameters:
        # exemplars, prototypes/class means, statistics, bias-correction params, etc.
        # It must be loadable by torch.load(weights_only=True): tensors, dicts, lists, numbers, strings.

def predict(model: torch.nn.Module, aux: dict, x: torch.Tensor) -> torch.Tensor
    # x: uint8 [B,3,32,32] on the device. Returns [B,100] scores; the prediction is argmax over all
    # 100 classes. No task id is given. Called under torch.no_grad() with model.eval().
```

## How the harness runs it

1. **train.py**, one process, under a hard `timeout`.
   - The harness loads the train split, builds `model` and `Learner`, then calls `learn_task` for each task with only that task's data.
   - After each call it **zeroes** that task's `x` and `y` in place and drops them.
   - The clock covers `build_model` + `Learner.__init__` + every `learn_task`. The total must be ≤ `budget_s` (900 s in full mode). `time_left_s` tells you how much is left. Audits between tasks are not timed.
   - At the end the harness itself saves `artifact/model.pt` (`learner.model.state_dict()`) and `artifact/aux.pt` (`aux_state()`). Nothing else is carried into eval.
2. **eval.py**, a separate process. It calls `build_model()`, loads `model.pt` and `aux.pt`, and calls `predict` on the CIFAR-100 test set. Test data and labels are never available during training.

## Constraints (checked mechanically after every task; any failure means `valid: false`)

- **Aux-state limit (1 MiB = 1,048,576 bytes).** Two checks:
  - the `torch.save` size of `aux_state()`, and of the final `aux.pt`, must be ≤ 1 MiB;
  - a reachability scan must find ≤ 1 MiB of non-exempt data. The scan covers every tensor, ndarray, large bytes object or long numeric list reachable from the learner object, including module globals of your solution files.
- **What counts as "model parameters" (exempt):**
  - the `nn.Parameter`s of `learner.model`;
  - tensors whose shape equals the shape of one of those parameters, for example frozen snapshots or teacher copies of the model, optimizer moments, gradients, per-parameter importance weights (Fisher), anchor weights, and BatchNorm running stats.
  - The exempt parameter-shaped copies may total at most 8× the model's parameter bytes.
  - Image-shaped tensors (`[...,3,32,32]`) are never exempt.
  - Everything else counts toward the 1 MiB: buffers you register, prototypes, exemplars, the parameters of any extra network that does not share the model's parameter shapes (e.g. a generator), and so on.
- **Hidden-memory check (GPU).** Live CUDA memory not reachable from the learner must stay ≤ 8 MiB.
- **File check.** No files may be written into the solution dir during training.
- **Completion.** All 10 tasks must be completed, and the budget must not be exceeded.

## Rules (also checked by review)

- Never keep, copy, or re-derive raw or encoded training examples beyond what fits in the 1 MiB aux budget. Don't stash data anywhere the scan can't see (other modules, files, environment, subprocesses, disk).
- Don't read dataset files yourself. Use only the tensors the harness passes in. No pretrained weights, no downloads, no external data.
- Don't use the test set, and don't fit anything at `predict` time beyond a deterministic function of `(model, aux, x)`. Don't train or adapt on the test batch.
- Don't smuggle data inside "parameter-shaped" tensors (e.g. exemplars reshaped to look like a weight). Exempt copies must be genuinely model-parameter-like.
- **Data format.** Images are raw uint8, so do normalization and augmentation yourself, on the GPU.
  - For GPU efficiency, use large batches, AMP and channels_last, and avoid CPU dataloaders.
  - The model is the only thing that grows freely, so size it so that training fits the budget.

## Modes

| mode | device | data per class (train/test) | budget_s |
| --- | --- | --- | --- |
| full | cuda | 500 / 100 | 900 |
| `--smoke` | cuda | 100 / 100 | 60 |
| `--cpu-smoke` | cpu | 5 / 10 | 90 |

Use `info["mode"]` to scale your epochs down in the smoke modes.

`metric.json`: `metric = "final_top1_100way"`, `value` in percent. `constraint_report` contains:
- `checks`;
- the max aux / scanned bytes;
- `per_task_acc`, which is the 100-way accuracy on the test images of each task's classes;
- `pred_task_hist`, which is how many predictions land in each task's label range;
- `per_task_learn_s`.
