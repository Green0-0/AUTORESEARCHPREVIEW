# Solution contract: t10_label_noise

Your solution is a directory containing `solution.py` (plus any other `.py` modules it imports from
the same directory). `harness/` is read-only; do not modify it. Start from `starter/solution.py`.

## Required functions in `solution.py`

```python
def train(images, labels, artifact_dir: str, info: dict) -> None
```
- `images`: `torch.uint8` tensor `[N, 3, 32, 32]` (raw 0-255 RGB, NCHW), already on `info["device"]`.
- `labels`: `torch.int64` tensor `[N]` on the same device: the **noisy** labels (40% wrong). Clean labels are never provided.
- `artifact_dir`: an existing empty directory. Save everything needed for inference here (≤ 512 MiB total).
- `info`: `{"mode": "full"|"smoke"|"cpu-smoke", "budget_s": float, "deadline": float (time.time() epoch seconds),
  "device": torch.device, "seed": int, "n_classes": 10}`.
  Use `info["mode"]` to shrink the run in smoke modes, and stop training before `info["deadline"]`.

```python
def load_predictor(artifact_dir: str, device: torch.device) -> Callable[[torch.Tensor], torch.Tensor]
```
- Called in a **separate process** after training. Returns `predict(images_u8)` mapping a `uint8 [B,3,32,32]`
  tensor on `device` to scores `[B, 10]` (logits or probabilities; argmax is taken). Called under
  `torch.inference_mode()` with B ≤ 1000. Test-time augmentation / ensembling of saved models is fine;
  any training, fine-tuning, or adaptation at eval time is not.

## Modes (`harness/run.sh <solution_dir> <out_dir> [--smoke | --cpu-smoke]`)
| mode | device | train data | train budget | eval | metric |
| --- | --- | --- | --- | --- | --- |
| full | 1 A5000 | all 50,000 noisy | 900 s | clean CIFAR-10 test (10k) | `clean_test_top1` |
| `--smoke` | GPU | 49,000 noisy | 60 s | last 1,000 train images vs their noisy labels | `noisy_holdout_top1` |
| `--cpu-smoke` | CPU | first 2,048 noisy | 90 s | as smoke | `noisy_holdout_top1` |

Smoke metrics are against noisy labels and only show that the code runs; they are not a quality signal.
Result: `<out_dir>/metric.json` (`valid`, `value`, `constraint_report`); logs in `train.log`, `eval.log`.

## Rules (checked mechanically; a violation makes the run invalid)
- **Budget**: the `train()` call must finish within `budget_s` (process killed after budget + 60 s).
  Eval must finish within 180 s (full).
- **No pretrained weights / external data / network**: training starts from random init. Don't use
  `torch.hub`, torchvision `weights=`/`pretrained=`, `timm`, `transformers`, HF hub, or any URL/download;
  don't ship weight or data files (`.pt/.pth/.npz/...`, or any non-code file > 64 KiB) in the solution dir.
  Training runs offline with empty cache dirs and a dead HTTP proxy.
- **Data**: use only the tensors passed to `train()`. Don't read any dataset files yourself (no
  `torchvision.datasets`, CIFAR paths, `AR_DATA`/`AR_ROOT` env vars, or the harness's private files).
- Only `torch`, `torchvision` (architectures with `weights=None` and transforms only) and `numpy`
  are guaranteed available. Any model architecture, loss, schedule, or semi-supervised trick is allowed,
  including using the training images without labels and multiple models, as long as it fits in the budget.
- GPU efficiency matters: keep the data on the GPU, augment on the GPU, use AMP and large batches.
- The source scan is regex-based and also reads comments/strings: avoid writing URLs, `timm`, `transformers`,
  `pretrained=True`, `private`, dataset file names, or `AR_*` env-var names anywhere in your solution files.
