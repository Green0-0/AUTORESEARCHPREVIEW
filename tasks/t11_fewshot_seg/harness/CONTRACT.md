# t11_fewshot_seg — solution contract

`harness/` is read-only. Write your solution in `solution/` and test it with
`harness/run.sh solution <out_dir> --cpu-smoke` (CPU node) or `--smoke` (GPU). The full run is
`harness/run.sh solution <out_dir>`.

The harness owns the data, the ImageNet backbone and the evaluation. You supply two files.

## `solution/model.py`
```python
def build_model(make_backbone) -> torch.nn.Module
```
- Call `make_backbone()` (optionally `make_backbone(replace_stride_with_dilation=[False, True, True])` or any other
  3-bool list; no other arguments) and store the result as **`model.backbone`**. At training time it returns a
  torchvision `resnet50` holding the provided **ImageNet IMAGENET1K_V2** weights; at evaluation time it returns
  the same architecture without weights, and your trained weights are loaded into the whole model.
- Everything else (decoder, heads, normalization buffers) is yours. You may use any of the backbone's modules
  (`conv1, bn1, relu, maxpool, layer1..4`, `fc`) in any way, but do not add/remove/replace modules inside the
  backbone: its `state_dict` keys and shapes must stay those of torchvision `resnet50`.
- `forward(x)`: `x` is float32 `(B,3,H,W)`, **RGB in [0,1]** (no normalization; put ImageNet mean/std inside the
  model, see `ctx["mean"]`, `ctx["std"]`). Return **logits `(B,21,h,w)`** at any resolution; the harness upsamples
  them bilinearly (`align_corners=False`) to `(H,W)` and takes the argmax. Class 0 is background.
- `build_model` is called again in a fresh process for evaluation, so all learned state must live in the model's
  parameters/buffers, and the module structure must not depend on training. `model.py` may import torch,
  torchvision, numpy and other `.py` files in `solution/` (they are copied into the artifact).
- At evaluation the model is in `eval()` mode, fp32, TF32 off, cuDNN deterministic. The output must be
  deterministic (checked). Test-time augmentation inside `forward` (e.g. flips, multi-scale) is allowed, as long
  as the whole val set evaluates within the eval cap (600 s in full mode, fp32 with TF32 off; the starter takes
  ~8 s, a DeepLabv3+ at output stride 16 with flip TTA ~135 s).

## `solution/train.py`
```python
def train(model, ctx) -> None
```
Train `model` (the object `build_model` returned, already on `ctx["device"]`) **in place**. The harness saves
`model.state_dict()` when `train` returns. Do not add modules or change parameter shapes after `build_model`.

`ctx` keys:
| key | value |
| --- | --- |
| `mode` | `"full"`, `"smoke"` or `"cpu-smoke"` |
| `device` | `torch.device` (cuda, or cpu in cpu-smoke) |
| `time_budget_s`, `deadline` | 900 s (full), 45 s (smoke), 40 s (cpu-smoke); `deadline` is a `time.time()` value |
| `seed` | int (`AR_SEED`) |
| `train_x` | uint8 tensor **on the device**, `(100,320,320,3)` RGB NHWC: the 100 few-shot training images |
| `train_y` | uint8 tensor on the device, `(100,320,320)`: labels 0..20, **255 = ignore** (VOC void boundaries + padding) |
| `train_hw` | int64 CPU tensor `(100,2)`: each image's (h, w) before padding (the content sits at the top-left) |
| `voc_ids`, `classes` | the VOC image ids; the 21 class names |
| `num_classes`, `ignore_index`, `res` | 21, 255, 320 |
| `mean`, `std` | ImageNet normalization constants (RGB) |
| `make_backbone` | the same factory `build_model` got: more ImageNet ResNet-50 copies (e.g. a frozen teacher / EMA) |
| `scratch_dir` | a writable directory for temporary files (deleted after training). `TMPDIR`, `TRITON_CACHE_DIR` and `TORCHINDUCTOR_CACHE_DIR` point into it, so `torch.compile` works. |

The 100 images: for each of the 20 object classes, 5 VOC 2012 *train* segmentation images that contain it (drawn
with a fixed seed, all distinct), each with its **full** 21-class annotation (so other classes, e.g. person, occur
in more than 5 images). Images were resized so the longer side is 320 and padded bottom/right to 320×320 with the
colour (124,116,104) and label 255. These are the only training data: there are no unlabeled VOC images, no SBD,
and no ImageNet images.

### Budget
Importing your two files, `build_model()` and the whole `train()` call must finish within `time_budget_s`
(+10 s grace), otherwise the run is **invalid**. The process is hard-killed at 1020 s (full). Stop on
`ctx["deadline"]` with a safety margin. Loading the data and the backbone file happens before the timer starts.
Smoke modes only need to show the code runs end to end (a few steps).

GPU efficiency: the data is already on the GPU; do augmentation there (e.g. `affine_grid`/`grid_sample`), use
AMP and reasonable batches, and don't use a CPU DataLoader. `starter/` shows the pattern.

### Allowed
Any decoder / head architecture; fine-tuning or freezing any part of the backbone, dilation; any loss, optimizer,
schedule, augmentation, batch size, crop size; self-training / pseudo-labels on the 100 training images; copies of
the ImageNet backbone from `ctx["make_backbone"]` as teachers; `torch.compile`; TTA inside `forward`.

### Not allowed
- Any pretrained weights other than the provided ImageNet ResNet-50: no segmentation- or detection-pretrained
  weights (torchvision `models.segmentation` / `models.detection`, DeepLab/FCN COCO/VOC weights, SAM, CLIP, DINO,
  timm, hub, `weights=` / `pretrained=True` on any torchvision builder), no weight files shipped in `solution/`.
  `model.backbone` must be the object `make_backbone` returned (identity-checked), and every `torch.load` during
  training must read only files you wrote into `scratch_dir` (audited).
- Any data other than `ctx["train_x"]` / `ctx["train_y"]`: no VOC val/test images, no other VOC/SBD/COCO/ImageNet
  files, nothing under `$AR_ROOT` / `$AR_DATA` / private dirs, no network (there is none).
- Writing anywhere except `scratch_dir`. Modifying `harness/`.

## What the harness checks (`metric.json` → `constraint_report`)
- `train.within_budget`: import + `build_model` + `train` wall time ≤ budget + 10 s.
- `train.structure` / `backbone_arch_ok`: `model.backbone` is a harness-built ImageNet ResNet-50 with torchvision
  resnet50 keys/shapes; the model's state_dict keys/shapes did not change during `train`; no hooks left; the
  artifact loads with `strict=True` into a fresh `build_model(...)`.
- `train.torch_load_events`: `torch.load` / hub downloads during training outside `scratch_dir` invalidate.
- all weights and val logits finite; `deterministic`: re-running the first val batch gives the same logits.
- `scan`: static scan of the solution (`harness/check.py`). Hard violations (network, other pretrained models or
  weights, other datasets, paths into the data tree or private data, weight/data files in `solution/`) invalidate
  the run. Review flags (file reads, `torch.load`, subprocess, hooks, own torchvision builders) go to the reviewer.
- `backbone_rel_change` (diagnostic): how far a few backbone tensors moved from the ImageNet weights.

**Metric:** `voc12_val_miou` (**higher is better**): mean IoU over the 21 classes on the **1,449 Pascal VOC 2012
val** images at the fixed 320×320 resolution (longer side 320, padded; padding and void pixels ignored), from one
confusion matrix accumulated over all val pixels. `constraint_report` also gives `pixel_acc` and `per_class_iou`.
Any violation makes the run `valid: false`; `value` is still reported (0 if the eval itself failed).

## Running
```
harness/run.sh <solution_dir> <out_dir> [--smoke | --cpu-smoke]
```
`--cpu-smoke`: CPU, 40 s training, 8 val images, < 1 min (numbers are meaningless).
`--smoke`: GPU, 45 s training, 256 val images (evenly spread fixed subset). No flag: full mode.
