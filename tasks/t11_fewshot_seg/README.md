# t11_fewshot_seg: Few-Shot Segmentation (dataset problem #11, Type I, train split)

Shortened proxy of problem #11 (original: 6 h on 1 A5000, ImageNet-1k provided). Pascal VOC 2012 semantic
segmentation with only 5 fully annotated training images per object class (100 images). The solution gets ONE fixed
ImageNet backbone (torchvision ResNet-50 `IMAGENET1K_V2`); the ImageNet dataset is not provided (that is the
shortening). Segmentation/detection-pretrained weights and all other weights are banned. Metric: mIoU over the 21
classes on the full VOC 2012 val set (1,449 images) at a fixed 320×320 resolution, **higher is better**. Budget: 900 s
on 1 A5000 for importing the solution, `build_model()` and `train()`. Added so that judge calibration is not fit only
on CIFAR classification.

## Layout
- `TASK.md`: the prompt the pipeline sees. `task.yaml`: metadata (floor measured, ref null).
- `prepare_data.py`: idempotent, needs network once (`source env.sh && python tasks/t11_fewshot_seg/prepare_data.py`, about 25 s on 4 cores when the download is fast). It writes:
  - `$AR_DATA/t11_fewshot_seg/` (138 MB): `train_x.npy` uint8 (100,320,320,3), `train_y.npy` uint8 (100,320,320), `train_hw.npy`, `train_meta.json` (VOC ids, the class each image was drawn for, class coverage), `resnet50_imagenet1k_v2.pth` (torchvision `resnet50-11ad3fa6.pth`, 102.5 MB) and `meta.json` (seeds, source md5, per-class pixel counts, sha256 of every output).
  - `$AR_ROOT/private/t11_fewshot_seg/` (569 MB): `val_x.npy` uint8 (1449,320,320,3), `val_y.npy` uint8 (1449,320,320), `val_hw.npy`. These are **.npy only**, so `scripts/make_decoy_private.py` fakes them (decoy labels are random 0..255; labels > 20 are ignored by the metric, so decoy evals still run).
  - The raw VOC tar and everything extracted from it are **deleted** at the end: the Claude jail decoys only `$AR_ROOT/private`, so raw val masks must not remain anywhere else. A re-run checks the hashes and is a no-op; a mismatch aborts (it never re-draws silently).
- `harness/t11lib.py`: mode table, the backbone factories (`make_backbone_pretrained` loads the local ImageNet file; `make_backbone_skeleton` is the same architecture without weights for eval), structure check, `torch.load` audit, data loading and mIoU.
- `harness/train_wrapper.py`: process 1. Loads the 100 images onto the GPU, imports `solution/model.py` and `train.py`, calls `build_model(make_backbone)` and `train(model, ctx)` under the timer and inside the load audit, checks structure and provenance, and saves `artifact/model.pt` (fp32 state_dict) plus a copy of the solution code.
- `harness/evaluate.py`: process 2. Rebuilds the model with `build_model(make_backbone_skeleton)`, loads it with `strict=True`, checks the backbone shapes, finiteness and determinism (re-runs the first batch), and scores the val set in fp32 (TF32 off, cuDNN deterministic).
- `harness/check.py`: static scan (hard violations and review flags).
- `harness/run.sh <solution_dir> <out_dir> [--smoke|--cpu-smoke]`: offline. Hard caps: the training process gets 1020 s (full), 120 s (smoke) or 100 s (cpu-smoke); the eval gets 600 s (full) or 120/100 s. `TMPDIR`, Triton/Inductor caches, `XDG_CACHE_HOME` and `TORCH_HOME` point into `out/scratch`, so no cached hub/torchvision weights are reachable through the usual paths.
- `harness/validate_gpu.sbatch`: GPU validation **inside `realjudge/jail.sh` harness mode** (read-only filesystem, no network), with `nvidia-smi dmon` logging.
- `starter/`: frozen ImageNet ResNet-50 (stride 32, BN in eval mode) plus a 1×1-conv linear head on layer4. The logits are upsampled bilinearly to 320×320 and trained with cross-entropy (ignore 255). No augmentation. AdamW lr 1e-3 (constant), batch 16, bf16 autocast, channels_last, training until 15 s before the deadline (about 15k steps).

## Data
- **Source:** the genuine `VOCtrainval_11-May-2012.tar` from the official Oxford host (`https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/voc2012/`, which `host.robots.ox.ac.uk` redirects to). It is 1,999,639,040 bytes, and its md5 `6cd6e144f989b92b3379bac3b3de84fd` matches the value torchvision's `VOCSegmentation` checks. The download took about 15 s on 2026-11-09 (the pjreddie mirror was just as fast). No HF mirror was needed.
- **Counts:** the segmentation split has 1,464 train and 1,449 val images, which are disjoint. Both counts are asserted.
- **Few-shot set:** for each class c = 1..20 in order, 5 images are drawn with `default_rng(1111)` from the train images that contain c and were not already drawn. That gives 100 distinct images, each with its full annotation. Because of co-occurrence, some classes appear in more than 5 images: person in 35, car 16, chair 15, pottedplant 14, sofa 13, and the rest in 5-8.
- **Resolution:** the longer side is resized to 320 (PIL bilinear with antialiasing for images, nearest for masks), then padded bottom/right to 320×320 with RGB (124,116,104) and label 255. A typical 500×375 image becomes 320×240. Label 255 covers VOC void boundaries and padding; it is ignored in training and in the metric.
- **Metric:** one 21×21 confusion matrix over all val pixels with labels 0..20, then mIoU = mean over classes of TP/(TP+FP+FN). This is the standard VOC protocol, but at 320 resolution instead of the original one, so the numbers are not directly comparable to published full-resolution VOC results. Smoke uses an evenly spread fixed subset of 256 val images and cpu-smoke 8.

## Interface (details in harness/CONTRACT.md)
- `model.py: build_model(make_backbone) -> nn.Module` must store `make_backbone(...)` as `model.backbone`. The only keyword `make_backbone` accepts is `replace_stride_with_dilation`.
- `forward(x)` takes float (B,3,H,W) RGB in [0,1] and returns logits (B,21,h,w); the harness upsamples them.
- `train.py: train(model, ctx)` trains in place. `ctx` provides `train_x`/`train_y` (uint8, on the GPU), `train_hw`, `voc_ids`, `classes`, `mean`/`std`, `make_backbone` (for extra ImageNet copies), `scratch_dir`, `deadline`, `seed` and `mode`.

## Measured
All runs were on RTX A5000s (fitz-18/24/26), through `validate_gpu.sbatch`, inside the harness jail, on 2026-11-09 (jobs 13486497 and 13486498). Outputs are in `$AR_ROOT/harness_runs/t11_fewshot_seg/`.

| run | val mIoU (metric) | pixel acc | train() wall | total wall (load / eval) | GPU SM util |
| --- | --- | --- | --- | --- | --- |
| **full, starter, seed 0** | **0.2016** | 0.749 | 885.1 s (15,018 steps) | 904.7 s (0.4 / 8.4) | 99.5% mean non-idle; 177/183 dmon samples ≥ 80% |
| **full, starter, seed 1** | **0.2016** | 0.748 | 885.1 s (15,271 steps) | 902.3 s | 99.5%; 177/181 |
| **full, starter, seed 2** | **0.2021** | 0.748 | 885.1 s (15,189 steps) | 902.3 s | 99.7%; 177/181 |
| full, probe, seed 0 | 0.5221 | 0.886 | 880.1 s (9,742 steps) | 1022.5 s (0.5 / 131.1) | 99.5%; 200/205 |
| full, probe, seed 1 | 0.5264 | 0.886 | 880.1 s (9,128 steps) | 1025.3 s (eval 135.0) | 99.9%; 202/206 |
| full, probe, seed 2 | 0.5226 | 0.886 | 880.1 s (9,220 steps) | 1024.5 s (eval 134.2) | 99.9%; 202/206 |
| smoke, starter | 0.290 (256 val imgs) | 0.811 | 40.1 s (664 steps) | 63.6 s | 100% during training |
| smoke, probe | 0.516 (256 val imgs) | | 40 s (417 steps) | | 99-100% during training |
| cpu-smoke, starter / probe (4 cores, harness jail) | 0.031 / 0.083 | | 1.0 s / 6.2 s (3 steps) | 11 s / 21 s | n/a |
| cpu-smoke, starter / probe, `JAIL_MODE=claude` (decoy val) | 0.010 / 0.002 (meaningless) | | | | n/a |

- The dmon samples below 80% are CUDA start-up and eval. Peak memory: starter 0.79 GB, probe 2.96 GB.
- **Floor = 0.2018 ± 0.0003** (mean ± sd over starter seeds 0-2), recorded in `task.yaml`.
- The starter's full run (0.20) scores below its own 40 s smoke run (0.29 on the 256-image subset). The linear head at a constant LR keeps fitting the 100 images for 15k steps and overfits. Even a schedule or early stop would raise the floor, which is fine for a naive starter.
- Starter per-class IoU: background 0.76, bus 0.40, and most object classes 0.03-0.27.

**Headroom probe** (`$AR_ROOT/t11_dev/probe_solution/`, outside starter/; its idea is in `scripts/make_anchor_ref.py` REF_IDEAS):
- Architecture: DeepLabv3+. The backbone is dilated in layer4 (output stride 16), with ASPP (rates 6/12/18 plus image pooling, 256 channels) and a decoder with a 48-channel layer1 skip.
- Training: the whole backbone is fine-tuned at lr 1e-3 with the head at 1e-2, backbone BN frozen. SGD-nesterov with wd 1e-4, 3% warmup, then time-based poly 0.9 decay. Batch 16, bf16.
- Augmentation, all on the GPU via `affine_grid`/`grid_sample`: scale 0.5-2.0, translation, flip and colour jitter.
- Eval: flip TTA.

Result: **0.5237 ± 0.0024**, so it raises mIoU by 0.322 (2.6× the floor). That gap is about 1,000× the starter's seed sd and about 190× the pooled sd, so the task has ample headroom. Probe per-class IoU: background 0.90, objects 0.11 (pottedplant) to 0.74 (bus).

## Constraint-check tests (cpu-smoke, harness jail; `$AR_ROOT/t11_dev/neg/`)
| deliberately bad / borderline solution | result |
| --- | --- |
| backbone = own `torchvision.models.resnet50()` (not from `make_backbone`) | **valid=false**: structure (backbone not harness-built) |
| `resnet50(weights="IMAGENET1K_V1")` | **valid=false**: scan hard violation; at runtime the patched torchvision loader raises |
| `torch.load` of the cached torchvision checkpoint via an obfuscated path (scan-invisible) | **valid=false**: torch.load audit |
| `train()` sleeps past the budget | **valid=false**: within_budget |
| NaN written into the head | **valid=false**: non-finite weights / logits |
| noise added in `forward` | **valid=false**: determinism |
| `model.extra = Linear` added during training | **valid=false**: state_dict changed + strict load fails |
| no training at all (control) | valid=true |
| `make_backbone(replace_stride_with_dilation=[False,True,True])` (control) | valid=true |

## Caveats
- **Private data is visible in harness mode.** The harness eval needs it, so this applies to all tasks. A solution that `np.load`s `$AR_ROOT/private` through an obfuscated path would escape the static scan; only `torch.load` is audited at runtime. Only the reviewer would catch it.
- The provenance check proves that `model.backbone` was created by the harness with the ImageNet weights. It cannot prove that the solution did not overwrite those weights with tensors computed some other way. Loading other weights is covered by the scan plus the `torch.load` audit, and by the jail (no network; other weights on disk are reachable only via `/usr/xtmp`, which the scan flags).
- Eval is fp32 with TF32 off. The probe's eval (OS16 plus flip TTA) takes about 130 s of the 600 s cap; output stride 8 plus multi-scale TTA could approach it. The full-mode wall time per run is then about 17 min.
- Seed noise is tiny for the starter (frozen backbone, deterministic eval) and 0.0024 for the probe.
- `artifact/model.pt` is about 100 MB (starter) or 163 MB (probe) per run.
- `$AR_ROOT/t11_dev/probe_solution` is readable from the Claude jail, as `t09_dev` is.
