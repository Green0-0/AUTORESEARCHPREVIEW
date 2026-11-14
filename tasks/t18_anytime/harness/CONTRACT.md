# t18_anytime — solution contract

Run: `harness/run.sh <solution_dir> <out_dir> [--smoke | --cpu-smoke]` -> `<out_dir>/metric.json`.
Do not edit anything in `harness/`. Use `starter/` as a template.

## Files the solution must provide (`<solution_dir>/`)
1. `train.py` — run as `cd <solution_dir> && python train.py --train <train.npz> --out <artifact_dir> --mode {full|smoke|cpu-smoke} --budget-s <seconds>`.
   - `<train.npz>`: `x` uint8 `[N,32,32,3]` (HWC, RGB), `y` int64 `[N]` (CIFAR-100 fine labels). Full mode: the 50k training set. Smoke modes: a 5k slice.
   - It must write everything needed for inference into `<artifact_dir>` and exit.
   - Hard wall-clock limit (process killed, run invalid): full 900 s, smoke 60 s, cpu-smoke 90 s, **including** Python/CUDA startup and saving. Use `--budget-s` to size your schedule and leave a margin (the starter trains for budget-75 s).
   - Full mode runs on 1 A5000 (8 CPU cores, 32 GB RAM). `--cpu-smoke` has no GPU (`CUDA_VISIBLE_DEVICES=""`): the code must also run on CPU, just with a tiny schedule.
   - Seed from env `AR_SEED` (default 0). Data paths come only from the arguments; never read other files under `$AR_DATA` / `$AR_ROOT`.
2. `model.py` — must define `load_model(artifact_dir: str, device: torch.device) -> model`, where `model.anytime(x)` is a **generator** (or any iterator):
   - input `x`: float32 tensor `[B,3,32,32]`, values in [0,1] (unnormalized RGB, NCHW), already on `device`;
   - yields logits/scores `[B,100]`, one tensor per prediction, cheapest first; 1 to 32 predictions.
   - It must be **lazy**: compute for prediction k+1 happens only after prediction k is yielded (the harness counts all compute done before each yield toward that prediction, so eager precomputation just makes earlier predictions expensive).
   - The harness calls it under `torch.no_grad()`, after `model.eval()` if it exists. You may use autocast (bf16/fp16), channels_last etc. inside `anytime`.

## How it is scored
- Per-image cost of prediction k = cumulative FLOPs counted by `torch.utils.flop_counter.FlopCounterMode` from the start of `anytime(x)` up to the k-th yield, divided by batch size (counted on probe batches of 64 and 128 eval images). Counted ops: convolutions, matmul/linear/bmm/baddbmm, scaled-dot-product attention (1 multiply-add = 2 FLOPs). Elementwise ops, normalization, pooling, softmax are free.
- Interruption budget B ~ LogUniform[2e6, 2e9] FLOPs/image. An image is scored with the latest prediction whose cost <= B; none -> wrong.
- Metric `expected_top1_at_interruption` = sum_k top1_k * P(c_k <= B < c_{k+1}) (exact expectation over B; equals averaging over infinitely many draws). Full mode uses the 10k CIFAR-100 test set; smoke modes use 2k held-out *training* images (numbers not meaningful).
- `metric.json.constraint_report` lists each prediction's cost, top-1, P(reached) and weight, plus `final_stage_only_score` and diagnostics.

## Rules (mechanically checked ones marked [checked]; violations -> `valid: false`)
- [checked] Per-image FLOPs of every prediction must be identical for all inputs and batch sizes (within 1%): no input-dependent or batch-dependent compute schedules.
- [checked] A prediction for an image must not depend on the other images in the batch (top-1 agreement >= 95% when batch composition changes). No state carried across calls/batches; no test-time training or adaptation.
- [checked] Number of predictions constant, 1..32; each `[B,100]`.
- [checked] Training finishes within the budget; evaluation within its time limit (900 s full).
- The model is never told the interruption budget and must not try to infer it.
- All inference compute must go through ordinary PyTorch ops in the calling thread and be visible to FlopCounterMode: no `torch.compile`/custom CUDA/Triton kernels/extra threads or processes in `anytime`, and no re-expressing convolutions/matmuls via uncounted elementwise ops (e.g. multiply-and-sum) to dodge counting. (`effective_tflops` in the report is audited.)
- Training uses only the provided training data; no pretrained weights, no downloads, no test data. Anything (distillation, ensembles, multiple networks, label smoothing, etc.) is allowed within the training budget.
