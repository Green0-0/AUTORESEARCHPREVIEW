# t18_anytime — An Anytime Predictor (dataset problem #18, Type I, test split)

CIFAR-100. The solution trains (15 min, 1 A5000) a model exposing `anytime(x)`, a lazy generator of successive
`[B,100]` predictions. The harness measures cost in FLOPs (not wall clock) and scores the expected top-1 under a
fixed interruption distribution. See `TASK.md` (prompt) and `harness/CONTRACT.md` (solution interface).

## Pipeline
1. `prepare_data.py` (idempotent): `$AR_DATA/t18_anytime/{train,smoke_train,smoke_eval}.npz`; test set only at
   `$AR_ROOT/private/t18_anytime/test.npz`. Smoke modes evaluate on 2k held-out *training* images, never test.
2. `harness/run.sh <sol> <out> [--smoke|--cpu-smoke]`: `train.py` under `timeout` (900 / 60 / 90 s, incl. startup)
   -> `<out>/artifact/`; then `harness/evaluate.py` in a separate process (loads `model.py:load_model`, reads eval
   data itself); `harness/finalize.py` writes `metric.json`. Logs `nvidia-smi dmon` to `<out>/dmon.log` on GPU.
3. Scoring (`harness/anytime.py`):
   - Per-image cumulative cost c_k = `FlopCounterMode` total at the k-th yield / batch size (1 MAC = 2 FLOPs;
     conv/matmul/attention counted; elementwise/norm/pool free). Counted once on probe batches of 64 and 128 images;
     the full eval pass is then uncounted, batched (500), and only stores argmax hits per stage.
   - Interruption B ~ LogUniform[2e6, 2e9] FLOPs/image (median ~63 MFLOP; ResNet-20 ~80 MFLOP, ResNet-18 ~1.1 GFLOP).
     Score = sum_k top1_k * P(c_k <= B < c_{k+1}); no prediction before B = wrong. Computed **exactly** (no
     sampling noise; checked against 2M Monte-Carlo draws, agreement to 1e-4), so no seed dependence in the metric.
   - The budget is never passed to the model; the harness always consumes every stage.
   - Checks (-> `valid: false`): per-image FLOPs identical across the two probe batches (1% tol), batch-composition
     independence of predictions (>= 95% top-1 agreement), constant stage count 1..32, `[B,100]` outputs, train/eval
     time limits.
   - Report also has `final_stage_only_score` (= final top-1 x P(B >= c_S)), per-stage table, `effective_tflops`.

## Starter (`starter/`)
ResNet-9-style CNN, widths 32/64/128/256, 3 exits (max-pool + linear) after each stage; joint mean-CE training
(label smoothing 0.1), GPU-resident data, GPU crop+flip, bf16 autocast, channels_last, batch 512, SGD-nesterov with a
time-based one-cycle LR that fills `budget-75 s`.

Measured (job 13373490, fitz-12, AR_SEED=0):

| mode | value | wall | notes |
| --- | --- | --- | --- |
| full | **0.4415** (valid) | 835 s (train 830 s incl. startup, eval 5.4 s) | 64.9k steps / 670 epochs; SM util 100% in 167/168 busy dmon samples (mean 99.5%) |
| smoke | 0.182 | 40 s | GPU, SM 100% |
| cpu-smoke | 0.032 | 49 s | 4-core CPU node |

Full-run stage table: 11.2 MFLOP -> 44.2% (weight .239), 58.4 MFLOP -> 63.5% (.209), 247 MFLOP -> 67.2% (.303);
P(no prediction) = 0.25.

**Sanity check:** the final exit alone would score 0.672 x P(B >= 247M) = **0.203**, far below its plain accuracy 0.672
and below the 3-exit anytime score 0.4415.

Headroom for ideas: a first prediction below 2 MFLOP (25% of mass currently lost), more/denser exits, a bigger
final stage up to 2 GFLOP, distillation between exits, better per-exit loss weighting, longer training efficiency.

## Caveats
- FLOP counting relies on `FlopCounterMode`: compute in uncounted ops (e.g. conv re-expressed as multiply+sum,
  custom kernels, `torch.compile`, other threads) escapes counting. Forbidden in CONTRACT.md; not mechanically
  gated — `effective_tflops` (starter: 23 TFLOP/s on GPU over the full pass) is reported for the reviewer.
- Input-adaptive compute (per-sample early exit) is ruled out by the identical-FLOPs check by design.
- Free ops (normalization, pooling, softmax) are genuinely free; a solution could lean on them somewhat.
- `train.py` hard limit includes Python/CUDA startup; the starter leaves 75 s margin (finished at 830/900 s).
- floor/ref in task.yaml left null (to be set from starter + reference runs).
