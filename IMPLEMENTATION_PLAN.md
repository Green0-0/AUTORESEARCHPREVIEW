# Autoresearch Preview — Implementation & Experiment Plan (v2)

Revised 2026-11-08. Deadline: **Mon 2026-11-16 23:59** (hard). Source spec: `class_notes/Autoresearch Preview Spec.md`.

What changed from v1:
- 4-bit Qwen3.8-27B served with pipeline parallelism (PP) instead of TP.
- Tasks split into train and test. The train tasks calibrate the surrogate judge.
- Seven shortened tasks, none needing ImageNet (3 train + 4 test).
- Very few real (Claude Code) evals: about 38 in total.
- No Claude-agent comparison arm. The baseline is a naive Qwen pipeline.
- A simpler statistical design with no SHAP.
- Building each task's eval harness is scheduled as the **last** build step.

---

## 0. Cluster constraints

| Fact | Consequence |
| --- | --- |
| `~` is at **493k / 500k files** (57G / 100G). | Venvs, the HF cache, datasets, the vLLM cache, run dirs and uv/pip caches all go under `/usr/xtmp/kg349/ar/`. Set `HF_HOME`, `UV_CACHE_DIR`, `XDG_CACHE_HOME`, `TMPDIR` and `VLLM_CACHE_ROOT` in every sbatch. Prune `~/.claude/projects/*` after headless Claude runs; copy the transcripts into the idea dir first. |
| A5000s (24 GB, Ampere, **PCIe, no NVLink**, 4 per node) on `compsci-gpu`, nodes `compsci-cluster-fitz-[01-04,09-34]`. | Always use `-p compsci-gpu --gres=gpu:a5000:N`. Never use the lab partitions (`bhuwan`, `nlplab*`, `grisman`, `rudin`, `wiseman`, `skynet`). |
| No vllm on the login node. Modules include `cuda/cuda-12.9`, `uv/0.11.26`, `python/3.12.8`. | Build one uv venv on xtmp from inside a GPU job. |
| `claude` 2.1.319 at `~/.local/bin/claude`, logged in with a subscription. | Real judging uses `claude -p` with **`ANTHROPIC_API_KEY` unset**, so it bills the subscription, not the API. |

**Day-0 checks: all passed on 2026-11-08** (details in §1.0). `claude -p` works from compute nodes; the 4-bit model is `RedHatAI/Qwen3.8-27B-INT4`; PP = 2 works in vLLM 0.33.0. CIFAR-10/100 are in `/usr/xtmp/kg349/ar/data`, with MD5s matching the official values.

---

## 1. Serving: 4-bit Qwen3.8-27B with pipeline parallelism

### 1.0 Confirmed by the Day-0 probe (2026-11-08, job 13373415, fitz-13)
- **Model:** `RedHatAI/Qwen3.8-27B-INT4`. This is W4A16 compressed-tensors (group 128), served through Marlin on Ampere. It is 19.5 GB on disk, at `/usr/xtmp/kg349/ar/hf`.
  - The architecture is a hybrid of linear and full attention (64 layers, 4 KV heads), so the KV cache per token is small.
- **Stack:** vLLM 0.33.0, torch 2.14 (cu130), driver 595.84. The venv is `/usr/xtmp/kg349/ar/venv`; always `source env.sh`.
- **Required flags and env:**
  - `--kv-cache-dtype bfloat16`: the checkpoint ships an FP8 KV scheme, which on Ampere needs a FlashInfer JIT build that fails here.
  - `--attention-backend TRITON_ATTN`.
  - `--limit-mm-per-prompt '{"image":0,"video":0}'`: the model is multimodal and images aren't used.
  - `VLLM_USE_FLASHINFER_SAMPLER=0`.
  - **No system CUDA module**: system nvcc 12.9 doesn't match torch's CUDA 13 build, so no JIT compile can work.
- **Capacity:** startup takes 160 s. Each PP = 2 replica has a 278k-token KV cache (10.2 GiB/GPU).
- **Throughput:** output tokens/s per replica (1024-token outputs, thinking off), measured by `probes/bench_vllm.py`:

  | Concurrency per replica | 16 | 32 | 64 |
  | --- | --- | --- | --- |
  | Output tok/s per replica | 399 | 655 | **852** |

  GPU SM utilization was 100% on all 4 GPUs. **Run each replica at about 64 in-flight requests**, which also satisfies the utilization policy.
- **Thinking mode:** 666 tok/s per replica at 32 concurrency, but **every request hit the 4096-token cap**. Qwen's reasoning on ideation prompts exceeds 4k tokens.
  - If thinking stays on, set `max_tokens` to 12–16k so the final answer isn't cut off, and log how often the cap is hit.
  - Measure thinking length in the smoke run, then decide on/off per call type.
- **KV-cache budget rule (learned in overnight smoke 1, 2026-11-08):** keep requests in flight × (prompt + max_tokens) at or below about 278k tokens per replica.
  - Smoke 1 ran thinking on (max 16k) with 96 requests per replica. The cache filled to 98%, vLLM thrashed, and throughput fell to 51 tok/s with zero calls completing.
  - Size `llm.max_inflight` per run accordingly, summed over all runs that share a replica.
- **Ideas per hour per replica, at about 850 tok/s ≈ 3M tokens/h:**
  - thinking off (about 5k tokens per idea, all calls included): about 600;
  - thinking on (about 15k tokens per idea): about 200.
  - Either way, far more than the sampler needs.
- **Claude Code:** `claude -p` works from compute nodes (CPU and GPU) with the subscription login. 3 concurrent sessions worked fine (Sonnet 5.5).

- A 4-bit 27B model is about **15–17 GB** of weights. On one 24 GB A5000 that leaves only about 6 GB for KV cache, which is too little for many concurrent multi-turn requests. So split each replica across 2 GPUs with **PP = 2**: about 8 GB of weights and about 13 GB of KV per GPU.
- **Default layout: one 4×A5000 node with 2 replicas, each PP = 2**, on ports 8000 and 8001.
  ```bash
  source env.sh   # see §1.0 for why each flag is needed
  CUDA_VISIBLE_DEVICES=0,1 vllm serve $AR_MODEL --served-model-name qwen --pipeline-parallel-size 2 \
    --max-model-len 32768 --gpu-memory-utilization 0.90 --max-num-seqs 64 --enable-prefix-caching \
    --kv-cache-dtype bfloat16 --attention-backend TRITON_ATTN \
    --limit-mm-per-prompt '{"image":0,"video":0}' --port 8000
  CUDA_VISIBLE_DEVICES=2,3 vllm serve ... --port 8001
  ```
  In the main experiment, **each arm gets its own replica** (§4). That gives both arms exactly equal LLM compute and wall-clock time, with no load-balancer needed.
- **Why PP and not TP.** No strong argument for TP applies here:
  - TP all-reduces twice per layer per token, and on PCIe without NVLink that cost is large.
  - PP only sends activations once per stage boundary.
  - TP's advantages are lower latency per request and a pooled KV cache. This workload needs throughput: dozens of independent ideation, tagging and judging requests in flight. With enough requests in flight, PP keeps both stages busy.
  - **PP's one requirement is enough concurrency**: enough requests in flight that every stage always has a batch to work on. Run the ideator with ≥ 16 concurrent conversations per replica.
  - Fallback: if one arm-replica is too slow, use a single PP = 4 server shared by both arms. The client then alternates requests between arms to keep compute equal.
- Ampere uses the AWQ/GPTQ **Marlin** kernels, which are fast. Avoid FP8 checkpoints, because Ampere has no native FP8.
- **Thinking mode** is set per call type via `chat_template_kwargs={"enable_thinking": …}` and frozen in the config. Start with it off for summary, tagging and assignment, and on (`max_tokens` about 8k) for ideation and judging. The judge's setting is chosen during calibration (§3.3).
- Benchmark tokens/s per replica at 16/32/64 concurrent requests on Day 1. That number sets how many ideas each run can produce (§4).

---

## 2. Pipeline architecture

```
┌───── Slurm job A: 1 fitz node, 4×A5000, long-lived ───────────────────────────────┐
│ vLLM replica 0 (GPUs 0,1, PP=2) :8000     vLLM replica 1 (GPUs 2,3, PP=2) :8001   │
│ orchestrator.py per run: ideator · tagger · surrogate judge · realjudge queue     │
│ journal: /usr/xtmp/kg349/ar/runs/<run>/journal.json   (filelock, 30-min claims)   │
└───────────────────────────────────────────────────────────────────────────────────┘
Real eval (rare, ~36 total): sbatch 1×A5000 → claude -p implements → task harness
  runs + scores → runs/<run>/real/<id>/result.json → ingested into the journal
```

- **Journal.** All journal writers run on one host (job A) under `filelock`, because locking across nodes on NFS is unreliable. Real-eval jobs only write `result.json` into their own idea directory, and the orchestrator ingests it. Writes are atomic (tmp + fsync + rename), with a `journal.jsonl` event log kept as a crash backup.

### 2.1 Resource-utilization policy (admins: no idle GPUs or CPUs)
- **The vLLM node only exists while a pipeline run is feeding it.** The orchestrator keeps ≥ 16 requests in flight per replica. A watchdog `scancel`s the job if a replica sees no requests for 10 min. Release the node between phases (train runs, calibration, test runs) instead of holding it.
- **Real-eval GPU jobs contain training only.** Claude's coding, repair and review run on CPU nodes. The GPU job starts the harness immediately and exits when it finishes.
- **Harnesses must keep the GPU busy:** load CIFAR into GPU memory once, do augmentation on the GPU, use large batches and AMP. They must not depend on a CPU dataloader. Check the starter's `nvidia-smi` utilization during harness prep, targeting > 80%.
- **Every GPU job logs `nvidia-smi dmon`** to its run dir, as evidence and so problems can be caught.
- **Request only what is used:** cores sized to the job (`-c 4`–`8` for evals), memory sized to it, and tight `-t` limits so a hung job dies quickly.
- **Probes and benchmarks exit as soon as they finish.** Interactive GPU sessions are never left open.

### Code layout
```
ar/
  prompts.py      verbatim prompt strings from the spec
  templating.py   utility parser/evaluator
  journal.py      lock, load/save, claims, stale-claim reaper, status transitions
  llm.py          async OpenAI-compatible client (endpoint per arm), retries, token accounting
  sampler.py      features, weighted Bayesian posterior, Thompson draws; "fixed" mode for the naive arm
  scoring.py      F with frozen b_t, confirmation call, calibration map
  workers/        ideator.py · tagger.py · judge.py · realjudge.py
  orchestrator.py one run per config; --resume
configs/          YAML per (task, arm); copied into the journal at run start
realjudge/        claude_prompt.md · review_prompt.md · job.sbatch
tasks/<task>/     TASK.md · facts.yaml · prepare_data.py · harness/ (read-only) · starter/   ← built last (§5)
analysis/         scripts reading journals only
slurm/  tests/
```

### Implementation notes (the parts that are easy to get wrong)
- **Templating.** Parse into an AST, then evaluate innermost first. Don't use regex replace in a loop, because substituted idea or tag text can contain `{{`. Each idea gets its own seeded RNG, and every draw is logged. Decimal N means a fraction of the items (min 1, capped at the number available). Sampled items are printed in creation order. Unit-test the nested forms from the spec, plus an idea whose text contains `{{prompt}}`.
- **Ideation.** Turn-1 order: tops (References, RefIdeas) → base turn 1 → bottoms (References, RefIdeas, tag slot, NoIdea). Appends go on turn 1 only. The idea is the final turn's output, with `<think>` stripped (the raw text is kept). Availability is decided from one journal snapshot taken at batch-request time.
- **Sampler.**
  - Features, in order: `[1, BetterGeneric, BestSample1, BestSample2, RandSample, NewSampleCritique, TopicIdea, NoTag, UseTag, NoIdea, References, RefIdeas]`.
  - Observations = ideas with status ≥ scored, ordered by scored time. Weight w_j = γ^(n−1−j).
  - Posterior: `Σ = (XᵀWX/σ² + Λ⁻¹)⁻¹` via Cholesky, `μ = Σ XᵀWy/σ²`. Draw `b̃ = μ + chol(Σ)z`.
  - Base slot: argmax over available bases (Generic = 0). Tag slot: argmax of {none = 0, NoTag, UseTag}. Binary slots: used iff the draw is > 0.
  - Log every refit and every draw.
- **F.** Comparison set = ideas created earlier that already have a score and are not failed. b_t is the best score holding tag t, or s_base if none. If F > 0, make a second judge call, take the mean, and recompute F against the **same frozen b_t**. F then never changes. The scores used inside F are the **calibrated** surrogate scores (§3.3).
- **Tagging.**
  - Trigger at 20 waiting ideas or 10 minutes. Creation uses S = 10, r = 2 (no duplicate idea inside a subset), then merges in groups of N = 4 down to K = 4 lists, then the JSON writer.
  - The writer's output is parsed from the last ```json block, with one retry. Assignment matches tag names exactly, with one retry before the idea is marked failed.
  - **Log the tag-pool size per cycle.** Tags that open new categories make F > 0 easy, and the report has to discuss this.
- **Judge.** Parse the last `\boxed{}`, clamp to [0, 1], allow 2 retries. Use temperature 0.6–0.7 so the confirmation call is an independent draw.

### Real eval procedure (per idea, 1×A5000 job)
1. **Implement** (CPU job): run `env -u ANTHROPIC_API_KEY timeout 40m claude -p "<prompt>" --model sonnet --max-turns 60 --output-format json --dangerously-skip-permissions` inside the idea dir.
   - The prompt contains TASK.md, the idea, and the harness contract: write `solution/`, smoke-test with `harness/run.sh --smoke`, don't touch `harness/`.
   - It also says: **implement the idea faithfully; don't swap in a different method.**
2. **Score:** the pre-built harness runs the solution under the task's proxy budget with a fixed seed and computes the metric. The test data never exists in the idea dir.
3. **Repair:** if the run crashed, make one `--resume` repair attempt, then score again.
4. **Review:** a short read-only Claude call grades adherence to the task constraints (1/0.5/0) and fidelity to the idea (1/0.5/0). The harness's `check.py` checks the hard constraints mechanically.
5. **real_score** = `clip((m − m_floor)/(m_ref − m_floor), 0, 1)`, set to 0 if adherence = 0. This answers the spec's open question. Usage (`num_turns`, `usage`, cost) is logged from the JSON output.

**Split CPU and GPU work (no idle GPUs; see §2.1).** Steps 1, 3 and 4 run in a **CPU-only** job (`-p compsci`), with a CPU smoke test on a tiny data slice. Only step 2 holds an A5000, through a separate GPU job submitted when the solution is ready. A Claude session never holds a GPU while it is writing code.

Because real evals are so few, the spec's "every 2 h cycle" doesn't matter. Real evals are triggered manually or by script at the points defined in §4.

---

## 3. Tasks: train/test split, harnesses, calibration

### 3.1 Task selection: shortened Type I/II tasks, all CIFAR-based
One dataset download serves every task, epochs are fast, and the metrics are objective.

Seven tasks, none of which needs ImageNet. The ImageNet-1k download is gated and large, so the ImageNet tasks (#1, #2, #3, #8, #20) are left out.

| Split | Task | Type | Proxy budget | Metric |
| --- | --- | --- | --- | --- |
| **Train** | #7 Activation Tuning (CIFAR-100, fixed 3-hidden-layer MLP) | I | 15 min, 1×A5000 | top-1 |
| **Train** | #15 Learning Without Forgetting (CIFAR-100, 10 sequential tasks, ≤ 1 MiB aux state) | I | 15 min | final top-1 over 100 classes, no task id |
| **Train** | #10 Label-Noise Survivor (CIFAR-10, fixed 40% noisy labels) | II | 15 min | clean test top-1 |
| **Test** | #18 Anytime Predictor (CIFAR-100) | I | 15 min | expected top-1 under a fixed interruption-time distribution |
| **Test** | #9 Tokenizer Transfer (GPT-2 small → new 32k Japanese tokenizer, fixed; 200k Wikipedia-ja rows) | I | 15 min | held-out perplexity per token, lower is better (`tasks/t09_tokenizer_transfer`; floor 52.5 ± 1.2, FVT probe 41.3 ± 0.1). Replaces #19 Data Ordering (no headroom: a loss-based selection method gave +0.08 pt) and then #2 Self-Regularizing (no headroom: two anchors at or below the starter) |
| **Test** | #13 Robustness without adversarial training (CIFAR-10) | II | 20 min | PGD-20 robust acc at ε=4/255 (lowered from 8/255 on 2026-11-08: at 8/255 honest methods fall in 0–3%), gated on 85% clean accuracy; 8/255 reported as secondary |
| **Test** | #16 Useful Synthetic Data (CIFAR-100, 10 real/class → ≤ 100 synthetic/class) | II | 15 min synthesis + fixed classifier training | mean accuracy of 3 fixed classifiers, one held out from Claude |
| *(optional)* | #4 Tiny Walker (BipedalWalkerHardcore, ≤ 700 params, ≤ 3.5 KiB) | I | 30 min CPU | mean return over fixed seeds |

- The split is by **task**, so the judge is calibrated on tasks it is never tested on. That makes the surrogate-validity result an honest out-of-sample number.
- Both splits contain both types: the train split has 2 Type I + 1 Type II, the test split has 2 + 2.
- Every task except #4 is built on CIFAR-10/100, so one data download covers them all. Download them early.
- The surrogate-only runs for all tasks share the same vLLM replicas, because ideation is throughput-bound and cheap.
- **Main risk:** each shortened budget must still separate good ideas from bad ones. During harness prep, set it by running the starter solution plus one stronger reference solution.

### 3.2 Real-eval budget (very small)
| Use | Count |
| --- | --- |
| Anchors (floor + ref per task; the ref is run once by Claude, then reused) | 7 tasks × 2 = 14 harness runs (not idea evals) |
| **Train calibration:** ideas from the train runs, stratified across the surrogate-score range | 3 tasks × 6 = **18** |
| **Test confirmation:** per test task and arm, the top 2 by surrogate score; plus 1 random idea per test task | 4 × 2 × 2 + 4 = **20** |
| **Total Claude implementation sessions** | **38** |

### 3.3 Judge calibration (train tasks only)
1. Run the full pipeline (arm A) on all three train tasks for about 6–8 h. This needs only vLLM.
2. Pick 6 ideas per train task, stratified by surrogate score (sextiles), and real-eval them.
3. Re-judge those same 18 ideas offline with a handful of judge variants:
   - thinking on or off;
   - with or without anchor facts in `{{facts}}` ("under this budget a standard X gets Y%, which is score 0.5"). The judge prompt itself stays the spec's Basic zero-shot prompt.
   Keep the variant with the highest **Spearman ρ** to the real scores.
4. Fit a simple **linear calibration** `s_cal = clip(a + b·s, 0, 1)` mapping that variant's scores to the real-score scale. This keeps s_base = 0.5 and s_min = 0.05 meaningful in real-score units.
5. Freeze the variant, the facts format and (a, b). Commit them before the test runs (pre-registration).

Overfitting risk: choosing among about 4 variants on 18 points can look better than it is. The held-out test ρ (§5, H2) is the honest number.

---

## 4. Main experiment (test tasks)

**Hypothesis H1:** at equal wall clock and equal LLM compute, the optimized pipeline produces better ideas than a naive pipeline that uses the same model.

| Arm | Definition |
| --- | --- |
| **A. Optimized** | Full spec pipeline: tagging, selective-improvement F, Thompson sampler, calibrated judge |
| **B. Naive** | Same Qwen, same judge and **the same prompt pool**. Each idea's prompt configuration is drawn uniformly at random from the spec's starting distribution (every base equally likely; each optional slot used half the time), and the sampler never learns (`sampler.mode: uniform`). Everything else runs identically (tagging, judging, F). The only difference between the arms is therefore whether the sampler learns, not which prompts are used. Corrected 2026-11-08: the earlier `fixed_generic` (Generic only) confounded the arm comparison with prompt choice. |

- Both arms run concurrently for the **same wall clock T ≈ 24–30 h**, each on its own PP = 2 replica. All 4 test tasks run at once on their arm's replica. That gives equal compute by construction.
- One run per (task, arm) is enough. Inference uses ideas as the units and within-task tests, plus a direction check across the 4 test tasks.
- At the end, real-eval the top 2 per (task, arm), plus 1 random idea per task (§3.2).
- Expected volume, from the Day-0 benchmark (§1.0): about 200 ideas/h per replica with thinking on, about 600 with it off. Over 24 h, split across 4 tasks, that is roughly **1,200–3,600 ideas per (task, arm) run**. That's more than needed, so T can shrink to about 12–16 h if the schedule gets tight.

---

## 5. Analysis

**Hypotheses and tests:** few, simple, and each one maps to a rubric line.

| # | Question | Test | Rubric line |
| --- | --- | --- | --- |
| **H1** | Does arm A produce higher-scoring ideas than arm B? | Per test task: **Mann–Whitney U** on calibrated surrogate scores of the top 10% of ideas per arm. **Primary: equal number of ideas.** Both arms run to the same wall-clock limit, so GPUs stay busy; the analysis then truncates each arm to its first N scored ideas, with N = the smaller arm's count. **Secondary: equal compute**, using all ideas in the same wall clock. Best-so-far plots vs idea index (primary) and vs wall clock (secondary). Decided 2026-11-08. Across tasks: in how many of the 4 test tasks A > B (direction check). Confirm with the real evals: mean real score of the top 2, A vs B (descriptive; n is small) | Hypothesis testing; Results |
| **H2** | Does the calibrated surrogate predict real performance on unseen tasks? | **Spearman ρ** with 95% bootstrap CI: on train (18 ideas) and held-out test (20 ideas) | Validity |
| **H3** | Does a nonlinear model predict an idea's surrogate score from its 12 prompt features better than the linear model? Linear regression vs a small MLP. | Repeated 10-fold CV on the arm-A ideas, with a **paired t-test on per-fold MSE** | "Compare ML algorithms (e.g., linear regression vs neural networks)" |

For **Type II**, also report a simple diversity number for each arm's top ideas: the number of distinct tags covered by the top-20 ideas. Run one post-hoc assignment pass over arm B's ideas with arm A's final tag pool so both arms use the same tags. This is descriptive; no test is needed.

**Feature importance:**
- The sampler's final **posterior mean ± 95% credible interval** for each subprompt's effect on F. This is feature weighting that comes straight out of the model.
- **Ablation**, given by arm A vs arm B (all subprompts + sampler vs none).
- Optionally, an OLS of the calibrated score on the 12 features to show which subprompts raise quality, as opposed to raising F through variance.

**Validity points to state in the report:**
- The harness computes the scores, and the Claude reviewer applies the adherence gate.
- Tasks are split train/test, and the calibration is frozen before the test runs.
- Pre-registration commit.
- Tag-pool growth.
- Anchor seed noise (3 seeds of the ref solution) compared with the observed differences.

**Run outputs (from the spec):** real-judged ideas ranked by score, the best idea per tag, and the final μ/sd per subprompt.

---

## 6. Build order and timeline

The eval harnesses are built **last**. Everything before them needs only vLLM.

| When | Work | Exit criterion |
| --- | --- | --- |
| **Sun 11-08 – Mon 11-09** | xtmp env; 4-bit Qwen, 2 × PP = 2 replicas; tokens/s benchmark; `claude -p` check on a compute node | Both endpoints answer; throughput known |
| **Mon 11-09 – Tue 11-10** | `templating`, `sampler`, `scoring` + unit tests; workers; orchestrator; naive arm config | A 2 h smoke run on a train task produces scored ideas, tags and sampler refits |
| **Tue 11-10** | Launch the **train-task pipeline runs** for all 3 train tasks (surrogate only, about 8 h) | Train ideas ready to sample |
| **Tue 11-10 – Wed 11-11** | **Last build step: per-task eval harnesses**, train tasks first, then test tasks (data prep, fixed parts of the model or training, `run.sh`, `eval.py`, `check.py`, starter); anchors; realjudge job script | Each harness scores its starter end to end |
| **Wed 11-11** | 18 train real evals, then judge calibration; freeze; **pre-registration commit** | (variant, a, b) committed |
| **Wed 11-11 evening → Fri 11-13 morning** | **Test runs** for all 4 test tasks, arms A and B concurrently, T ≈ 24–30 h. Finish the test-task harnesses during this window if needed, since they're only used at the end | Journals complete |
| **Fri 11-13** | 20 test real evals | All results ingested |
| **Fri 11-13 – Sat 11-14** | Analysis scripts, figures | Tables and plots produced from scripts |
| **Sun 11-15 – Mon 11-16** | 4-page A4 report, README (PDF/DOC), code link, contributions section | Submitted with margin |

---

## 7. Risks

| Risk | Mitigation |
| --- | --- |
| No good 4-bit checkpoint exists | Quantize it yourself with AutoAWQ/llm-compressor on a 4×A5000 job (about 1–2 h), or use a community GPTQ-int4 build |
| One PP = 2 replica per arm is too slow | Lower max tokens or turn thinking off for ideation, or use a shared PP = 4 server with alternating arm requests |
| The home file quota fills | Everything on xtmp; prune Claude transcripts |
| Surrogate scores cluster high | Calibration step (§3.3); report the score histogram |
| Claude "improves" the idea | Faithfulness instruction plus the fidelity grade; report it |
| 4 test tasks still give limited generalization | State it plainly. The statistics are within-task at the idea level, plus a 4-task direction check |
| 7 harnesses are a lot to build | Build the train-task harnesses first; the test-task harnesses are only needed for the final evals |

## 8. Assignment-fit gaps to settle

1. **"Human information processing" data.** Frame the analysed data as a text corpus: generated research ideas judged against human-written task specs. Check this with the instructor early.
2. **Open datasets.** The task set is self-authored, but all the data is open (CIFAR-10/100). Release the tasks, harnesses and journals.
3. **Contributions section.** Natural split: serving/infra, pipeline core, harnesses/real evals, analysis/report.

---

## 9. Revision v3 (2026-11-09): judge-strategy study and larger runs

**Why.** The Basic zero-shot judge tracks real outcomes poorly: pooled ρ ≈ 0.2 or below on 18 train ideas, and single-call scores are mostly noise. The user also corrected the scope: every **LLM-only** judging strategy from the draft spec belongs in the experiment. The CPU soundness-check tier stays out. Two more facts:
- the presentation's own method is "collect a pool of evaluated ideas, test LLM evaluation strategies, measure Spearman to the true evaluation";
- we are about 2.5 days ahead of §6.

### 9.1 Judge strategies (all LLM-only; draft spec "Surrogate Judging" graph)
All strategies use the General System Prompt. Each is run with and without anchor facts in `{{facts}}`; facts are judge-only and never shown to ideators.

| ID | Strategy | Prompt | Absolute score for F? |
| --- | --- | --- | --- |
| J1 | Basic zero-shot, K=1 | spec "Basic" | yes |
| J2 | Basic zero-shot, mean of K=5 | spec "Basic" | yes |
| J3 | Arbitrary Rubric (zero-shot) | spec "Arbitrary Rubric": builds its own rubric, averages it | yes |
| J4 | Few-shot ("Basic FS") | spec "Basic FS": previous ideas with their judging results. Offline: same-task real-evaluated ideas, leave-one-out | yes |
| J5 | Ranking-based | several new ideas + reference ideas with known positions (the task's starter = 0, reference method = 1); score each so the whole set ranks correctly | yes (anchored by the references) |
| mod | Thinking on/off | the best one or two strategies only (6× tokens, ~40% truncation at 12k) | — |

### 9.2 Pools and evaluation
- **Train pool:** 60 real-evaluated train ideas (20 per task). The 18 we have, plus 42 stratified across the s_cal range from the re-pilot runs (train_*_s3).
- **Test pool:** the test real evals (§9.3), about 64 ideas.
- **Metric:** Spearman ρ between strategy score and the unclipped normalized real value u = (v − floor)/(ref − floor), pooled and per task, with a 95% bootstrap CI.
- **Comparison test:** paired bootstrap of ρ(strategy) − ρ(J1) over ideas, with Holm correction across strategies. This is the "statistical comparison of methods" result.
- **Selection:** pick the pipeline judge by train-pool ρ, then freeze it (pre-registration note) before the test runs. Every strategy is then re-scored on the test pool, so the out-of-sample comparison covers all of them, not just the chosen one.

### 9.3 Larger runs
- **Test runs:** 4 tasks (t18, t09, t13, t16) × 2 arms × **2 seeds** × **24 h**, on 4 nodes (2 runs per replica, max_inflight ~7 per run). H1 then has seed-level replication, and best-so-far curves get two seeds.
- **Test real evals:** per run, top 3 by s_cal + 1 random gives 4 × 16 runs = **64**. This allows **H1-real**: real score of each arm's top ideas, A vs B, Mann–Whitney per task, plus a direction count. That tests the research question on real performance, not only on the surrogate.
- **Train runs:** optionally resume train_*_s3 to 10 h (1 node, +5 h) for more H3/feature-importance data and a longer H1 pilot.
- **Budget:**
  - Claude sessions: about 42 (train pool) + 64 (test) ≈ 106 more. They have been short, 1–5 min of Sonnet each.
  - A5000 time: real evals about 30 GPU-h; test runs about 384 GPU-h.

### 9.4 Timeline (supersedes §6 from Mon 11-09)
| When | Work |
| --- | --- |
| Mon 11-09 | Queue the 42 train real evals. Implement J1–J5 (offline evaluator + pipeline judge modes) with tests |
| Mon night | Offline judge study on the train pool (dedicated vLLM job); select + freeze; pre-registration note |
| Tue 11-10 → Wed 11-11 | Test runs, 24 h, 4 nodes |
| Wed 11-11 → Thu 11-12 | 64 test real evals; re-score all strategies on the test pool |
| Thu 11-12 → Fri 11-13 | Analysis: H1 surrogate (equal ideas) + H1-real, H2 strategy comparison, H3, feature importance |
| Sat 11-14 → Mon 11-16 | Report and submission, with margin |

### 9.5 More train tasks (user, 2026-11-09)
Judges calibrated only on CIFAR classification could overfit that one task family, so two more train tasks are being built, each with a required headroom probe:
- **t04_tiny_walker** (#4, Type I): RL, CPU only, < 700 params / 3.5 KiB, mean return on private eval seeds. The real-eval chain gets a CPU-only path, so these runs hold no GPU.
- **t11_fewshot_seg** (#11, Type I): Pascal VOC 2012, 5 images/class, one fixed ImageNet backbone (no segmentation/detection weights), mIoU, 15 min on an A5000.

If both pass the headroom check, the train split has 5 tasks (2 Type I, 1 Type II, plus these 2 Type I). Each new task gets:
- pipeline train runs (optimized + uniform naive);
- 20 real evals for the train pool, so the pool grows to about 100 ideas.

The judge study then also reports **leave-one-task-out** ρ: choose the strategy on 4 train tasks, score it on the 5th. That is a direct check that the strategy choice doesn't overfit to one task family. This pushes the judge freeze to about Tue 11-10 evening and the test runs to Wed 11-11, which is still on the original §6 schedule.

### 9.6 Judge selection criterion (pre-registered 2026-11-09)
- **Criterion: mean within-task Spearman ρ** vs u, not pooled ρ. Pooled ρ also rewards between-task calibration, which the pipeline never uses, since it only ranks ideas within one run.
  - Example: J4r has pooled 0.34 but ≤ 0.19 within every task.
- **Comparison test:** a paired bootstrap stratified by task (Δ mean within-task ρ vs J1), with Holm correction.
- **LOTO** uses the same criterion.
- **On 60 CIFAR train ideas:** J2_facts (Basic ×5 + anchor facts) is best (0.33; 0.40 / 0.46 / 0.14). It is chosen in 2 of 3 LOTO folds, held-out ρ 0.40 / 0.46 / −0.25.
- **Final pick:** after the 5-task pool (+ t04, t11) is re-run.
- **Calibration: none (identity a=0, b=1)**, decided 2026-11-09.
  - Monotone maps don't change ρ or within-run rankings; they only move scores relative to F's thresholds (s_base 0.5, s_min 0.05).
  - The regression fit (b ≈ 0.54) caps s_cal below s_base, so no tag-first idea could ever have F > 0.
  - Quantile-mapping onto real scores would tie 43% of ideas at 0.
  - Raw J2_facts scores spread 0.20–0.71 (10th–90th percentile), and 33% are > 0.5. The anchor facts already define the scale (starter = 0, reference = 1).
  - Nothing is fit, so nothing overfits the train pool.

### 9.7 FROZEN judge (2026-11-10, before any test run)
- Strategy **J2_facts**: `judge: {strategy: basic_k, k: 5, anchor_facts: true}`, thinking off, calibration identity (a=0, b=1).
- Chosen by the pre-registered criterion (mean within-task ρ, no-thinking variants) on the 100-idea, 5-task train pool. $AR_ROOT/analysis/judge_study/judge_study.md: 0.21; per task t04 0.15, t07 0.40, t10 0.46, t11 −0.10, t15 0.14. LOTO picks it in 5/5 folds, held-out mean 0.21.
- vs J1_nofacts: Δ within 0.17 [0.02, 0.32], p_holm 0.48.
- Thinking variants were excluded for cost (~49k tokens per idea for J2_facts_think). Their train-pool runs were incomplete (about 60/100 ideas) after the study job hit its 3 h limit.
- Test configs: configs/test/*.yaml (scripts/make_test_configs.py). H2 on the test pool re-scores ALL strategies.
