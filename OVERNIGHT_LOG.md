# Overnight log (2026-11-08 → 11-09)

Autonomous work while the user sleeps. Every job has a hard Slurm `-t` limit and exits when its work is done, so nothing idles if the Claude session dies.

## Timeline
- 07:33 submitted live smoke pipeline (3 train tasks × 2 arms, 45 min, -t 1:40)
- 07:34 submitted floor-anchor array (7 tasks × seeds 0-2, ≤4 GPUs at once, -t 0:45 each)
- 07:35 BLOCKED: realjudge submit (headless claude with --dangerously-skip-permissions) denied by the permission classifier; not worked around. Ref anchors + chain test need user approval. Dry-run dir left at $AR_ROOT/runs/anchor_ref_t07_activation_tuning (nothing submitted).
- 07:47 smoke1 CANCELLED after ~12 min: with thinking on (ideation max 16k, judge 12k) and 3 runs×32 in flight per replica, KV cache hit 98% and vLLM thrashed (51 tok/s vs 852 benchmarked, 0 completed calls). Root cause: in-flight × context ≫ 278k-token KV cache.
- 07:48 submitted smoke2: same 3 train tasks, optimized arm; replica 0 thinking OFF (20 in flight/run), replica 1 thinking ON capped (ideation 8k, judge 6k, 9 in flight/run); 45 min
- 08:xx smoke2 results at 20 min (then cancelled):
  - thinking ON (ideation max 8192, judge 6144, 27 in flight): 545 tok/s, but **78–95% of ideation calls truncated at the cap** → thinking is not viable here (Qwen's thoughts exceed 8k).
  - thinking OFF with 60 in flight × max_tokens 4000: still KV-thrashed (97.5% KV, 50 tok/s). The KV rule bites even without thinking.
  - **Decision (autonomous, reversible):** thinking OFF for all call types; ~16 in flight per run (≈48 per replica, ≈220k tokens worst case).
- 08:10 submitted train-task runs segment 1 (t07/t10/t15 × optimized/naive, thinking off, 16 in flight/run, max_hours 1.0; extend with --resume if healthy)
- 08:32 train s1 CANCELLED at 20 min (runs kept under $AR_ROOT/runs/train_*_s1, configs moved to configs/train/aborted/):
  - optimized replica KV-thrashed again (99% KV, 51 tok/s) at 48 in flight: optimized prompts are longer (appends).
  - judge truncated 25/31 calls at max_tokens 1500 (no \boxed → retries/fails); ideation truncated up to 31% at 4000 on Generic prompts.
  - Found a spec-level scaling bug: fractional list sizes (NoIdea's {{idea_summaries;{{rangefloat;0.1;1}}}}) grow with the run → prompts would pass 32k context at ~1000 ideas. **Deviation:** fractional draws capped at 30 items (ar/templating.py MAX_FRACTION_ITEMS; test added). Integer N unchanged.
- 08:32 submitted train s2: thinking off, 10 in flight/run, judge 4096 / ideation 6000 tokens, 1 h segment.
- floor anchors done (21/21 runs valid; outputs in $AR_ROOT/anchors/floor/). Written to each tasks/*/task.yaml as `floor` (starter mean, seeds 0-2) + `floor_seed_sd`:
  t07 0.3230 (sd 0.0015) · t10 0.6861 (0.0046) · t15 7.563 (0.187) · t13 0 (0) · t16 0.1695 (0.0021) · t18 0.4430 (0.0014) · t19 0.7953 (0.0013).
  Seed noise is tiny everywhere (t19 ±0.13 pt), so even small real-eval differences are detectable. `ref` stays null (needs the blocked Claude chain).
- 08:54 s2 healthy at 20 min (KV ~50%, ~885 tok/s total, 0 truncation, 106-181 ideas/h/run). Queued resume job (afterany dependency) extending all 6 train runs to max_hours 5.0 cumulative, -t 4:40.

## 10:16 — resume job 13373681 running (fitz-12)
- All 6 train s2 runs resumed, heartbeats live, 0 errors in any orchestrator log.
- Replica 8000 (optimized arms) KV climbing 60%→86% as tag pools grow, gen ~330 tok/s; 8001 (naive) KV 43%. No vLLM preemptions. Monitoring; will cancel only if preemption/thrash begins.
- Example status (t15): optimized 64 scored, s_cal max 0.665; naive 84 scored, max 0.75. Truncation <1% overall.

## 10:52 — health OK; mid-run pilot analysis
- Job 13373681 at 36 min: no preemptions; KV 71% (8000) / 54% (8001); 30 running per replica.
- Pilot `analysis/analyze.py` on the 3 train pairs (output in $AR_ROOT/analysis/pilot_midrun):
  - t07: top-10% 0.690 opt vs 0.704 naive, p=0.80
  - t10: 0.834 vs 0.833, p=0.23
  - t15: 0.586 vs 0.652, p=0.998
  - Direction: optimized wins 1/3. H3: linear beats MLP on t07/t10 (n≈90); tie on t15.
- Observation: naive generates ~25–50% more ideas per hour at equal compute, because its prompts are shorter. That is part of what H1 measures. Early-run data, so no conclusions yet. The analysis script works end to end on real journals.

## 11:42 — job 13373681 at 1h26: healthy (no preemptions, KV 72%/64%, no tracebacks).

## 12:37 — job 13373681 at 2h22: healthy (no preemptions, KV 75%/62%).

## MORNING SUMMARY (written 12:37; final numbers appended when runs finish ~13:45)
**Job states:** only ar-pipeline 13373681 is left. It is bounded by `-t 04:40`, the orchestrator's max_hours 5.0 (cumulative), and the watchdog, and it tears itself down when the runs finish. No other ar- jobs are queued.

**Autonomous decisions and deviations:**
- Thinking mode is OFF for all LLM calls: 78–95% of calls truncated even at 8k tokens.
- max_inflight is 10 per run, keeping KV under ~80% and avoiding thrash at ~51 tok/s.
- Token limits: judge max_tokens 4096 (it truncated at 1500); ideation 6000.
- `ar/templating.py`: added MAX_FRACTION_ITEMS=30, a spec deviation. Fractional list sizes grew with the run and would have overflowed the 32k context. A test was added.
- Floor anchors (starter, seeds 0–2) were written into every `tasks/*/task.yaml`. `ref` is still null.
- The s1 train runs were aborted (configs in `configs/train/aborted/`). The s2 runs are the real pilot.

**Findings:**
- Pipeline mechanics all work on real vLLM: ideation, tagging, judging, F/confirmation, refits.
- Mid-run pilot H1: optimized wins only 1/3 train tasks. Naive gets more ideas per hour from its shorter prompts.
- H3: linear ≥ MLP.
- The judge invents its own score scales, which argues for the anchor-facts calibration variant.

**Needs your approval / decision:**
1. The Claude real-eval chain. `realjudge submit` from headless claude was blocked by the permission classifier; I did not work around it. To run it: `python scripts/make_anchor_ref.py` then `python -m ar.realjudge submit --run anchor_ref_<task> --ids i000000 --force`. This provides the `ref` anchors.
2. The spec's "uniform at start" is false under its own model: Generic is chosen 1/64 of the time, and the tag slot ~75%.
3. Claude CPU steps are delayed by queue waits; they could run inside the cpu-kg349 allocation instead.
4. t13: keep the 85% clean floor plus strengthened PGD?
5. t19 effects may be small; seed sd is ~0.13 pt.
6. Ask the instructor whether the "human information processing" framing is acceptable.

## Final (job 13373681 COMPLETED 14:09, all 6 runs stopped by max_hours ~5.2–5.5 h, no tracebacks; GPUs released)
- Runs scored ideas, optimized/naive: t07 265/304, t10 269/357, t15 259/296.
- H1 (top-10% s_cal, opt vs naive): t07 0.675 vs 0.728; t10 0.819 vs 0.823; t15 0.585 vs 0.645. Optimized wins 0/3 (all p≥0.7).
- Naive has more ideas per hour (67–84 vs 51–52) and more tag diversity in its top 20.
- H3: linear beats MLP on all 3 tasks (p<1e-3, optimistic paired t).
- Output: $AR_ROOT/analysis/pilot_final.
- Implication to discuss: under equal compute, the Thompson-sampled prompt overhead (longer prompts → fewer ideas) currently costs more than the selection gains. Possible levers: an equal-ideas comparison, shorter appends, or more sampler warmup. The surrogate scores are uncalibrated; real-eval is still pending approval.
- Feature importance: no subprompt's posterior 95% CI excludes 0 on any train task (the sampler learned no confident weights in about 260 ideas).

## 2026-11-08 afternoon — probes for decisions #4 (t13) and #5 (t19)
Probe sources are in $AR_ROOT/probes/src; outputs in $AR_ROOT/probes/. All probe jobs are finished.
- **t13.** Robust accuracy, PGD-20 with CE+margin attacks, counted correct only if correct at every iterate, attack step = eps/4:

  | eps | starter (3 seeds) | Gaussian noise aug σ=0.1 (clean 86.7%) |
  | --- | --- | --- |
  | 1/255 | 0.48 | 0.74 |
  | 2/255 | 0.085 | 0.61 |
  | 4/255 | 0.0003 | 0.30 |
  | 8/255 | 0 | 0.027 |

  At 8/255 the useful range is compressed into 0–3%, which is a weak metric.
- **t19.** Online batch selection by loss vs the uniform starter:

  | steps | selection | starter |
  | --- | --- | --- |
  | 6000 | 0.7965 / 0.7958 | 0.7953 ± 0.0013 |
  | 1500 | 0.7693 / 0.7704 | 0.7693 / 0.7705 |

  At 1500 steps the selection run really did change the presentation counts (gini 0.13 vs 0.015), and still no gain. The known-good method gives about 0 effect, so ref ≈ floor and real-score normalization breaks.

## 2026-11-08 afternoon — decisions applied (user approved #4, #5)
- **#4 t13.** Scored eps is now 4/255 (step 1/255). 8/255 is reported as `robust_acc_eps8_secondary` and not scored.
  - Re-measured floor: 0.000333 ± 0.000289 (seeds 0–2). Noise-aug probe through the real harness gives 0.2975.
  - Backup of the old evaluator: $AR_ROOT/probes/src/evaluate_t13_eps8_backup.py.
- **#5 t19.** Being replaced by t02_self_reg: #2 on CIFAR-100, 5 images per class, top-5. A harness agent is building it.
- **Real-eval chain no longer uses --dangerously-skip-permissions.**
  - Claude Code's built-in sandbox fails on this cluster: apply-seccomp needs a nested userns, which the kernel refuses.
  - Instead, `realjudge/jail.sh` (one bwrap layer) and `claude -p --permission-mode dontAsk --permission-prompts none --settings <per-session allow/deny, read-only>`. The reviewer also runs `--restricted` with Read/Glob/Grep only.
  - The CPU smoke and the GPU full run (solution code) run in harness mode: read-only filesystem except the idea dir, repo/~/.claude/other runs hidden, no network.
  - Breakout test: a private-file read via python saw no such file; the repo read was denied; writes outside the idea dir were discarded; torch and training data were OK.
  - Jailed CPU smoke (t07) and GPU smoke (t13) are valid. 193 tests pass. `AR_NO_JAIL=1` is for unit tests only.
- Residual risks:
  - Claude-mode network is open, since the API needs it.
  - Claude's Bash could read ~/.claude credentials: the Read tool is denied, but a python subprocess is not.
  - In harness mode, solution code can see private/ during eval, because the eval needs it. Static scan and Claude review are the guard.
- **#3 note.** The CPU partition filled up this afternoon (52 idle cores, jobs pending on Priority), so the Claude CPU steps may queue after all.

## 2026-11-08 evening — real-eval chain live (user said "go")
- t07 anchor chain is end-to-end OK: implement 65 s (4 turns), GPU 14 min, review 11 s.
  - Value 0.4154, valid, adherence 1, fidelity 1. Set as t07 `ref` (floor 0.323).
- Fix: the claude-mode jail now mounts random decoys over private/ and the raw CIFAR-100 test pickle (`scripts/make_decoy_private.py`, re-run it after adding a task).
  - Before the fix, Claude's own cpu-smoke self-tests failed because they couldn't find test files.
  - All 7 harnesses' cpu-smoke now runs valid inside the claude jail, at chance accuracy.
- t02_self_reg harness done: floor 0.3109 ± 0.0021 (top-5). Plan table updated; t19 marked `split: dropped`.
- Submitted anchor chains for t10, t15, t16, t18, t13, t02.

## 2026-11-08 evening — H1 primary = equal number of ideas (user decision)
- `analysis/analyze.py`: both arms run to the same wall clock. H1 is computed on the first N scored ideas per arm (N = min) as primary, and on all ideas (equal compute) as secondary. Holm correction for each, a direction count for each, and a 2-row best-so-far figure.
- Re-ran on the train pilot ($AR_ROOT/analysis/pilot_final_v2). **Naive still wins 0/3 at equal ideas:**

  | Task | top-10% mean, opt vs naive |
  | --- | --- |
  | t07 | 0.675 vs 0.731 |
  | t10 | 0.819 vs 0.826 |
  | t15 | 0.585 vs 0.648 |

  So the earlier "fewer ideas per hour" explanation was wrong or incomplete: the optimized arm's ideas themselves score lower under the (uncalibrated) judge.
- Candidate causes:
  - an uncalibrated judge with self-invented scales feeding the sampler;
  - the sampler has learned nothing confident (all posterior CIs include 0);
  - prompt appends lowering idea quality.
- t02 anchor #2 (starter net + K=2 in-forward flip/crop averaging, label smoothing, EMA, 690 s training) scored 0.3066 top-5. That is below the floor 0.3109 ± 0.0021. Both anchors are at or below the starter, so t02 shows no headroom so far. ref left null; decision pending from the user.

## 2026-11-08 evening — naive arm corrected (user)
- The naive arm must sample uniformly from the same prompt pool, not Generic only. Generic-only confounds H1 with prompt choice.
- Added `sampler.mode: uniform`, now the naive default and in configs/example_naive.yaml. `fixed_generic` is kept but no longer used. Test added; plan §4 updated.
- **All overnight H1 pilot numbers (train s2 naive runs = Generic only) are a Thompson vs Generic-prompt comparison, not H1. Retracted as H1 evidence.** The optimized s2 runs remain valid for calibration and H3; the naive s2 ideas can still be used as calibration ideas.
- User decisions: replace t02 with #9 Tokenizer Transfer (harness agent building `tasks/t09_tokenizer_transfer`), and re-pilot.
- Re-pilot submitted: job 13483910, `configs/train_s3/*` (t07/t10/t15 × optimized/uniform-naive, seed 3, 5 h).
- Calibration real evals submitted: 18 ideas, 3 stratified per s2 train run (seed 11), 54 rj- jobs.

## 2026-11-08 20:15 — calibration real evals done
- 18/18 valid, all adherence 1. The anchor's real_score clips many ideas to 0: t07 has 4/6 below the floor, and t15 has two "1.0" results with fidelity 0.5.
- **The current (uncalibrated, nothink, no facts) judge is anti-correlated with real results.**
  - Within-task Spearman vs raw value, n=6 each, none significant: t07 −0.47, t10 −0.54, t15 −0.26.
  - Pooled vs real_score: −0.21.
- Running `analysis/rejudge_calibration.py` as a step on pipeline node 13483910: 4 variants (thinking × anchor facts) × K=3 on the 18 ideas, throttled to about 8 extra in-flight requests.
- Selection criterion: pooled ρ vs unclipped normalized value.
- Anchor facts must be judge-only if adopted: cfg `facts` also feeds the ideation prompts.
- Re-pilot 13483910 is healthy at 20 min (KV ~60/63%, no preemption).
- t09_tokenizer_transfer harness done.
  - Starter 52.52 ± 1.24 ppl, FVT probe 41.31 ± 0.09: 9× the starter sd.
  - higher_is_better false; normalize() handles it via the sign of ref−floor.
  - About 540 MB of disk per run.
- t02 marked dropped, plan table updated, t09 anchor chain submitted (13484312–14).
- t09 ref = 44.7437 (anchor chain; probe was 41.31).

## 2026-11-08 23:10 — re-judge moved to a dedicated job
- The session dropped and killed the srun re-judge step; no summary was saved. It was also unusable anyway:
  - sharing the busy pipeline vLLM gave about 1,700 s latency per no-think call;
  - every thinking call hit the 1 h timeout.
- Now `slurm/rejudge.sbatch`: own 2 replicas, each serving one no-think and one think variant so neither idles, inflight 16 per variant.
- The script appends each call to `rejudge_calls.jsonl` and resumes on resubmit.
- Job 13485444, afterany: re-pilot 13483910. Partial logs from the shared run are in calib/aborted_shared_run/.

## 2026-11-09 early — re-pilot and re-judge finished (both while the session was disconnected; GPUs released)
- **Re-judge** (`$AR_ROOT/analysis/calib/summary.json`): 18 ideas × 4 variants × K=3, 0 failed calls. Pooled Spearman vs unclipped normalized real value:

  | variant | pooled ρ | linear fit |
  | --- | --- | --- |
  | think_facts | 0.22 | — |
  | nothink_nofacts | 0.18 | a=-0.019, b=0.52 |
  | think_nofacts | 0.11 | — |
  | nothink_facts | 0.07 | — |
  | original single pipeline scores | -0.24 | — |

  - None is significant (n=18).
  - Thinking variants truncate 35–48% at 12k tokens and use ~11k vs ~1.9k completion tokens per call (~6×).
  - Same config, K=3 mean (+0.18) vs single call (−0.24): single-call judge noise dominates.
- **Re-pilot H1** (corrected uniform naive; `$AR_ROOT/analysis/repilot_s3`), equal ideas:

  | task | top-10% opt vs naive | p (Holm) |
  | --- | --- | --- |
  | t07 | 0.681 vs 0.667 | 0.30 |
  | t10 | 0.827 vs 0.832 | 0.67 |
  | t15 | 0.594 vs 0.539 | 0.001 |

  - Optimized wins 2/3 at equal ideas; equal compute also 2/3. Ideas/h are now about equal (36–41 vs 38–39).
  - H3: linear ≤ MLP on all 3 (t10 tie).
- 2026-11-09 06:30 plan revision v3 (§9): judge-strategy study J1–J5, train pool 60, test runs 2 seeds × 24 h, 64 test real evals, H1-real.
- Resumed train_*_s3 to max_hours 10 (+5 h); 42 train-pool real evals queued; judge-strategy build agent started.
- 2026-11-09: building train tasks t04_tiny_walker and t11_fewshot_seg (harness agents); plan §9.5 (LOTO judge check; freeze ~Tue evening; test runs Wed).
- 2026-11-09: 42 s3 train-pool real evals done and ingested (all ok/valid/adherence 1). Train pool is now 60. Per-run Spearman of the in-pipeline J1 scores: 0.31, 0.21, 0.85, 0.07, 0.18, −0.32.
- 2026-11-09: judge-strategy build done (241 tests). Judge study launched on the 60-idea train pool (no-think ~1,008 calls, then think top-2).
- t11_fewshot_seg harness done: starter 0.2018 ± 0.0003 mIoU, probe (DeepLabv3+ decoder, fine-tuned backbone, strong aug) 0.5237 ± 0.0024. Anchor chain submitted.
- Leak fixes:
  - jail.sh now allow-lists $AR_ROOT top-level dirs: data, venv, hf, cache, uv-cache, uv-python, sandbox, private, decoy_*. Everything else is hidden in both modes, including *_dev probe solutions, harness_runs, analysis, logs, runs (except the idea dir), anchors, probes and tmp.
  - make_decoy_private.py swaps atomically (old kept as .old); decoy_raw is built once.
  - Verified in the jail. 243 tests pass.
- t11 ref = 0.4909 (anchor chain).
- t04_tiny_walker done (plain BipedalWalker-v3, 10 min, 16 CPU cores; Hardcore had no headroom): starter 5.99 ± 0.24, ARS probe 313.75 ± 12.4. CPU realjudge path added. Anchor submitted. CONTRACTS updated. Train runs for t04/t11 (opt + uniform naive, 6 h) launched.

## 2026-11-09 17:00 — judge study (60 CIFAR train ideas) done; t04/t11 real evals queued
- All jobs finished while the session was disconnected. t04 ref = 317.64 (anchor via the CPU path; probe 313.75).
- 40 real evals queued: 10 stratified per run from train_t04_{opt,naive}_s1 and train_t11_{opt,naive}_s1.
- ar/anchors.py: added t04/t11 starter descriptions and short reference names (used by facts and by the J5 references).
- Judge study ($AR_ROOT/analysis/judge_study/judge_study.md):
  - Pooled ρ, best first:

    | variant | pooled ρ |
    | --- | --- |
    | J5_facts_think | 0.46 |
    | J5_facts | 0.38 |
    | J4r_nofacts | 0.34 |
    | J2_facts | 0.34 |
    | J1_facts | 0.25 |
    | J1_nofacts | 0.09 |

  - **Caveat: pooled ρ mixes between-task calibration with within-task ranking.** J4r_nofacts has pooled 0.34 but negative or near-zero ρ within every task.
  - By mean within-task ρ, J2_facts is the most consistent: 0.40 / 0.46 / 0.14, positive on all three. J5_facts_think is 0.41 / 0.49 / −0.02. J5_facts is 0.53 / 0.35 / −0.25.
  - Anchor facts help consistently: J1 goes 0.09 → 0.25 and J2 0.14 → 0.34.
  - t15 is hard for every variant.
- Judge selection switched to mean within-task ρ (plan §9.6). On 60 ideas J2_facts wins (0.33); LOTO picks J2_facts in 2/3 folds.
- CPU partition ~75% offline: moved 99 pending rj CPU steps to compsci-gpu as CPU-only jobs (user-approved); realjudge.cpu_partition() defaults to compsci-gpu, override with AR_CPU_PARTITION.
- Follow-up job queued afterany all rj jobs: ingest t04/t11, then sbatch judge_study (5-task pool; cached calls reused).

## 2026-11-09 19:05 — t04 evals done, t11 waiting on GPU queue
- t04 real evals: 20/20 ok, valid, adherence 1. i000033 failed its first CPU smoke and was recovered by the repair path.
  - Returns spread widely: −99 … 324 (median ~67), not all-or-nothing.
  - Pipeline J1 ρ per run: −0.28 / −0.04.
- t11: 1 of 20 GPU runs running, 19 pending on Priority. The GPU queue is congested (185 pending, every A5000 node in use) and our fair-share is low (0.003).
- The ingest + 5-task judge-study follow-up job is queued (afterany, all rj jobs).
- Risk: 4 nodes × 24 h for the test runs may queue for a long time at this priority.
- Calibration decision: identity (no calibration); rationale in plan §9.6. make_test_configs.py defaults to a=0, b=1.
- 2026-11-10 18:35: 5-task judge study (100 ideas; job timed out at 3 h during thinking, no-think complete). FROZEN J2_facts (5/5 LOTO folds). Test configs generated.

## 2026-11-11 19:20 — test runs finished 24 h; big scoring backlog → drain phase
- All 16 test runs ran 24.5 h with no tracebacks. But only ~15–30% of ideas were scored: 110–364 per run of 630–1,424 generated.
- Ideation (40–60/h per run) outran tagging + J2 judging (5 calls per idea + confirmation).
- The optimized sampler learned from fewer F observations than intended. Report this as a limitation.
- t18 ideation truncated 18–23% (others ≈ 1%).
- Drain: resumed all 16 runs (`--resume --drain-minutes 840`, -t 15 h). max_hours is already exceeded, so ideation is off; tagger + judge finish every idea. max_inflight 8 → 14 for the drain only (scoring speed; no new ideas; same for both arms).

## 2026-11-11 21:00 — ROOT CAUSE: tagging broke in every test run
- After cycle 8–26 (of 35–101), every tagging cycle failed: the tag pool reached 194–331 tags, and creation prompts overflowed the 32k context. A few cycles also had malformed writer JSON.
- Failed cycles released their ideas, so no new idea was tagged or judged, and the sampler got no F for the last ~19–20 h of each 24 h run. Train runs (5–10 h, ~80 tags) never hit it.
- Drain jobs cancelled: the drain would hit the same overflow.
- Fixes (246 tests pass):
  - tagger shows the pool compactly (descriptions cut to 15 words);
  - merger/writer prompts are capped at 40 summaries of ≤ 600 chars, with a 24k-token budget check that drops summaries to fit;
  - a failed creation step now assigns against the existing pool instead of stranding the batch;
  - new `ideation.max_backlog` backpressure: ideation pauses while ≥ N ideas are unscored. Default off.
- Rerun decided (user: 24 h). Old test runs archived to runs/_aborted_test1/. Tag-fix smoke submitted (stuck t09 copy + fresh backpressure run).
- Tag-fix smoke passed. On the stuck t09 copy (331 tags), the first post-fix cycle succeeded with no errors and tagged 95 ideas. The fresh backpressure run had 35 ideas, 16 scored, after ~30 min.
- Expected throughput: ~10 scored ideas/h per run with J2 (~13k completion tokens/idea), so ~200–250 ideas per 24 h run, almost all scored.
- Launched test v2: 4 jobs × 4 runs, 24 h, configs/test (max_backlog 60, ideation max_tokens 8000, max_inflight 12).

## 2026-11-12 23:40 — test v2 status
- t09 and t18 (started Wed 21:31) finished 24 h at ~22:05: 712–794 / 352–365 ideas per run, 82–99% scored, 0 failed tagging cycles.
- t13 and t16 started Thu 08:15 (queue), 15 h in, healthy: KV 43–54%, no preemption.
- Tag pools are large (up to 975 tags, t09).
- Queued 32 test real evals: top 3 + 1 random per run (seed 21) for the 8 t09/t18 runs.
- 2026-11-13: all 64 test real evals done (61 valid; 3 t13 invalid: clean < 85%). judge_study target u capped at 0 for invalid/adherence-0 runs. Test-pool judge study job 13767965 (no thinking).

## 2026-11-13 17:00 — final analysis v1 (analysis/final_analysis.py → $AR_ROOT/analysis/final/)
- H1 surrogate, equal ideas, seeds pooled. Optimized > naive in 2/4 tasks; none significant after Holm:

  | task | top-10% mean, opt vs naive | p | p_holm |
  | --- | --- | --- | --- |
  | t13 | 0.807 vs 0.798 | 0.055 | 0.17 |
  | t16 | 0.522 vs 0.511 | 0.018 | 0.07 |
  | t18 | 0.755 vs 0.788 | 1 | — |
  | t09 | 0.912 vs 0.922 | 0.99 | — |

- H1-real (raw metric of top picks; invalid = worst): optimized median better in 0/4 tasks; p = 0.65, 0.85, 0.59 and t18.
- H3: linear < MLP MSE on all 4 test tasks (p < 1e-4, optimistic paired t).
- FI: no posterior CI excludes 0 in any optimized test run.
- The H2 test-pool judge study (job 13767965) is running.

## 2026-11-13 22:45 — H2 out-of-sample (test pool, 64 ideas; $AR_ROOT/analysis/judge_study_test)
- The frozen J2_facts drops from 0.21 (train) to **0.10** on the test pool (t09 −0.16, t13 0.11, t16 0.22, t18 0.23). The pipeline's live scores are pooled ρ −0.16.
- The best test-pool variants are J2_nofacts and J5_facts (0.20); nothing beats J1 significantly (all p_holm 1.0).
- Strategy ranking train vs test: Spearman −0.31 over 12 variants. **Strategy selection did not transfer.**
- Test-pool LOTO: held-out ρ −0.04 … 0.08.
- J2_facts truncated/failed to parse 20% at 4096 tokens on the test pool (vs 3% on train).
- final_analysis.py re-run; the H2 column is filled.

## 2026-11-14 — why the judge choice "didn't transfer" (sensitivity analysis; post hoc)
- **t09 is an outlier.** 94% of test ideas are below the floor (median u −6.2, sd 12.8). The frozen J2_facts gets −0.16 there.
  - Excluding t09, J2_facts test = 0.19, against train 0.21: the frozen judge's level transfers on the other 3 tasks.
  - Train/test rank correlation of strategies goes −0.31 → −0.06 without t09.
- **Selection artifact.** The test pool is top-3 by the frozen judge's own live scores + 1 random per run.
  - Simulating that rule on the train pool drops J2_facts 0.23 → 0.02 and J1_facts 0.29 → 0.19, while most other variants barely move (rank corr full vs selected 0.82).
  - So the test-pool comparison is biased against the selecting judge.
- **Noise.** The SE of a 4-task mean within-task ρ at n=16 is ≈ 0.14. Variant differences (≤ 0.2) are within noise on both pools, so strategy rankings are not reliable at this sample size.
- Report: the t09 exclusion as a sensitivity analysis (not primary), and the selection artifact as a design limitation (the random picks are too few to estimate ρ alone).
