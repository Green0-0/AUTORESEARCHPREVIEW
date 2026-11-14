# Judge study (test, 64 ideas, 2026-11-13 22:45)

Target: unclipped u = (value − floor)/(ref − floor). Baseline J1_nofacts (ρ = mean over its 5 single draws). Bootstrap B = 2000 over ideas; p two-sided, Holm across variants.

**Selection criterion: mean within-task ρ** (pre-registered 2026-11-09). Pooled ρ also rewards between-task calibration, which the pipeline never uses; it is shown for reference. The within-task bootstrap resamples ideas within each task.

| variant | n | **mean within-task ρ** | Δ within vs J1 (95% CI) | p_holm (within) | ρ t09 | ρ t13 | ρ t16 | ρ t18 | pooled ρ (95% CI) | Δ pooled vs J1 | fit a, b | tok/idea | trunc | parse fail |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| J2_nofacts | 64 | **0.20** | 0.13 [-0.04, 0.24] | 1.000 | -0.03 | 0.01 | 0.30 | 0.53 | -0.09 [-0.35, 0.16] | -0.04 (p_holm 1.000) | 0.17, 0.24 | 30265 | 0.08 | 0.08 |
| J5_facts | 64 | **0.20** | 0.09 [-0.24, 0.39] | 1.000 | 0.03 | 0.17 | 0.53 | 0.08 | -0.16 [-0.39, 0.11] | -0.12 (p_holm 1.000) | 0.33, -0.06 | 7302 | 0.00 | 0.02 |
| J3_facts | 59 | **0.18** | 0.08 [-0.18, 0.30] | 1.000 | 0.01 | 0.64 | -0.04 | 0.13 | -0.24 [-0.49, 0.01] | -0.14 (p_holm 1.000) | 0.41, -0.17 | 8945 | 0.34 | 0.34 |
| J5_nofacts | 64 | **0.13** | 0.01 [-0.26, 0.27] | 1.000 | 0.26 | 0.02 | 0.13 | 0.11 | -0.06 [-0.32, 0.20] | -0.03 (p_holm 1.000) | 0.36, -0.10 | 6894 | 0.00 | 0.04 |
| J4s_facts | 64 | **0.11** | -0.00 [-0.28, 0.27] | 1.000 | -0.17 | 0.54 | -0.01 | 0.07 | -0.22 [-0.46, 0.05] | -0.17 (p_holm 1.000) | 0.30, -0.01 | 6323 | 0.00 | 0.00 |
| J2_facts | 64 | **0.10** | 0.01 [-0.26, 0.26] | 1.000 | -0.16 | 0.11 | 0.22 | 0.23 | -0.10 [-0.36, 0.16] | -0.04 (p_holm 1.000) | 0.25, 0.05 | 35747 | 0.20 | 0.20 |
| J1_nofacts | 63 | **0.09** | — | — | -0.01 | -0.09 | 0.17 | 0.31 | -0.06 [-0.24, 0.11] | — | 0.15, 0.29 | 6222 | 0.11 | 0.11 |
| J1_facts | 61 | **0.09** | -0.01 [-0.22, 0.20] | 1.000 | -0.02 | 0.09 | 0.11 | 0.20 | -0.13 [-0.36, 0.10] | -0.07 (p_holm 1.000) | 0.33, -0.06 | 7086 | 0.20 | 0.20 |
| J4s_nofacts | 64 | **0.06** | -0.01 [-0.29, 0.26] | 1.000 | 0.00 | 0.20 | -0.03 | 0.05 | -0.20 [-0.45, 0.06] | -0.12 (p_holm 1.000) | 0.40, -0.18 | 5894 | 0.02 | 0.02 |
| J3_nofacts | 64 | **0.05** | -0.04 [-0.32, 0.24] | 1.000 | 0.24 | -0.03 | 0.11 | -0.11 | -0.01 [-0.27, 0.25] | 0.07 (p_holm 1.000) | 0.29, -0.00 | 6410 | 0.07 | 0.07 |
| J4r_nofacts | 64 | **0.05** | -0.02 [-0.36, 0.33] | 1.000 | -0.24 | -0.28 | 0.36 | 0.34 | 0.61 [0.40, 0.76] | 0.69 (p_holm 0.000) | 0.10, 0.46 | 5639 | 0.00 | 0.00 |
| J4r_facts | 64 | **-0.01** | -0.11 [-0.39, 0.20] | 1.000 | -0.39 | 0.09 | 0.13 | 0.14 | 0.27 [0.00, 0.52] | 0.33 (p_holm 0.230) | 0.11, 0.33 | 6924 | 0.07 | 0.07 |

Original pipeline s_cal: ρ = -0.16 (n = 64).
Thinking variants: J1_nofacts_think, J2_nofacts_think, J5_facts_think.

## LOTO

Leave-one-task-out: for each held-out task, the variant with the best mean within-task ρ on the other tasks (no-thinking variants only), and its ρ on the held-out task.

| held-out task | chosen on other tasks | mean within ρ, other tasks | ρ held-out | n | = overall choice |
|---|---|---|---|---|---|
| t09_tokenizer_transfer | J2_nofacts | 0.28 | -0.03 | 16 | yes |
| t13_robust_noadv | J2_nofacts | 0.27 | 0.01 | 16 | yes |
| t16_synthetic_data | J3_facts | 0.26 | -0.04 | 16 | no |
| t18_anytime | J5_facts | 0.24 | 0.08 | 16 | no |

Overall choice (all tasks): J2_nofacts (ρ = 0.20). Held-out ρ: mean 0.01, min -0.04; chosen = overall in 50% of folds.

Per-task ρ by variant (a single task driving the winner shows up here):

| variant | t09 | t13 | t16 | t18 |
|---|---|---|---|---|
| J2_nofacts | -0.03 | 0.01 | 0.30 | 0.53 |
| J5_facts | 0.03 | 0.17 | 0.53 | 0.08 |
| J3_facts | 0.01 | 0.64 | -0.04 | 0.13 |
| J5_nofacts | 0.26 | 0.02 | 0.13 | 0.11 |
| J4s_facts | -0.17 | 0.54 | -0.01 | 0.07 |
| J2_facts | -0.16 | 0.11 | 0.22 | 0.23 |
| J1_nofacts | -0.01 | -0.09 | 0.17 | 0.31 |
| J1_facts | -0.02 | 0.09 | 0.11 | 0.20 |
| J4s_nofacts | 0.00 | 0.20 | -0.03 | 0.05 |
| J3_nofacts | 0.24 | -0.03 | 0.11 | -0.11 |
| J4r_nofacts | -0.24 | -0.28 | 0.36 | 0.34 |
| J4r_facts | -0.39 | 0.09 | 0.13 | 0.14 |
