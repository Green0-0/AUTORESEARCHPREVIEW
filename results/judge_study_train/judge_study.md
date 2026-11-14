# Judge study (train, 100 ideas, 2026-11-10 18:33)

Target: unclipped u = (value − floor)/(ref − floor). Baseline J1_nofacts (ρ = mean over its 5 single draws). Bootstrap B = 2000 over ideas; p two-sided, Holm across variants.

**Selection criterion: mean within-task ρ** (pre-registered 2026-11-09). Pooled ρ also rewards between-task calibration, which the pipeline never uses; it is shown for reference. The within-task bootstrap resamples ideas within each task.

| variant | n | **mean within-task ρ** | Δ within vs J1 (95% CI) | p_holm (within) | ρ t04 | ρ t07 | ρ t10 | ρ t11 | ρ t15 | pooled ρ (95% CI) | Δ pooled vs J1 | fit a, b | tok/idea | trunc | parse fail |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| J2_facts_think | 62 | **0.30** | 0.20 [-0.02, 0.38] | 0.936 | 0.38 | 0.31 | 0.68 | 0.24 | -0.13 | 0.17 [-0.09, 0.38] | 0.22 (p_holm 0.153) | 0.10, 0.61 | 49275 | 0.17 | 0.05 |
| J1_facts_think | 61 | **0.24** | 0.14 [-0.04, 0.29] | 1.000 | 0.35 | 0.30 | 0.63 | 0.13 | -0.21 | 0.13 [-0.08, 0.32] | 0.18 (p_holm 0.352) | 0.15, 0.41 | 10453 | 0.14 | 0.08 |
| J2_facts | 100 | **0.21** | 0.17 [0.02, 0.32] | 0.476 | 0.15 | 0.40 | 0.46 | -0.10 | 0.14 | 0.18 [-0.01, 0.35] | 0.24 (p_holm 0.011) | 0.20, 0.29 | 27040 | 0.02 | 0.02 |
| J1_facts | 100 | **0.14** | 0.11 [-0.02, 0.22] | 1.000 | 0.09 | 0.30 | 0.40 | -0.08 | 0.02 | 0.14 [-0.00, 0.28] | 0.21 (p_holm 0.000) | 0.22, 0.26 | 5534 | 0.03 | 0.03 |
| J4s_facts | 100 | **0.11** | 0.08 [-0.12, 0.26] | 1.000 | 0.09 | 0.19 | 0.27 | 0.25 | -0.22 | 0.02 [-0.17, 0.20] | 0.08 (p_holm 1.000) | 0.26, 0.12 | 5894 | 0.01 | 0.01 |
| J4s_facts_think | 97 | **0.10** | 0.08 [-0.14, 0.28] | 1.000 | -0.11 | 0.06 | 0.28 | 0.36 | -0.07 | -0.04 [-0.23, 0.16] | 0.05 (p_holm 1.000) | 0.31, -0.00 | 11442 | 0.01 | 0.00 |
| J3_nofacts | 100 | **0.07** | 0.03 [-0.16, 0.23] | 1.000 | 0.14 | 0.43 | 0.12 | -0.04 | -0.29 | -0.05 [-0.26, 0.15] | 0.01 (p_holm 1.000) | 0.31, -0.00 | 5588 | 0.03 | 0.03 |
| J4r_facts | 100 | **0.06** | 0.02 [-0.16, 0.20] | 1.000 | 0.49 | -0.13 | 0.16 | 0.04 | -0.25 | 0.42 [0.25, 0.57] | 0.48 (p_holm 0.000) | 0.14, 0.56 | 5964 | 0.00 | 0.00 |
| J2_nofacts | 100 | **0.05** | 0.02 [-0.08, 0.10] | 1.000 | -0.01 | 0.18 | 0.26 | -0.15 | -0.00 | -0.07 [-0.25, 0.12] | -0.01 (p_holm 1.000) | 0.35, -0.08 | 24772 | 0.00 | 0.00 |
| J4s_nofacts | 100 | **0.05** | 0.01 [-0.18, 0.20] | 1.000 | 0.08 | 0.08 | 0.33 | 0.16 | -0.42 | -0.05 [-0.24, 0.14] | 0.02 (p_holm 1.000) | 0.30, 0.03 | 5373 | 0.00 | 0.00 |
| J1_nofacts | 100 | **0.04** | — | — | -0.00 | 0.16 | 0.22 | -0.14 | -0.04 | -0.06 [-0.21, 0.09] | — | 0.34, -0.07 | 4936 | 0.00 | 0.00 |
| J4r_nofacts | 100 | **0.02** | -0.02 [-0.21, 0.16] | 1.000 | 0.12 | -0.05 | 0.19 | -0.02 | -0.15 | 0.45 [0.27, 0.61] | 0.51 (p_holm 0.000) | 0.12, 0.60 | 5412 | 0.00 | 0.00 |
| J5_facts | 100 | **0.02** | -0.02 [-0.22, 0.18] | 1.000 | -0.31 | 0.53 | 0.35 | -0.24 | -0.25 | 0.21 [0.03, 0.38] | 0.27 (p_holm 0.050) | 0.23, 0.27 | 6673 | 0.00 | 0.05 |
| J3_facts | 99 | **-0.02** | -0.05 [-0.24, 0.14] | 1.000 | -0.22 | 0.49 | 0.22 | 0.04 | -0.61 | -0.05 [-0.22, 0.15] | 0.02 (p_holm 1.000) | 0.35, -0.09 | 7931 | 0.20 | 0.20 |
| J5_nofacts | 100 | **-0.09** | -0.13 [-0.33, 0.07] | 1.000 | 0.00 | 0.13 | 0.33 | -0.35 | -0.57 | 0.11 [-0.07, 0.29] | 0.17 (p_holm 0.546) | 0.25, 0.16 | 6996 | 0.00 | 0.11 |

Original pipeline s_cal: ρ = -0.08 (n = 100).
Thinking variants: J1_facts_think, J2_facts_think, J4s_facts_think.

## LOTO

Leave-one-task-out: for each held-out task, the variant with the best mean within-task ρ on the other tasks (no-thinking variants only), and its ρ on the held-out task.

| held-out task | chosen on other tasks | mean within ρ, other tasks | ρ held-out | n | = overall choice |
|---|---|---|---|---|---|
| t04_tiny_walker | J2_facts | 0.22 | 0.15 | 20 | yes |
| t07_activation_tuning | J2_facts | 0.16 | 0.40 | 20 | yes |
| t10_label_noise | J2_facts | 0.15 | 0.46 | 20 | yes |
| t11_fewshot_seg | J2_facts | 0.29 | -0.10 | 20 | yes |
| t15_lwf | J2_facts | 0.23 | 0.14 | 20 | yes |

Overall choice (all tasks): J2_facts (ρ = 0.21). Held-out ρ: mean 0.21, min -0.10; chosen = overall in 100% of folds.

Per-task ρ by variant (a single task driving the winner shows up here):

| variant | t04 | t07 | t10 | t11 | t15 |
|---|---|---|---|---|---|
| J2_facts | 0.15 | 0.40 | 0.46 | -0.10 | 0.14 |
| J1_facts | 0.09 | 0.30 | 0.40 | -0.08 | 0.02 |
| J4s_facts | 0.09 | 0.19 | 0.27 | 0.25 | -0.22 |
| J3_nofacts | 0.14 | 0.43 | 0.12 | -0.04 | -0.29 |
| J4r_facts | 0.49 | -0.13 | 0.16 | 0.04 | -0.25 |
| J2_nofacts | -0.01 | 0.18 | 0.26 | -0.15 | -0.00 |
| J4s_nofacts | 0.08 | 0.08 | 0.33 | 0.16 | -0.42 |
| J1_nofacts | -0.00 | 0.16 | 0.22 | -0.14 | -0.04 |
| J4r_nofacts | 0.12 | -0.05 | 0.19 | -0.02 | -0.15 |
| J5_facts | -0.31 | 0.53 | 0.35 | -0.24 | -0.25 |
| J3_facts | -0.22 | 0.49 | 0.22 | 0.04 | -0.61 |
| J5_nofacts | 0.00 | 0.13 | 0.33 | -0.35 | -0.57 |
