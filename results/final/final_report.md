# Final analysis

## H1 surrogate (equal ideas; top 10% per run, both seeds pooled)

| task | N/run | top10 mean opt | naive | U | p | p_holm | equal-compute p |
|---|---|---|---|---|---|---|---|
| t18 | 288 | 0.755 | 0.788 | 1039 | 1 | 1 | 1 |
| t09 | 681 | 0.912 | 0.922 | 7754 | 0.989 | 1 | 0.987 |
| t13 | 704 | 0.807 | 0.798 | 10884 | 0.0549 | 0.165 | 0.0327 |
| t16 | 788 | 0.522 | 0.511 | 14188 | 0.0179 | 0.0714 | 0.0213 |

Direction (optimized > naive, equal ideas): 2/4 tasks

## H1-real (raw metric of each arm's top picks; invalid = worst)

| task | metric | opt top-6 | naive top-6 | U | p | p_holm |
|---|---|---|---|---|---|---|
| t18 | expected_top1_at_interruption (↑; floor 0.443, ref 0.5112) | 0.543, 0.498, 0.492, 0.459, 0.455, 0.413 | 0.582, 0.554, 0.535, 0.492, 0.461, 0.429 | 12 | 0.845 | 1 |
| t09 | heldout_ppl_per_token (↓; floor 52.52, ref 44.74) | 75.7, 91.4, 105, 118, 169, 252 | 51.8, 85.3, 93.1, 96.3, 178, 257 | 16 | 0.65 | 1 |
| t13 | pgd20_robust_acc (↑; floor 0.000333, ref 0.5025) | 0.461, 0.342, 0.199, 0.0145, invalid, invalid | 0.531, 0.413, 0.366, 0.302, 0.014, 0.0005 | 12 | 0.851 | 1 |
| t16 | mean_top1_3arch (↑; floor 0.1695, ref 0.1919) | 0.2, 0.178, 0.175, 0.175, 0.156, 0.0689 | 0.189, 0.185, 0.182, 0.17, 0.131, 0.115 | 17 | 0.591 | 1 |

Direction (optimized median > naive median): 0/4 tasks

## H2 judge strategies: mean within-task ρ

| variant | train pool | test pool |
|---|---|---|
| J2_nofacts | 0.05 | 0.20 |
| J5_facts | 0.02 | 0.20 |
| J3_facts | -0.02 | 0.18 |
| J5_nofacts | -0.09 | 0.13 |
| J4s_facts | 0.11 | 0.11 |
| J2_facts | 0.21 | 0.10 |
| J1_nofacts | 0.04 | 0.09 |
| J1_facts | 0.14 | 0.09 |
| J4s_nofacts | 0.05 | 0.06 |
| J3_nofacts | 0.07 | 0.05 |
| J4r_nofacts | 0.02 | 0.05 |
| J4r_facts | 0.06 | -0.01 |
| J4s_facts_think | 0.10 | — |
| J1_facts_think | 0.24 | — |
| J2_facts_think | 0.30 | — |

## H3 (linear vs MLP, optimized runs pooled per task)

| task | n | MSE linear | MSE MLP | t | p |
|---|---|---|---|---|---|
| t18 | 600.0 | 0.0378 | 0.0392 | -5.43 | 4.07e-07 |
| t09 | 1393.0 | 0.0385 | 0.0391 | -4.40 | 2.72e-05 |
| t13 | 1508.0 | 0.0353 | 0.0361 | -5.49 | 3.06e-07 |
| t16 | 1678.0 | 0.0121 | 0.0126 | -7.52 | 2.51e-11 |

## Feature importance: posterior features whose 95% CI excludes 0 (optimized runs)

- t18 seed 1: none
- t18 seed 2: none
- t09 seed 1: none
- t09 seed 2: none
- t13 seed 1: none
- t13 seed 2: none
- t16 seed 1: none
- t16 seed 2: none
