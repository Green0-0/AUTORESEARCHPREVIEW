# Results (copied from $AR_ROOT/analysis; the full journals and run outputs stay on /usr/xtmp)

| folder | contents |
| --- | --- |
| `final/` | `final_report.md/json` from `analysis/final_analysis.py`: H1 surrogate (equal ideas, seeds pooled), H1-real, H2 train vs test, H3, feature importance |
| `judge_study_train/` | judge-strategy study on the 100-idea, 5-task train pool (`analysis/judge_study.py`). The frozen judge J2_facts was selected here (plan §9.6–9.7) |
| `judge_study_test/` | the same study on the 64-idea test pool (out-of-sample H2) |
| `calibration/` | the first re-judge (thinking × facts, J1 only, 18 ideas) |
| `repilot_train_s3/` | train re-pilot: optimized vs corrected uniform naive (t07/t10/t15) |
| `test_runs/` | test runs v2 (4 tasks × 2 arms × 2 seeds, 24 h): per-seed H1 tables and best-so-far curves |
| `superseded_generic_naive_pilot/` | first train pilot, where the naive arm was Generic-only. It is a prompt comparison, not H1; kept for the record |

See `OVERNIGHT_LOG.md` for the full timeline, including failures and fixes:
- the tagging context overflow that invalidated test runs v1;
- the corrected naive arm;
- the pooled → within-task selection criterion;
- the H2 transfer diagnosis (t09 outlier, selection artifact).
