"""Final analysis for the report (plan §5 + §9). Reads journals and the judge-study summaries only.

  H1 (surrogate, PRIMARY equal ideas): per test task, both seeds pooled. Each run is truncated to its first N scored
      ideas (N = min over the task's 4 runs); the top 10% of each run go into a one-sided Mann-Whitney U
      (optimized > naive). Holm across the 4 tasks; direction count; per-seed table; equal-compute secondary.
  H1-real: per test task, raw metric values of each arm's real-evaluated TOP picks (top-3 per run, 2 seeds = 6 per
      arm); sign flipped for lower-is-better; INVALID runs rank as the worst. One-sided Mann-Whitney + direction.
  H2: judge strategies, mean within-task Spearman rho on the train pool (selection) and the test pool
      (out-of-sample), from analysis/judge_study.py summaries.
  H3: linear vs MLP predicting s_cal from the 12 prompt features (optimized test runs, pooled per task).
  FI: final posterior mean +/- 95% CI per subprompt feature (optimized runs).

usage: python analysis/final_analysis.py --out $AR_ROOT/analysis/final
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import analyze as A  # noqa: E402
from ar import sampler  # noqa: E402

AR = Path("/usr/xtmp/kg349/ar")
REPO = Path(__file__).resolve().parents[1]
TEST = {"t18": "t18_anytime", "t09": "t09_tokenizer_transfer", "t13": "t13_robust_noadv", "t16": "t16_synthetic_data"}
TRAIN_PAIRS = {"t07": "t07_activation_tuning", "t10": "t10_label_noise", "t15": "t15_lwf"}


def run_id(t, arm, s):
    return f"test_{t}_{arm}_s{s}"


def h1_surrogate(t: str) -> dict:
    st = {(arm, s): A.load(run_id(t, arm, s)) for arm in ("optimized", "naive") for s in (1, 2)}
    n_eq = min(len(A.scored(x)) for x in st.values())
    top_eq = {arm: np.concatenate([A.top_frac(A.first_n(st[(arm, s)], n_eq)) for s in (1, 2)]) for arm in ("optimized", "naive")}
    top_all = {arm: np.concatenate([A.top_frac(np.array([i["s_cal"] for i in A.scored(st[(arm, s)])])) for s in (1, 2)])
               for arm in ("optimized", "naive")}

    def mw(d):
        u, p = stats.mannwhitneyu(d["optimized"], d["naive"], alternative="greater")
        return {"U": float(u), "p": float(p), "mean_opt": float(d["optimized"].mean()), "mean_naive": float(d["naive"].mean()),
                "n_top_opt": int(len(d["optimized"])), "n_top_naive": int(len(d["naive"]))}
    per_seed = {}
    for s in (1, 2):
        per_seed[s] = A.h1_test(A.first_n(st[("optimized", s)], n_eq), A.first_n(st[("naive", s)], n_eq))
    return {"N_equal_ideas_per_run": n_eq, "equal_ideas": mw(top_eq), "equal_compute": mw(top_all),
            "per_seed_equal_ideas": per_seed,
            "n_scored": {f"{a}_s{s}": len(A.scored(st[(a, s)])) for (a, s) in st}}


def h1_real(t: str) -> dict:
    ty = yaml.safe_load((REPO / "tasks" / TEST[t] / "task.yaml").read_text())
    sign = 1.0 if ty.get("higher_is_better", True) else -1.0
    vals = {"optimized": [], "naive": []}
    rnd = {"optimized": [], "naive": []}
    for arm in vals:
        for s in (1, 2):
            for i in A.load(run_id(t, arm, s))["ideas"].values():
                r = i.get("real") or {}
                if r.get("value") is None:
                    continue
                v = sign * float(r["value"])
                if r.get("valid") is False or r.get("adherence") == 0:
                    v = -np.inf                   # invalid / non-adherent = worst
                (vals if "top" in str(r.get("selection", "")) else rnd)[arm].append(v)
    a, b = np.array(vals["optimized"]), np.array(vals["naive"])
    u, p = stats.mannwhitneyu(a, b, alternative="greater")
    fin = lambda x: [float(sign * v) if np.isfinite(v) else "invalid" for v in sorted(x, reverse=True)]
    return {"metric": ty["metric"], "higher_is_better": bool(sign > 0), "floor": ty["floor"], "ref": ty["ref"],
            "top_optimized": fin(a), "top_naive": fin(b), "random_optimized": fin(rnd["optimized"]), "random_naive": fin(rnd["naive"]),
            "U": float(u), "p": float(p),
            "median_opt": float(sign * np.median(a)) if np.isfinite(np.median(a)) else None,
            "median_naive": float(sign * np.median(b)) if np.isfinite(np.median(b)) else None,
            "direction_opt_better": bool(np.median(a) > np.median(b))}


def h3_fi(t: str) -> dict:
    ideas, last = [], []
    for s in (1, 2):
        st = A.load(run_id(t, "optimized", s))
        ideas += [i for i in A.scored(st) if (i.get("prompt") or {}).get("config")]
        last.append(A.posterior_table(st))
    h3 = A.h3_cv({"ideas": {str(k): i for k, i in enumerate(ideas)}})
    return {"H3": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in h3.items()},
            "posterior_seed1": last[0], "posterior_seed2": last[1]}


def judge(path: Path) -> dict:
    if not (path / "summary.json").exists():
        return {"missing": str(path)}
    S = json.loads((path / "summary.json").read_text())
    res = S.get("results", S)
    out = {k: {"mean_within_task_rho": v.get("mean_within_task_rho"), "per_task_rho": v.get("per_task_rho"),
               "pooled_rho": v.get("rho"), "delta_within_vs_J1": (v.get("delta_within_vs_baseline") or {})}
           for k, v in res.items() if not k.startswith("_") and isinstance(v, dict) and "rho" in v}
    return {"variants": out, "loto": res.get("_loto", {}), "n_items": S.get("meta", {}).get("n_items")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(AR / "analysis" / "final"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rep = {"H1_surrogate": {}, "H1_real": {}, "H3_FI": {}}
    for t in TEST:
        rep["H1_surrogate"][t] = h1_surrogate(t)
        rep["H1_real"][t] = h1_real(t)
        rep["H3_FI"][t] = h3_fi(t)
    ps = [rep["H1_surrogate"][t]["equal_ideas"]["p"] for t in TEST]
    for t, ph in zip(TEST, A.holm(ps)):
        rep["H1_surrogate"][t]["equal_ideas"]["p_holm"] = ph
    ps = [rep["H1_real"][t]["p"] for t in TEST]
    for t, ph in zip(TEST, A.holm(ps)):
        rep["H1_real"][t]["p_holm"] = ph
    rep["H1_direction_equal_ideas"] = sum(rep["H1_surrogate"][t]["equal_ideas"]["mean_opt"] > rep["H1_surrogate"][t]["equal_ideas"]["mean_naive"] for t in TEST)
    rep["H1_real_direction"] = sum(rep["H1_real"][t]["direction_opt_better"] for t in TEST)
    rep["H2_train"] = judge(AR / "analysis" / "judge_study")
    rep["H2_test"] = judge(AR / "analysis" / "judge_study_test")
    (out / "final_report.json").write_text(json.dumps(rep, indent=1, default=float))

    L = ["# Final analysis", "", "## H1 surrogate (equal ideas; top 10% per run, both seeds pooled)", "",
         "| task | N/run | top10 mean opt | naive | U | p | p_holm | equal-compute p |", "|---|---|---|---|---|---|---|---|"]
    for t in TEST:
        h = rep["H1_surrogate"][t]
        e, c = h["equal_ideas"], h["equal_compute"]
        L.append(f"| {t} | {h['N_equal_ideas_per_run']} | {e['mean_opt']:.3f} | {e['mean_naive']:.3f} | {e['U']:.0f} | {e['p']:.3g} | {e['p_holm']:.3g} | {c['p']:.3g} |")
    L += ["", f"Direction (optimized > naive, equal ideas): {rep['H1_direction_equal_ideas']}/4 tasks", "",
          "## H1-real (raw metric of each arm's top picks; invalid = worst)", "",
          "| task | metric | opt top-6 | naive top-6 | U | p | p_holm |", "|---|---|---|---|---|---|---|"]
    for t in TEST:
        h = rep["H1_real"][t]
        f = lambda xs: ", ".join(x if isinstance(x, str) else f"{x:.3g}" for x in xs)
        L.append(f"| {t} | {h['metric']} ({'↑' if h['higher_is_better'] else '↓'}; floor {h['floor']:.4g}, ref {h['ref']:.4g}) | {f(h['top_optimized'])} | {f(h['top_naive'])} | {h['U']:.0f} | {h['p']:.3g} | {h['p_holm']:.3g} |")
    L += ["", f"Direction (optimized median > naive median): {rep['H1_real_direction']}/4 tasks", "",
          "## H2 judge strategies: mean within-task ρ", "", "| variant | train pool | test pool |", "|---|---|---|"]
    tv, te = rep["H2_train"].get("variants", {}), rep["H2_test"].get("variants", {})
    for k in sorted(set(tv) | set(te), key=lambda k: -((te.get(k) or {}).get("mean_within_task_rho") or -9)):
        g = lambda d: (f"{d['mean_within_task_rho']:.2f}" if d and d.get("mean_within_task_rho") == d.get("mean_within_task_rho") and d.get("mean_within_task_rho") is not None else "—")
        L.append(f"| {k} | {g(tv.get(k))} | {g(te.get(k))} |")
    L += ["", "## H3 (linear vs MLP, optimized runs pooled per task)", "", "| task | n | MSE linear | MSE MLP | t | p |", "|---|---|---|---|---|---|"]
    for t in TEST:
        h = rep["H3_FI"][t]["H3"]
        if "mse_linear" in h:
            L.append(f"| {t} | {h['n']} | {h['mse_linear']:.4f} | {h['mse_mlp']:.4f} | {h['t']:.2f} | {h['p']:.3g} |")
    L += ["", "## Feature importance: posterior features whose 95% CI excludes 0 (optimized runs)", ""]
    for t in TEST:
        for s in (1, 2):
            post = [x for x in rep["H3_FI"][t][f"posterior_seed{s}"] if "feature" in x]
            sig = [f"{x['feature']} ({x['mu']:+.3f})" for x in post if x["lo"] > 0 or x["hi"] < 0]
            L.append(f"- {t} seed {s}: {', '.join(sig) if sig else 'none'}")
    (out / "final_report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
