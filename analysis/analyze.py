"""Analysis for the report (plan §5). Reads run journals only; writes tables + figures.

usage:
  python analysis/analyze.py --pair <optimized_run> <naive_run> [--pair ...] [--out DIR]

For each (optimized, naive) pair on the same task:
  H1  Mann-Whitney U (one-sided, optimized > naive) on calibrated surrogate scores of each arm's
      top 10% ideas. PRIMARY: equal number of ideas (each arm truncated to its first N scored ideas by
      creation time, N = min over the two arms). SECONDARY: equal compute (all ideas within the same
      wall-clock budget). Best-so-far curves vs idea index (primary) and vs wall-clock (secondary);
      across pairs, a direction count for each.
  H2  Spearman rho (+ 95% bootstrap CI) between s_cal and real_score over real-judged ideas (if any).
  H3  Predicting an idea's s_cal from its 12 prompt features (optimized arm): linear regression vs a
      small MLP, repeated 10-fold CV, paired t-test on per-fold MSE.
  FI  Final sampler posterior mean +/- 95% credible interval per subprompt (feature weighting).
  Div Distinct tags covered by each arm's top-20 ideas (descriptive).
Holm correction is applied across the per-task H1 p-values.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ar import sampler  # noqa: E402

RUNS = Path("/usr/xtmp/kg349/ar/runs")


def load(run: str) -> dict:
    p = Path(run)
    p = p if p.is_dir() else RUNS / run
    return json.loads((p / "journal.json").read_text())


def scored(state: dict) -> list[dict]:
    return [i for i in state["ideas"].values() if i.get("status") != "failed" and i.get("s_cal") is not None]


def top_frac(vals: np.ndarray, frac=0.10) -> np.ndarray:
    k = max(1, int(round(len(vals) * frac)))
    return np.sort(vals)[-k:]


def first_n(state: dict, n: int) -> np.ndarray:
    """s_cal of the first n scored ideas in creation order (equal-ideas comparison)."""
    ideas = sorted(scored(state), key=lambda i: i["created_at"])[:n]
    return np.array([i["s_cal"] for i in ideas])


def h1_test(vo: np.ndarray, vn: np.ndarray) -> dict:
    to, tn = top_frac(vo), top_frac(vn)
    u, p = stats.mannwhitneyu(to, tn, alternative="greater") if len(to) and len(tn) else (np.nan, np.nan)
    return {"n_opt": len(vo), "n_naive": len(vn), "top10_mean_opt": float(to.mean()) if len(to) else None,
            "top10_mean_naive": float(tn.mean()) if len(tn) else None, "U": float(u), "p": float(p),
            "best_opt": float(vo.max()) if len(vo) else None, "best_naive": float(vn.max()) if len(vn) else None}


def best_so_far(state: dict) -> tuple[np.ndarray, np.ndarray]:
    ideas = sorted(scored(state), key=lambda i: i["created_at"])
    if not ideas:
        return np.array([]), np.array([])
    t0 = min(i["created_at"] for i in state["ideas"].values())
    t = np.array([(i["created_at"] - t0) / 3600 for i in ideas])
    return t, np.maximum.accumulate([i["s_cal"] for i in ideas])


def holm(ps: list[float]) -> list[float]:
    order = np.argsort(ps)
    m, out, run = len(ps), [0.0] * len(ps), 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (m - r) * ps[i]))
        out[i] = run
    return out


def boot_spearman(x, y, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    x, y = np.asarray(x), np.asarray(y)
    rs = []
    for _ in range(n):
        b = rng.integers(0, len(x), len(x))
        if len(set(x[b])) > 1 and len(set(y[b])) > 1:
            rs.append(stats.spearmanr(x[b], y[b])[0])
    return np.percentile(rs, [2.5, 97.5]) if rs else (np.nan, np.nan)


def h3_cv(state: dict, reps=10, folds=10, seed=0) -> dict:
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import KFold
    from sklearn.neural_network import MLPRegressor
    ideas = [i for i in scored(state) if (i.get("prompt") or {}).get("config")]
    if len(ideas) < 50:
        return {"n": len(ideas), "skipped": "fewer than 50 scored ideas"}
    X = np.array([sampler.featurize(i["prompt"]["config"]) for i in ideas])[:, 1:]  # drop intercept
    y = np.array([i["s_cal"] for i in ideas])
    lin, mlp = [], []
    for r in range(reps):
        for tr, te in KFold(folds, shuffle=True, random_state=seed + r).split(X):
            a = LinearRegression().fit(X[tr], y[tr])
            b = MLPRegressor(hidden_layer_sizes=(32, 32), alpha=1e-3, max_iter=2000,
                             random_state=seed + r).fit(X[tr], y[tr])
            lin.append(np.mean((a.predict(X[te]) - y[te]) ** 2))
            mlp.append(np.mean((b.predict(X[te]) - y[te]) ** 2))
    lin, mlp = np.array(lin), np.array(mlp)
    t, p = stats.ttest_rel(lin, mlp)
    return {"n": len(ideas), "mse_linear": lin.mean(), "mse_mlp": mlp.mean(), "t": t, "p": p,
            "note": "plain paired t over reps*folds; folds overlap so p is optimistic (report as such)"}


def posterior_table(state: dict) -> list[dict]:
    refits = state["logs"].get("refits") or []
    if not refits:
        return []
    last = refits[-1]
    return [{"feature": f, "mu": m, "sd": s, "lo": m - 1.96 * s, "hi": m + 1.96 * s}
            for f, m, s in zip(sampler.FEATURES, last["mu"], last["sd"])] + [{"n_obs": last["n"]}]


def tag_coverage(state: dict, k=20) -> int:
    top = sorted(scored(state), key=lambda i: -i["s_cal"])[:k]
    return len({t for i in top for t in i.get("tags") or []})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", nargs=2, action="append", required=True, metavar=("OPT", "NAIVE"))
    ap.add_argument("--out", default="analysis/out")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    report, p_eq, p_cmp = [], [], []
    fig, axes = plt.subplots(2, len(a.pair), figsize=(4.5 * len(a.pair), 6.4), squeeze=False)
    for k, (ro, rn) in enumerate(a.pair):
        so, sn = load(ro), load(rn)
        vo = np.array([i["s_cal"] for i in scored(so)])
        vn = np.array([i["s_cal"] for i in scored(sn)])
        n_eq = min(len(vo), len(vn))
        eq = h1_test(first_n(so, n_eq), first_n(sn, n_eq))          # PRIMARY: equal number of ideas
        cmp_ = h1_test(vo, vn)                                       # SECONDARY: equal compute (same wall clock)
        p_eq.append(eq["p"]); p_cmp.append(cmp_["p"])
        real = [(i["s_cal"], i["real_score"]) for s in (so, sn) for i in scored(s) if i.get("real_score") is not None]
        rho = stats.spearmanr(*zip(*real))[0] if len(real) >= 4 else None
        hours = {lab: (max(i["created_at"] for i in st["ideas"].values()) - min(i["created_at"] for i in st["ideas"].values())) / 3600
                 for st, lab in ((so, "opt"), (sn, "naive"))}
        row = {"task": so["config"]["task_id"], "opt_run": ro, "naive_run": rn,
               "H1_equal_ideas": eq, "H1_equal_compute": cmp_,
               "ideas_per_hour_opt": len(vo) / max(hours["opt"], 1e-9), "ideas_per_hour_naive": len(vn) / max(hours["naive"], 1e-9),
               "mean_opt": float(vo.mean()) if len(vo) else None, "mean_naive": float(vn.mean()) if len(vn) else None,
               "tags_top20_opt": tag_coverage(so), "tags_top20_naive": tag_coverage(sn),
               "H2_n_real": len(real), "H2_rho": rho,
               "H2_ci": list(boot_spearman(*zip(*real))) if len(real) >= 4 else None,
               "H3": h3_cv(so), "posterior": posterior_table(so)}
        report.append(row)
        for st, lab in ((so, "optimized"), (sn, "naive")):
            ideas = sorted(scored(st), key=lambda i: i["created_at"])
            b = np.maximum.accumulate([i["s_cal"] for i in ideas]) if ideas else np.array([])
            axes[0][k].step(np.arange(1, len(b) + 1), b, where="post", label=lab)
            t, b2 = best_so_far(st)
            axes[1][k].step(t, b2, where="post", label=lab)
        axes[0][k].axvline(n_eq, color="gray", lw=0.8, ls="--")
        axes[0][k].set_title(f"{row['task']} (equal ideas, N={n_eq})", fontsize=9)
        axes[0][k].set_xlabel("ideas generated")
        axes[1][k].set_title(f"{row['task']} (equal compute)", fontsize=9)
        axes[1][k].set_xlabel("hours")
        for ax in (axes[0][k], axes[1][k]):
            ax.set_ylabel("best calibrated surrogate score")
            ax.legend(fontsize=8)
    for row, pe, pc in zip(report, holm(p_eq), holm(p_cmp)):
        row["H1_equal_ideas"]["p_holm"] = pe
        row["H1_equal_compute"]["p_holm"] = pc
    fig.tight_layout()
    fig.savefig(out / "best_so_far.png", dpi=150)
    wins = lambda key: sum(1 for r in report if (r[key]["top10_mean_opt"] or 0) > (r[key]["top10_mean_naive"] or 0))
    summary = {"pairs": report,
               "H1_direction_equal_ideas (primary)": f"optimized > naive (top-10% mean) in {wins('H1_equal_ideas')}/{len(report)} tasks",
               "H1_direction_equal_compute (secondary)": f"optimized > naive (top-10% mean) in {wins('H1_equal_compute')}/{len(report)} tasks"}
    (out / "report.json").write_text(json.dumps(summary, indent=1, default=float))
    for r in report:
        for key, lab in (("H1_equal_ideas", "equal ideas  "), ("H1_equal_compute", "equal compute")):
            h = r[key]
            print(f"{r['task']} [{lab}] n={h['n_opt']}/{h['n_naive']}  top10 {h['top10_mean_opt']:.3f} vs "
                  f"{h['top10_mean_naive']:.3f}  U={h['U']:.0f} p={h['p']:.3g} (holm {h['p_holm']:.3g})")
        print(f"   ideas/h {r['ideas_per_hour_opt']:.0f}/{r['ideas_per_hour_naive']:.0f}  tags@20 {r['tags_top20_opt']}/{r['tags_top20_naive']}"
              f"  H3={ {k: v for k, v in r['H3'].items() if k != 'note'} }")
    for k in summary:
        if k.startswith("H1_direction"):
            print(k, ":", summary[k])
    print("->", out)


if __name__ == "__main__":
    main()
