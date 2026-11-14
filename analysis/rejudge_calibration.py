"""Judge calibration (plan §3.3, step 3): re-judge the real-evaluated train ideas offline with judge variants
(thinking on/off x anchor facts on/off), K samples each, and compare to the real results.

Selection criterion: pooled Spearman rho between the mean judge score and the UNCLIPPED normalized real value
u = (value - floor) / (ref - floor). Clipping at 0 would turn every below-floor idea into a tie, so it is not used here.
Also reports the mean within-task rho on raw values.

Runs as its own GPU job with two dedicated vLLM replicas (each serves one no-thinking and one thinking variant):
  sbatch slurm/rejudge.sbatch
Results are appended per call to <out>/rejudge_calls.jsonl, and a rerun skips calls already done (resumable).
"""
from __future__ import annotations

import argparse, asyncio, json, sys
from pathlib import Path

import numpy as np
import yaml
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ar import llm as L, prompts, scoring  # noqa: E402

RUNS = Path("/usr/xtmp/kg349/ar/runs")
REPO = Path(__file__).resolve().parents[1]
TRAIN = ["t07_activation_tuning", "t10_label_noise", "t15_lwf"]
SHORT = {"t07_activation_tuning": "t07", "t10_label_noise": "t10", "t15_lwf": "t15"}
ANCHOR_SHORT = {   # one-line names of the reference-anchor methods (scripts/make_anchor_ref.py)
    "t07_activation_tuning": "per-channel BatchNorm + PReLU activations with AdamW one-cycle, label smoothing and mixup",
    "t10_label_noise": "Early-Learning Regularization (ELR) with strong augmentation and mixup",
    "t15_lwf": "iCaRL-style exemplar memory within the 1 MiB budget with distillation",
}


def anchor_facts(task_id: str) -> list[str]:
    t = yaml.safe_load((REPO / "tasks" / task_id / "task.yaml").read_text())
    return [f"Under this exact budget and harness, the provided starter solution reaches {t['metric']} = {t['floor']:.4g}; "
            f"treat starter-level results as score 0.",
            f"A strong known method ({ANCHOR_SHORT[task_id]}) reaches {t['metric']} = {t['ref']:.4g} under the same "
            f"budget; treat that level as score 1.0. Scores scale linearly between these two anchors."]


def load_items() -> list[dict]:
    items = []
    for task in TRAIN:
        t = yaml.safe_load((REPO / "tasks" / task / "task.yaml").read_text())
        for arm in ("optimized", "naive"):
            st = json.loads((RUNS / f"train_{SHORT[task]}_{arm}_s2" / "journal.json").read_text())
            for i in st["ideas"].values():
                real = i.get("real") or {}
                if i.get("real_score") is None or real.get("value") is None:
                    continue
                v = float(real["value"])
                items.append({"task": task, "run": f"train_{SHORT[task]}_{arm}_s2", "id": i["id"],
                              "description": i["description"], "s_orig": i.get("s_cal"), "value": v,
                              "u": (v - t["floor"]) / (t["ref"] - t["floor"]), "real_score": i["real_score"]})
    return items


async def judge_all(items, variants, K, base_url, out: Path, inflight: int):
    llm_cfg = {"base_url": base_url, "model": "qwen", "max_inflight": inflight,
               "purposes": {"judge": {"thinking": False, "max_tokens": 4096, "temperature": 0.7}}}
    clients = {}
    for name, v in variants.items():
        cfg = json.loads(json.dumps(llm_cfg))
        cfg["purposes"]["judge"].update(thinking=v["thinking"], max_tokens=v["max_tokens"])
        if v.get("port"):   # balance: each replica gets one no-thinking and one thinking variant
            cfg["base_url"] = base_url.replace(":8000", f":{v['port']}")
        clients[name] = L.LLM(cfg, usage_path=out / f"usage_{name}.jsonl", timeout_s=1800)
    task_prompt = {t: (REPO / "tasks" / t / "TASK.md").read_text() for t in TRAIN}
    rng = np.random.default_rng(0)
    systems = {(name, t): prompts.judge_system(task_prompt[t], anchor_facts(t) if v["facts"] else [], rng)
               for name, v in variants.items() for t in TRAIN}

    calls_p = out / "rejudge_calls.jsonl"
    done = {}
    if calls_p.exists():
        for line in calls_p.read_text().splitlines():
            r = json.loads(line)
            if r[4] is not None:
                done[(r[0], r[1], r[2], r[3])] = r

    async def one(name, it, k):
        key = (name, it["id"], it["run"], k)
        if key in done:
            return done[key]
        r = await _one(name, it, k)
        with open(calls_p, "a") as f:
            f.write(json.dumps(r) + "\n")
        return r

    async def _one(name, it, k):
        msgs = [{"role": "system", "content": systems[(name, it["task"])]},
                {"role": "user", "content": prompts.judge_prompt(it["description"])}]
        for _ in range(3):
            try:
                r = await clients[name].chat(msgs, "judge")
            except Exception as e:  # noqa: BLE001
                err = repr(e)
                continue
            s = scoring.parse_boxed(r.text)
            if s is not None:
                return [name, it["id"], it["run"], k, s, r.finish_reason]
            err = f"unparseable ({r.finish_reason})"
        return [name, it["id"], it["run"], k, None, err]

    res = await asyncio.gather(*[one(n, it, k) for n in variants for it in items for k in range(K)])
    for c in clients.values():
        await c.aclose()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--inflight", type=int, default=16, help="max in-flight requests per variant")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    variants = {"nothink_nofacts": dict(thinking=False, facts=False, max_tokens=4096, port=8000),
                "think_facts":     dict(thinking=True,  facts=True,  max_tokens=12000, port=8000),
                "nothink_facts":   dict(thinking=False, facts=True,  max_tokens=4096, port=8001),
                "think_nofacts":   dict(thinking=True,  facts=False, max_tokens=12000, port=8001)}
    items = load_items()
    print(f"{len(items)} real-evaluated train ideas")
    res = asyncio.run(judge_all(items, variants, a.K, a.base_url, out, a.inflight))
    (out / "rejudge_raw.json").write_text(json.dumps(res, indent=0))
    summary = {}
    for name in variants:
        sc = {}
        for n, iid, run, k, s, fr in res:
            if n == name and s is not None:
                sc.setdefault((run, iid), []).append(s)
        fails = sum(1 for r in res if r[0] == name and r[4] is None)
        m = np.array([np.mean(sc.get((it["run"], it["id"]), [np.nan])) for it in items])
        u = np.array([it["u"] for it in items]); ok = ~np.isnan(m)
        rho = stats.spearmanr(m[ok], u[ok])
        within = [stats.spearmanr(m[[i for i, it in enumerate(items) if it["task"] == t and ok[i]]],
                                  [it["value"] for i, it in enumerate(items) if it["task"] == t and ok[i]])[0] for t in TRAIN]
        b, a0 = np.polyfit(m[ok], np.clip(u[ok], 0, 1), 1)   # s_cal = clip(a + b*s, 0, 1) target: real_score scale
        summary[name] = {"pooled_rho_u": float(rho[0]), "p": float(rho[1]), "within_task_rho": [float(x) for x in within],
                         "mean_within_rho": float(np.nanmean(within)), "n_ok": int(ok.sum()), "failed_calls": fails,
                         "linear_fit": {"a": float(a0), "b": float(b)}, "mean_score": float(np.nanmean(m))}
        print(name, json.dumps(summary[name]))
    s0 = np.array([it["s_orig"] for it in items]); u = np.array([it["u"] for it in items])
    summary["original_pipeline_scores"] = {"pooled_rho_u": float(stats.spearmanr(s0, u)[0])}
    print("original", summary["original_pipeline_scores"])
    (out / "summary.json").write_text(json.dumps({"items": items, "variants": variants, "summary": summary}, indent=1))


if __name__ == "__main__":
    main()
