"""Offline judge-strategy study (plan §9.1-9.2): re-judge every real-evaluated idea of a pool with the LLM-only
strategies J1-J5, each with and without anchor facts (thinking off), then with thinking on for the best 1-2
strategies, and compare each to the real outcome.

Strategies (prompts: ar/prompts.py; selection/batching: ar/judging.py):
  J1   Basic zero-shot, one sample. Its rho is the mean over the K single-sample rhos (the J2 draws), so the
       baseline does not rest on one lucky or unlucky draw; "J1@1" in the json is draw 1 alone.
  J2   Basic, mean of K samples (--K, default 5). Shares its calls with J1.
  J3   Arbitrary Rubric (1 call per idea).
  J4r  Basic FS with real examples: same-task pool ideas with their real_score (normalized, clipped to
       [0,1]), leave-one-out (never the idea itself or a near-duplicate of its summary). --fewshot-n (default 6)
       examples, stratified: candidates sorted by unclipped u, cut into n equal-count bins, one drawn per bin,
       shown in random order; rng seeded per idea.
  J4s  Basic FS with surrogate examples: same-run ideas created before the idea, with their pipeline s_cal
       (what a live pipeline has). Same stratified selection on s_cal.
  J5   Ranking: per task, ideas shuffled into batches of ~--rank-batch (5), each idea in --R (2) different
       batches (rep 2 minimizes repeated pairs); references = the task's starter (known 0) and reference
       method (known 1); score = mean over the batches.
  Facts: "_facts" variants put the two anchor facts (ar/anchors.anchor_facts) in {{facts}}; "_nofacts" none.
  Thinking: "_think" variants (--think-top best strategies by MEAN WITHIN-TASK rho, at their better facts setting).

Target: unclipped normalized real value u = (value - floor) / (ref - floor) from tasks/<task>/task.yaml.
Metrics per variant: pooled Spearman rho vs u + 95% bootstrap CI, per-task rho, linear fit
clip(u,0,1) ~ a + b*score, tokens per idea, truncation rate (finish_reason=length per call), parse-failure rate
(per call attempt; for J5 per idea slot) and final missing rate. Paired bootstrap over ideas of
rho(variant) - rho(J1_nofacts), two-sided p, Holm-adjusted across variants.

Calls are resumable: every attempt is appended to <out>/calls.jsonl keyed by a hash of (settings, messages,
sample, attempt), and a rerun reuses them (so new pool ideas only cost their own calls; J4r/J5 prompts change
when the pool grows, so those are re-asked). Two vLLM replicas (--ports) share one work queue with a KV-token
budget per replica, so both stay busy until the queue is empty.

usage:
  python analysis/judge_study.py --out $AR_ROOT/analysis/judge_study                 # full study (GPU job)
  python analysis/judge_study.py --out DIR --plan                                    # count calls/tokens only
  python analysis/judge_study.py --out DIR --mock --limit 2 --K 1 --R 1              # offline smoke
  python analysis/judge_study.py --out DIR --phase analyze                           # recompute from cache only
  --pool train (default) | test | <comma-separated run-id globs>
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import fnmatch
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ar import anchors, judging, prompts, scoring  # noqa: E402
from ar import llm as L  # noqa: E402

AR_ROOT = Path("/usr/xtmp/kg349/ar")
RUNS = AR_ROOT / "runs"
REPO = Path(__file__).resolve().parents[1]
POOLS = {"train": ["train_t07_*_s2", "train_t07_*_s3", "train_t10_*_s2", "train_t10_*_s3",
                   "train_t15_*_s2", "train_t15_*_s3", "train_t04_*_s1", "train_t11_*_s1"],
         "test": ["test_*"]}
MAX_MODEL_LEN = 32768
CHARS_PER_TOKEN = 3.3          # conservative (measured ~3.6 on idea descriptions)
BASELINE = "J1_nofacts"
STRATS = ["J1", "J2", "J3", "J4r", "J4s", "J5"]
CALL_KIND = {"J1": "basic", "J2": "basic", "J3": "rubric", "J4r": "fs_real", "J4s": "fs_surr", "J5": "rank"}
# rough completion-token estimates for --plan (measured nothink basic ~1.9k; thinking ~10.5k at a 12k cap)
EST_COMPLETION = {"basic": 1900, "rubric": 2500, "fs_real": 1900, "fs_surr": 1900, "rank": 3500}
EST_COMPLETION_THINK = 11000


# ============================================================================ pool
def _target_u(v: float, t: dict, real: dict) -> float:
    """Unclipped normalized value. An INVALID run (constraint violated, e.g. t13 clean acc < 85%) or adherence 0
    counts as a failure: u is capped at 0 (floor level), so breaking the rules is never rewarded (2026-11-13)."""
    u = (v - t["floor"]) / (t["ref"] - t["floor"])
    if real.get("valid") is False or real.get("adherence") == 0:
        u = min(u, 0.0)
    return u


def load_pool(spec: str, limit: int | None = None) -> tuple[list[dict], dict]:
    """-> (items, journals). items carry task, run, id, key, description, summary, value, u, real_score, s_orig,
    created_at. journals[run] = state (for J4s candidates)."""
    pats = POOLS.get(spec, [p.strip() for p in spec.split(",") if p.strip()])
    runs = sorted({d.name for d in RUNS.iterdir() if (d / "journal.json").exists()
                   and any(fnmatch.fnmatch(d.name, p) for p in pats)})
    items, journals = [], {}
    for run in runs:
        st = json.loads((RUNS / run / "journal.json").read_text())
        task = st["config"]["task_id"]
        t = anchors.task_meta(task)
        if t.get("floor") is None or t.get("ref") is None:
            print(f"skip {run}: {task} has no floor/ref", file=sys.stderr)
            continue
        journals[run] = st
        for i in st["ideas"].values():
            real = i.get("real") or {}
            if i.get("real_score") is None or real.get("value") is None:
                continue
            v = float(real["value"])
            items.append({"task": task, "run": run, "id": i["id"], "key": f"{run}/{i['id']}",
                          "description": i["description"], "summary": i.get("summary") or "",
                          "value": v, "u": _target_u(v, t, real),
                          "real_score": float(i["real_score"]), "s_orig": i.get("s_cal"),
                          "created_at": i.get("created_at")})
    if limit:
        items = items[:limit]
    return items, journals


# ============================================================================ call jobs
def _hash(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:24]


def est_tokens(messages: list[dict]) -> int:
    return int(sum(len(m["content"]) for m in messages) / CHARS_PER_TOKEN) + 64


class Job:
    """One logical judge call (with re-asks on failure). kind 'single' -> one score; 'rank' -> one per member."""

    def __init__(self, cond: str, kind: str, members: list[str], k: int, build, thinking: bool, cap: int,
                 attempt: int = 1, max_attempts: int = 3, meta: dict | None = None):
        self.cond, self.kind, self.members, self.k = cond, kind, list(members), k
        self.build, self.thinking, self.cap = build, thinking, cap
        self.attempt, self.max_attempts, self.meta = attempt, max_attempts, meta or {}
        self.messages = build(self.members, attempt)
        self.prompt_est = est_tokens(self.messages)
        self.max_tokens = max(256, min(cap, MAX_MODEL_LEN - self.prompt_est - 256))
        self.settings = {"thinking": thinking, "max_tokens": self.max_tokens, "temperature": 0.7}
        self.key = _hash([cond, self.settings, self.messages, k, attempt])
        self.need = self.prompt_est + self.max_tokens
        self.reserved = 0

    def followup(self, missing: list[str]) -> "Job | None":
        if not missing or self.attempt >= self.max_attempts:
            return None
        return Job(self.cond, self.kind, missing, self.k, self.build, self.thinking, self.cap,
                   self.attempt + 1, self.max_attempts, self.meta)


class Study:
    def __init__(self, a, items, journals):
        self.a, self.items, self.journals = a, items, journals
        self.by_key = {it["key"]: it for it in items}
        self.tasks = sorted({it["task"] for it in items})
        self.task_prompt = {t: (REPO / "tasks" / t / "TASK.md").read_text().strip() for t in self.tasks}
        self.refs = {t: anchors.ranking_references(t) for t in self.tasks}
        self.facts = {t: anchors.anchor_facts(t) for t in self.tasks}
        self.out = Path(a.out)
        self.calls_p = self.out / "calls.jsonl"
        self.cache: dict[str, dict] = {}
        if self.calls_p.exists():
            for line in self.calls_p.read_text().splitlines():
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                self.cache[r["key"]] = r
        self.records: list[dict] = []      # records of this invocation's jobs (cached or new), all attempts

    # -------------------------------------------------------------- prompts
    def system(self, task: str, facts: bool) -> str:
        return prompts.judge_system(self.task_prompt[task], self.facts[task] if facts else [], np.random.default_rng(0))

    def fs_real_examples(self, it: dict) -> list[tuple[str, float]]:
        cands = [{"id": o["key"], "summary": o["summary"], "score": o["real_score"], "sort": o["u"]}
                 for o in self.items if o["task"] == it["task"] and o["key"] != it["key"]
                 and o["description"] != it["description"]]
        rng = np.random.default_rng(judging.stable_int(self.a.seed, it["key"], "fs_real"))
        ch = judging.select_examples(cands, self.a.fewshot_n, rng, exclude_id=it["key"], exclude_texts=(it["summary"],))
        return judging.fewshot_examples(ch)

    def fs_surr_examples(self, it: dict) -> list[tuple[str, float]]:
        st = self.journals[it["run"]]
        cands = [{"id": i["id"], "summary": i.get("summary"), "score": float(i["s_cal"])}
                 for i in st["ideas"].values()
                 if i["id"] != it["id"] and i.get("status") != "failed" and i.get("s_cal") is not None
                 and i.get("created_at") is not None and it["created_at"] is not None
                 and i["created_at"] < it["created_at"]]
        rng = np.random.default_rng(judging.stable_int(self.a.seed, it["key"], "fs_surr"))
        ch = judging.select_examples(cands, self.a.fewshot_n, rng, exclude_id=it["id"], exclude_texts=(it["summary"],))
        return judging.fewshot_examples(ch)

    # -------------------------------------------------------------- jobs
    def jobs_for(self, strat: str, facts: bool, thinking: bool, K: int) -> list[Job]:
        cond = f"{strat}_{'facts' if facts else 'nofacts'}{'_think' if thinking else ''}"
        cap = self.a.think_max_tokens if thinking else (self.a.rank_max_tokens if strat == "J5" else self.a.max_tokens)
        att = 1 + (self.a.think_retries if thinking else self.a.retries)
        jobs = []
        if strat == "J5":
            for t in self.tasks:
                keys = [it["key"] for it in self.items if it["task"] == t]
                reps = judging.ranking_batches(keys, self.a.rank_batch, self.a.R, judging.stable_int(self.a.seed, t))
                sysmsg = self.system(t, facts)
                for r, batches in enumerate(reps):
                    for b, members in enumerate(batches):
                        def build(mem, attempt, _s=sysmsg, _t=t, _r=r):
                            if attempt > 1:   # re-ask the missing ideas in a reshuffled smaller batch
                                rng = np.random.default_rng(judging.stable_int(self.a.seed, tuple(mem), attempt))
                                mem[:] = [mem[int(i)] for i in rng.permutation(len(mem))]
                            return judging.messages_ranking(_s, [self.by_key[m]["description"] for m in mem],
                                                            self.refs[_t], self.a.rank_max_idea_chars)
                        jobs.append(Job(cond, "rank", members, r, build, thinking, cap, max_attempts=att,
                                        meta={"rep": r, "batch": b}))
            return jobs
        kind = CALL_KIND[strat]
        n_samples = K if strat in ("J1", "J2") else 1
        for it in self.items:
            sysmsg = self.system(it["task"], facts)
            if kind == "basic":
                msgs = judging.messages_single("basic", sysmsg, it["description"])
            elif kind == "rubric":
                msgs = judging.messages_single("rubric", sysmsg, it["description"])
            else:
                ex = self.fs_real_examples(it) if kind == "fs_real" else self.fs_surr_examples(it)
                msgs = judging.messages_single("fewshot", sysmsg, it["description"], ex)
            for k in range(n_samples):
                jobs.append(Job(cond, "single", [it["key"]], k, lambda mem, attempt, _m=msgs: _m, thinking, cap,
                                max_attempts=att))
        return jobs

    # -------------------------------------------------------------- execution
    def _record(self, job: Job, r: L.ChatResult | None, err: str | None) -> dict:
        if job.kind == "rank":
            scores = scoring.parse_ranked(r.text, len(job.members)) if r else [None] * len(job.members)
        else:
            scores = [scoring.parse_boxed(r.text) if r else None]
        return {"key": job.key, "cond": job.cond, "kind": job.kind, "members": job.members, "k": job.k,
                "attempt": job.attempt, "meta": job.meta, "scores": scores,
                "finish": r.finish_reason if r else None, "error": err,
                "prompt_tokens": r.prompt_tokens if r else 0, "completion_tokens": r.completion_tokens if r else 0,
                "latency": round(r.latency, 2) if r else 0.0, "max_tokens": job.max_tokens,
                "answer": (r.text if r else "")[-6000:], "t": time.time()}

    async def run_jobs(self, jobs: list[Job]) -> None:
        if not jobs:
            return
        replicas = []
        for port in self.a.ports:
            cfg = {"base_url": f"http://127.0.0.1:{port}/v1", "model": "qwen", "max_inflight": 10_000, "purposes": {}}
            usage = self.out / f"usage_{port}.jsonl"
            client = (L.MockLLM(cfg, usage, seed=port) if self.a.mock else L.LLM(cfg, usage, timeout_s=3600))
            replicas.append({"port": port, "llm": client, "used": 0, "n": 0, "done": 0})
        queue = collections.deque(sorted(jobs, key=lambda j: -j.need))     # longest first: short tail
        cond = asyncio.Condition()
        state = {"running": 0, "new_calls": 0}
        tasks: set[asyncio.Task] = set()
        t0 = time.time()
        total = len(jobs)

        def pick(rep):
            if rep["n"] >= self.a.max_seqs:
                return None
            for j in queue:
                if j.key in self.cache or rep["used"] + j.need <= self.a.kv_budget or rep["used"] == 0:
                    return j
            return None

        async def execute(rep, job):
            rec = None
            try:
                if job.key in self.cache:
                    rec = self.cache[job.key]
                else:
                    purpose = f"judge_{'t' if job.thinking else 'n'}{job.max_tokens}"
                    rep["llm"].purposes.setdefault(purpose, dict(job.settings))
                    try:
                        r = await rep["llm"].chat(job.messages, purpose)
                        rec = self._record(job, r, None)
                    except Exception as e:  # noqa: BLE001
                        rec = self._record(job, None, repr(e)[:300])
                    with open(self.calls_p, "a") as f:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    self.cache[job.key] = rec
                    state["new_calls"] += 1
                self.records.append(rec)
            finally:
                follow = None
                if rec is not None:
                    missing = [m for m, s in zip(rec["members"], rec["scores"]) if s is None]
                    follow = job.followup(missing) if missing else None
                async with cond:
                    rep["used"] -= job.reserved
                    rep["n"] -= 1
                    rep["done"] += 1
                    state["running"] -= 1
                    if follow is not None:
                        queue.appendleft(follow)
                    cond.notify_all()

        async def dispatch(rep):
            last_log = 0.0
            while True:
                async with cond:
                    while True:
                        if not queue and state["running"] == 0:
                            cond.notify_all()
                            return
                        job = pick(rep) if queue else None
                        if job is not None:
                            break
                        await cond.wait()
                    queue.remove(job)
                    job.reserved = 0 if job.key in self.cache else job.need
                    rep["used"] += job.reserved
                    rep["n"] += 1
                    state["running"] += 1
                task = asyncio.create_task(execute(rep, job))
                tasks.add(task)                      # strong reference until done
                task.add_done_callback(tasks.discard)
                if time.time() - last_log > 60:
                    last_log = time.time()
                    print(f"[{time.time() - t0:7.0f}s] port {rep['port']}: in-flight {rep['n']} kv {rep['used']} "
                          f"done {rep['done']}  queue {len(queue)}/{total}  new calls {state['new_calls']}", flush=True)

        await asyncio.gather(*[dispatch(r) for r in replicas])
        for r in replicas:
            await r["llm"].aclose()
        print(f"jobs done in {time.time() - t0:.0f}s; new calls {state['new_calls']}", flush=True)

    # -------------------------------------------------------------- scores
    def variant_scores(self, source: str, strat: str) -> dict:
        """source = the call condition (J1 reads J2's calls). -> {"scores": {key: float}, "per": {key: {k: float}},
        "calls": [records]}; per[k] = score of sample k (J1/J2) or of repetition k (J5)."""
        recs, seen = [], set()
        for r in self.records:
            if r["cond"] == source and r["key"] not in seen:
                seen.add(r["key"])
                recs.append(r)
        per: dict[str, dict[int, float]] = collections.defaultdict(dict)
        for r in recs:
            for m, s in zip(r["members"], r["scores"]):
                if s is not None:
                    per[m][r["k"]] = s
        if strat == "J1":
            scores = {m: d[0] for m, d in per.items() if 0 in d}
            recs = [r for r in recs if r["k"] == 0]    # tokens / failures of J1 = draw-0 calls only
        else:
            scores = {m: float(np.mean(list(d.values()))) for m, d in per.items() if d}
        return {"scores": scores, "per": per, "calls": recs}


# ============================================================================ statistics
def spearman(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or np.all(x == x[0]) or np.all(y == y[0]):
        return float("nan")
    rx, ry = stats.rankdata(x), stats.rankdata(y)      # average ranks for ties (= scipy spearmanr)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    return float((rx @ ry) / np.sqrt((rx @ rx) * (ry @ ry)))


def holm(pvals: dict[str, float]) -> dict[str, float]:
    keys = [k for k, p in sorted(pvals.items(), key=lambda kv: kv[1]) if p == p]
    m, out, run = len(keys), {}, 0.0
    for i, k in enumerate(keys):
        run = max(run, min(1.0, (m - i) * pvals[k]))
        out[k] = run
    return out


def analyze(study: Study, variants: list[tuple[str, str, str, int]], B: int, seed: int) -> dict:
    """variants = [(name, strategy, source call condition, K draws for J1)]."""
    items = study.items
    keys = [it["key"] for it in items]
    u = {it["key"]: it["u"] for it in items}
    rng = np.random.default_rng(seed)
    boot_idx = [rng.integers(0, len(keys), len(keys)) for _ in range(B)] if len(keys) >= 4 else []
    res, vecs = {}, {}
    for cond, strat, source, K in variants:
        vs = study.variant_scores(source, strat)
        sc, recs = vs["scores"], vs["calls"]
        if not recs:
            continue
        s = np.array([sc.get(k, np.nan) for k in keys])
        uu = np.array([u[k] for k in keys])
        ok = ~np.isnan(s)
        # J1: rho averaged over the K independent single draws
        if strat == "J1":
            draws = [np.array([vs["per"].get(k, {}).get(d, np.nan) for k in keys]) for d in range(K)]
        else:
            draws = [s]
        vecs[cond] = draws

        def rho_of(idx, _draws=draws):
            vals = []
            for dv in _draws:
                x, y = dv[idx], uu[idx]
                m = ~np.isnan(x)
                vals.append(spearman(x[m], y[m]))
            return float(np.nanmean(vals)) if not all(v != v for v in vals) else float("nan")

        all_idx = np.arange(len(keys))
        rho = rho_of(all_idx)
        boots = np.array([rho_of(ix) for ix in boot_idx]) if boot_idx else np.array([])
        boots = boots[~np.isnan(boots)]
        ci = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))] if len(boots) else [None, None]
        per_task = {}
        for t in study.tasks:
            ix = np.array([i for i, it in enumerate(items) if it["task"] == t])
            per_task[t] = rho_of(ix) if len(ix) else float("nan")
        fit = {"a": None, "b": None}
        if ok.sum() >= 3 and np.ptp(s[ok]) > 0:
            b, a0 = np.polyfit(s[ok], np.clip(uu[ok], 0, 1), 1)
            fit = {"a": float(a0), "b": float(b)}
        n_calls = len(recs)
        if strat == "J5":
            slots = sum(len(r["members"]) for r in recs)
            parse_fail = sum(1 for r in recs for x in r["scores"] if x is None) / max(1, slots)
        else:
            parse_fail = sum(1 for r in recs if r["scores"][0] is None) / max(1, n_calls)
        pt = sum(r["prompt_tokens"] for r in recs)
        ct = sum(r["completion_tokens"] for r in recs)
        res[cond] = {
            "strategy": strat, "n_ok": int(ok.sum()), "n_items": len(keys), "missing_rate": float(1 - ok.mean()) if len(keys) else None,
            "rho": rho, "ci95": ci, "per_task_rho": per_task,
            "mean_within_task_rho": float(np.nanmean(list(per_task.values()))) if per_task and not all(v != v for v in per_task.values()) else float("nan"),
            "linear_fit": fit, "mean_score": float(np.nanmean(s)) if ok.any() else None,
            "calls": n_calls, "prompt_tokens_per_idea": pt / max(1, len(keys)),
            "completion_tokens_per_idea": ct / max(1, len(keys)), "tokens_per_idea": (pt + ct) / max(1, len(keys)),
            "truncation_rate": sum(1 for r in recs if r["finish"] == "length") / max(1, n_calls),
            "call_error_rate": sum(1 for r in recs if r["error"]) / max(1, n_calls),
            "parse_failure_rate": parse_fail,
            "scores": {k: sc[k] for k in keys if k in sc},
        }
        if strat == "J1":
            d0 = draws[0]
            m = ~np.isnan(d0)
            res[cond]["rho_draw1"] = spearman(d0[m], uu[m])
    # paired bootstrap vs baseline
    pvals = {}
    if BASELINE in vecs and boot_idx:
        uu = np.array([u[k] for k in keys])

        def rho_draws(draws, idx):
            vals = []
            for dv in draws:
                x, y = dv[idx], uu[idx]
                m = ~np.isnan(x)
                vals.append(spearman(x[m], y[m]))
            return float(np.nanmean(vals))
        base = vecs[BASELINE]
        for cond, draws in vecs.items():
            if cond == BASELINE:
                continue
            # common ideas: present in the variant and in the baseline draw 1
            common = np.array([i for i in range(len(keys)) if not np.isnan(draws[0][i]) and not np.isnan(base[0][i])])
            if len(common) < 4:
                continue
            d_obs = rho_draws(draws, common) - rho_draws(base, common)
            ds = []
            for ix in boot_idx:
                sub = common[rng.integers(0, len(common), len(common))]
                d = rho_draws(draws, sub) - rho_draws(base, sub)
                if d == d:
                    ds.append(d)
            ds = np.array(ds)
            p = float(min(1.0, 2 * min((ds <= 0).mean(), (ds >= 0).mean()))) if len(ds) else float("nan")
            res[cond]["delta_vs_baseline"] = {"delta_rho": d_obs, "ci95": [float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))] if len(ds) else [None, None],
                                              "p": p, "n_common": int(len(common))}
            pvals[cond] = p
    for cond, ph in holm(pvals).items():
        res[cond]["delta_vs_baseline"]["p_holm"] = ph
    # PRIMARY (2026-11-09): mean WITHIN-task rho. Pooled rho also rewards between-task calibration, which the
    # pipeline never uses (it only ranks ideas within one run). Paired bootstrap resamples ideas within each task.
    pw = {}
    if BASELINE in vecs and boot_idx:
        uu = np.array([u[k] for k in keys])
        task_arr = np.array([it["task"] for it in items])
        base = vecs[BASELINE]
        rng_w = np.random.default_rng(12345)
        for cond, draws in vecs.items():
            if cond == BASELINE:
                continue
            common = np.array([i for i in range(len(keys)) if not np.isnan(draws[0][i]) and not np.isnan(base[0][i])])
            if len(common) < 4:
                continue
            groups = [common[task_arr[common] == t] for t in study.tasks]
            groups = [g for g in groups if len(g) >= 4]
            d_obs = within_rho(draws, uu, groups) - within_rho(base, uu, groups)
            ds = []
            for _ in range(len(boot_idx)):
                gs = [g[rng_w.integers(0, len(g), len(g))] for g in groups]
                d = within_rho(draws, uu, gs) - within_rho(base, uu, gs)
                if d == d:
                    ds.append(d)
            ds = np.array(ds)
            pv = float(min(1.0, 2 * min((ds <= 0).mean(), (ds >= 0).mean()))) if len(ds) else float("nan")
            res[cond]["delta_within_vs_baseline"] = {
                "delta_within_rho": d_obs, "p": pv, "n_common": int(len(common)),
                "ci95": [float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))] if len(ds) else [None, None]}
            pw[cond] = pv
    for cond, ph in holm(pw).items():
        res[cond]["delta_within_vs_baseline"]["p_holm"] = ph
    s0 = np.array([it["s_orig"] if it["s_orig"] is not None else np.nan for it in items])
    m = ~np.isnan(s0)
    res["_original_pipeline_s_cal"] = {"rho": spearman(s0[m], np.array([it["u"] for it in items])[m]), "n_ok": int(m.sum())}
    res["_loto"] = loto(vecs, items, study.tasks)
    return res


def _rho_draws(draws, uu, idx) -> float:
    """rho over ideas idx; J1 = mean over its single draws (as in the main table)."""
    vals = []
    for dv in draws:
        x, y = dv[idx], uu[idx]
        m = ~np.isnan(x)
        vals.append(spearman(x[m], y[m]))
    vals = [v for v in vals if v == v]
    return float(np.mean(vals)) if vals else float("nan")


def within_rho(draws, uu, groups) -> float:
    """Mean over tasks (groups of idea indices) of the per-task rho (J1-style mean over draws)."""
    vals = [_rho_draws(draws, uu, g) for g in groups if len(g) >= 3]
    vals = [v for v in vals if v == v]
    return float(np.mean(vals)) if vals else float("nan")


def loto(vecs: dict, items: list[dict], tasks: list[str]) -> dict:
    """Leave-one-task-out selection check (analysis only). For each held-out task T, pick the variant with the
    best MEAN WITHIN-TASK rho vs u on the OTHER tasks (selection criterion since 2026-11-09), then report that
    variant's rho on T. Candidates: every
    no-thinking variant (thinking variants are excluded: which ones exist was chosen using all tasks). J4r is a
    strategy choice, not a fit, so it is treated like any other variant."""
    cands = {c: d for c, d in vecs.items() if not c.endswith("_think")}
    uu = np.array([it["u"] for it in items])
    task_of = np.array([it["task"] for it in items])
    if not cands or len(tasks) < 2:
        return {"note": "needs >= 2 tasks and >= 1 variant", "folds": {}}

    def best(idx):
        groups = [idx[task_of[idx] == t] for t in tasks if (task_of[idx] == t).any()]
        scored = [(c, within_rho(d, uu, groups)) for c, d in cands.items()]
        scored = [(c, r) for c, r in scored if r == r]
        return max(scored, key=lambda cr: (cr[1], cr[0])) if scored else (None, float("nan"))

    all_idx = np.arange(len(items))
    overall, overall_rho = best(all_idx)
    folds, held = {}, []
    for t in tasks:
        tr, te = all_idx[task_of != t], all_idx[task_of == t]
        ch, tr_rho = best(tr)
        te_rho = _rho_draws(cands[ch], uu, te) if ch else float("nan")
        folds[t] = {"chosen": ch, "train_rho": tr_rho, "heldout_rho": te_rho, "n_heldout": int(len(te)),
                    "matches_overall": ch == overall}
        if te_rho == te_rho:
            held.append(te_rho)
    per_task = {c: {t: _rho_draws(d, uu, all_idx[task_of == t]) for t in tasks} for c, d in cands.items()}
    return {"overall_choice": overall, "overall_rho": overall_rho, "folds": folds,
            "mean_heldout_rho": float(np.mean(held)) if held else float("nan"),
            "min_heldout_rho": float(np.min(held)) if held else float("nan"),
            "match_rate": float(np.mean([f["matches_overall"] for f in folds.values()])),
            "per_task_rho": per_task}


def fmt(x, nd=2):
    return "—" if x is None or (isinstance(x, float) and x != x) else f"{x:.{nd}f}"


def loto_markdown(lo: dict, tasks: list[str]) -> list[str]:
    if not lo.get("folds"):
        return ["", "## LOTO", "", lo.get("note", "not available")]
    short = {t: t.split("_")[0] for t in tasks}
    out = ["", "## LOTO", "",
           "Leave-one-task-out: for each held-out task, the variant with the best mean within-task ρ on the other tasks "
           "(no-thinking variants only), and its ρ on the held-out task.", "",
           "| held-out task | chosen on other tasks | mean within ρ, other tasks | ρ held-out | n | = overall choice |", "|---|---|---|---|---|---|"]
    for t, f in lo["folds"].items():
        out.append(f"| {t} | {f['chosen']} | {fmt(f['train_rho'])} | {fmt(f['heldout_rho'])} | {f['n_heldout']} | "
                   f"{'yes' if f['matches_overall'] else 'no'} |")
    out += ["", f"Overall choice (all tasks): {lo['overall_choice']} (ρ = {fmt(lo['overall_rho'])}). "
            f"Held-out ρ: mean {fmt(lo['mean_heldout_rho'])}, min {fmt(lo['min_heldout_rho'])}; "
            f"chosen = overall in {lo['match_rate'] * 100:.0f}% of folds.", "",
            "Per-task ρ by variant (a single task driving the winner shows up here):", "",
            "| variant | " + " | ".join(short[t] for t in tasks) + " |", "|---|" + "---|" * len(tasks)]
    pt = lo["per_task_rho"]
    def mean_rho(c):
        v = [x for x in pt[c].values() if x == x]
        return -float(np.mean(v)) if v else 9.0
    for c in sorted(pt, key=mean_rho):
        out.append(f"| {c} | " + " | ".join(fmt(pt[c][t]) for t in tasks) + " |")
    return out


def markdown(res: dict, tasks: list[str], meta: dict) -> str:
    short = {t: t.split("_")[0] for t in tasks}
    head = ["variant", "n", "**mean within-task ρ**", "Δ within vs J1 (95% CI)", "p_holm (within)"] + \
           [f"ρ {short[t]}" for t in tasks] + ["pooled ρ (95% CI)", "Δ pooled vs J1", "fit a, b", "tok/idea", "trunc", "parse fail"]
    lines = [f"# Judge study ({meta['pool']}, {meta['n_items']} ideas, {meta['when']})", "",
             f"Target: unclipped u = (value − floor)/(ref − floor). Baseline {BASELINE} (ρ = mean over its {meta['K']} "
             f"single draws). Bootstrap B = {meta['B']} over ideas; p two-sided, Holm across variants.", "",
             "**Selection criterion: mean within-task ρ** (pre-registered 2026-11-09). Pooled ρ also rewards between-task "
             "calibration, which the pipeline never uses; it is shown for reference. The within-task bootstrap resamples "
             "ideas within each task.", "",
             "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    wk = lambda k: res[k].get("mean_within_task_rho")
    order = sorted((k for k in res if not k.startswith("_") and "rho" in res[k]), key=lambda k: -(wk(k) if wk(k) == wk(k) and wk(k) is not None else -9))
    for k in order:
        r = res[k]
        d = r.get("delta_vs_baseline") or {}
        dw = r.get("delta_within_vs_baseline") or {}
        lines.append("| " + " | ".join([
            k, str(r["n_ok"]), f"**{fmt(r.get('mean_within_task_rho'))}**",
            (f"{fmt(dw.get('delta_within_rho'))} [{fmt(dw['ci95'][0])}, {fmt(dw['ci95'][1])}]" if dw else "—"),
            fmt(dw.get("p_holm"), 3) if dw else "—",
            *[fmt(r["per_task_rho"].get(t)) for t in tasks],
            f"{fmt(r['rho'])} [{fmt(r['ci95'][0])}, {fmt(r['ci95'][1])}]",
            (f"{fmt(d.get('delta_rho'))} (p_holm {fmt(d.get('p_holm'), 3)})" if d else "—"),
            f"{fmt(r['linear_fit']['a'])}, {fmt(r['linear_fit']['b'])}",
            f"{r['tokens_per_idea']:.0f}", fmt(r["truncation_rate"]), fmt(r["parse_failure_rate"])]) + " |")
    o = res.get("_original_pipeline_s_cal") or {}
    lines += ["", f"Original pipeline s_cal: ρ = {fmt(o.get('rho'))} (n = {o.get('n_ok')}).",
              f"Thinking variants: {', '.join(meta.get('think_selected') or []) or 'none'}."]
    lines += loto_markdown(res.get("_loto") or {}, tasks)
    return "\n".join(lines) + "\n"


# ============================================================================ main
def plan_report(jobs_by_cond: dict[str, list[Job]], a) -> dict:
    rows, tot_c, tot_p, tot_calls = {}, 0, 0, 0
    for cond, jobs in jobs_by_cond.items():
        strat = cond.split("_")[0]
        think = cond.endswith("_think")
        comp = sum(min(j.max_tokens, EST_COMPLETION_THINK if think else
                       EST_COMPLETION[CALL_KIND[strat]] * (len(j.members) / 2.5 if strat == "J5" else 1)) for j in jobs)
        prm = sum(j.prompt_est for j in jobs)
        rows[cond] = {"calls": len(jobs), "prompt_tokens": prm, "completion_tokens_est": int(comp)}
        tot_c += comp
        tot_p += prm
        tot_calls += len(jobs)
    tput = a.plan_tok_s * len(a.ports)
    return {"conds": rows, "calls": tot_calls, "prompt_tokens": tot_p, "completion_tokens_est": int(tot_c),
            "est_hours_at_decode_rate": tot_c / tput / 3600, "assumed_decode_tok_s_total": tput,
            "note": "first attempts only (re-asks add ~parse-failure rate); J1 shares J2's calls"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pool", default="train", help="train | test | comma-separated run-id globs under $AR_ROOT/runs")
    ap.add_argument("--strategies", default=",".join(STRATS))
    ap.add_argument("--facts", default="both", choices=["both", "on", "off"])
    ap.add_argument("--K", type=int, default=5, help="J2 samples (J1 = their single draws)")
    ap.add_argument("--R", type=int, default=2, help="J5 batches per idea")
    ap.add_argument("--rank-batch", type=int, default=5)
    ap.add_argument("--rank-max-idea-chars", type=int, default=12000)
    ap.add_argument("--fewshot-n", type=int, default=6)
    ap.add_argument("--retries", type=int, default=2, help="re-asks after an unparseable/failed call")
    ap.add_argument("--think-retries", type=int, default=1, help="re-asks for thinking calls (truncation is costly)")
    ap.add_argument("--max-tokens", type=int, default=4096, help="no-thinking cap (J1-J4)")
    ap.add_argument("--rank-max-tokens", type=int, default=8192, help="no-thinking cap (J5)")
    ap.add_argument("--think-max-tokens", type=int, default=16000, help="thinking cap (also capped by context)")
    ap.add_argument("--think-top", type=int, default=2, help="thinking for the best N strategies (0 = none)")
    ap.add_argument("--think-variants", default="", help="explicit thinking variants, e.g. J3_facts,J2_nofacts")
    ap.add_argument("--think-k", type=int, default=5, help="samples for a J1/J2 thinking variant")
    ap.add_argument("--phase", default="all", choices=["all", "nothink", "analyze"])
    ap.add_argument("--ports", type=int, nargs="+", default=[8000, 8001])
    ap.add_argument("--kv-budget", type=int, default=250_000, help="per-replica sum of prompt+max_tokens in flight")
    ap.add_argument("--max-seqs", type=int, default=48, help="per-replica max in-flight requests")
    ap.add_argument("--B", type=int, default=2000, help="bootstrap resamples")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="first N pool ideas only (smoke)")
    ap.add_argument("--mock", action="store_true", help="MockLLM instead of vLLM (offline smoke)")
    ap.add_argument("--plan", action="store_true", help="count calls and estimate tokens; no LLM calls")
    ap.add_argument("--plan-tok-s", type=float, default=700.0, help="--plan: decode tok/s per replica")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    items, journals = load_pool(a.pool, a.limit)
    if not items:
        sys.exit(f"no real-evaluated ideas in pool {a.pool!r}")
    by_task = collections.Counter(it["task"] for it in items)
    print(f"pool {a.pool}: {len(items)} ideas {dict(by_task)}", flush=True)
    study = Study(a, items, journals)
    strats = [s for s in a.strategies.split(",") if s]
    facts_opts = {"both": [False, True], "on": [True], "off": [False]}[a.facts]
    # J1 is computed from J2's calls; make sure they exist
    nothink = [(s, f) for s in strats for f in facts_opts]
    call_strats = sorted({("J2" if s == "J1" else s, f) for s, f in nothink})
    jobs_by_cond: dict[str, list[Job]] = {}
    for s, f in call_strats:
        js = study.jobs_for(s, f, False, a.K)
        jobs_by_cond[js[0].cond if js else f"{s}_{f}"] = js
    if a.plan:
        rep = plan_report(jobs_by_cond, a)
        think_est = {"note": "thinking phase: per selected variant ~ (calls of that variant) x ~11k completion tokens"}
        print(json.dumps({"nothink": rep, "think": think_est}, indent=1))
        return
    if a.phase != "analyze":
        asyncio.run(study.run_jobs([j for js in jobs_by_cond.values() for j in js]))
    else:
        for js in jobs_by_cond.values():   # analysis from cache only: replay cached attempts
            _replay(study, js)

    def variants_for(pairs, think):
        out_ = []
        for s, f in pairs:
            base = f"{'J2' if s == 'J1' else s}_{'facts' if f else 'nofacts'}{'_think' if think else ''}"
            name = f"{s}_{'facts' if f else 'nofacts'}{'_think' if think else ''}"
            out_.append((name, s, base))
        return out_

    def run_analysis(pairs_nt, pairs_t, B):
        vlist = [(n, s, b, a.K) for n, s, b in variants_for(pairs_nt, False)] + \
                [(n, s, b, a.think_k) for n, s, b in variants_for(pairs_t, True)]
        return analyze(study, vlist, B, a.seed)

    # selection for the thinking phase needs only the point rhos (no bootstrap: keeps the GPUs busy)
    want_think = a.phase != "nothink" and (bool(a.think_variants) or a.think_top > 0)
    res = run_analysis(nothink, [], 0 if want_think else a.B)
    think_pairs = []
    if not want_think:
        pass
    elif a.think_variants:
        for v in a.think_variants.split(","):
            s, f = v.split("_")[0], v.split("_")[1] == "facts"
            think_pairs.append((s, f))
    elif a.think_top > 0:
        ranked = sorted(((k, r) for k, r in res.items() if not k.startswith("_") and r.get("mean_within_task_rho") == r.get("mean_within_task_rho")),
                        key=lambda kv: -kv[1]["mean_within_task_rho"])   # selection criterion: mean within-task rho
        seen = set()
        for k, r in ranked:   # J1 and J2 are one family (same calls): thinking J2 calls give both
            fam = "J12" if r["strategy"] in ("J1", "J2") else r["strategy"]
            if fam in seen:
                continue
            seen.add(fam)
            f = k.split("_")[1] == "facts"
            think_pairs += [("J1", f), ("J2", f)] if fam == "J12" else [(r["strategy"], f)]
            if len(seen) >= a.think_top:
                break
    think_selected = [f"{s}_{'facts' if f else 'nofacts'}_think" for s, f in think_pairs]
    if think_pairs:
        print("thinking variants:", think_selected, flush=True)
        tj = []
        for s, f in sorted({("J2" if s == "J1" else s, f) for s, f in think_pairs}):
            tj += study.jobs_for(s, f, True, a.think_k)
        if a.phase == "all":
            asyncio.run(study.run_jobs(tj))
        else:
            _replay(study, tj)
    if want_think:
        res = run_analysis(nothink, think_pairs, a.B)
    meta = {"pool": a.pool, "n_items": len(items), "per_task": dict(by_task), "K": a.K, "R": a.R, "B": a.B,
            "rank_batch": a.rank_batch, "fewshot_n": a.fewshot_n, "think_selected": think_selected,
            "when": time.strftime("%Y-%m-%d %H:%M"), "args": vars(a)}
    items_out = [{k: v for k, v in it.items() if k not in ("description", "summary")} for it in items]
    (out / "summary.json").write_text(json.dumps({"meta": meta, "items": items_out, "results": res}, indent=1))
    md = markdown(res, study.tasks, meta)
    (out / "judge_study.md").write_text(md)
    print(md)


def _replay(study: Study, jobs: list[Job]) -> None:
    """Analysis-only: walk each job's attempts through the cache (no LLM)."""
    stack = list(jobs)
    while stack:
        j = stack.pop()
        rec = study.cache.get(j.key)
        if rec is None:
            continue
        study.records.append(rec)
        missing = [m for m, s in zip(rec["members"], rec["scores"]) if s is None]
        f = j.followup(missing) if missing else None
        if f is not None:
            stack.append(f)


if __name__ == "__main__":
    main()
