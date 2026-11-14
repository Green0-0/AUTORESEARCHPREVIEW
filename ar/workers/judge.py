"""Surrogate judge worker (spec "Judging", "Objective: selective improvement"; plan §9.1 strategies).

``judge.strategy`` selects how ONE judgment (an absolute score in [0,1] per idea) is made:
* ``basic`` (J1, default): one "Basic" call, ``\\boxed{}`` parsed. Identical to the original judge.
* ``basic_k`` (J2): ``judge.k`` independent "Basic" calls; the judgment is the mean of the parsed samples.
* ``rubric`` (J3): one "Arbitrary Rubric" call; the final ``\\boxed{}`` (the rubric average) is the score.
* ``fewshot`` (J4): one "Basic FS" call with ``judge.fewshot_n`` stratified examples (ar/judging.py): same-run
  earlier ideas with their s_cal (``fewshot_source: surrogate``) or ideas with real results, from this run and
  ``judge.fewshot_runs`` (``real``, shown as real_score). No examples yet -> the Basic prompt.
* ``ranking`` (J5): the worker claims a batch of up to ``judge.rank_batch`` ideas (waiting at most
  ``judge.rank_max_wait_s`` for a full batch) and makes one "Ranking" call with the task's starter (known
  result 0) and reference method (known result 1) as reference ideas; every idea gets its own score.

The system prompt is always judge_system(prompt, judge facts) (judge-only facts: ar/judging.judge_facts).

F mechanics are the same for every strategy (LLM calls always outside the lock):
1. first judgment; an unparseable answer (or a call error) is retried up to ``judge.retries`` times, then
   the idea is failed (ranking: only the ideas whose score is missing are re-asked, in a smaller batch).
2. transaction: raw score + output recorded, ``s_cal = calibrate(s_raw)``, first F computed against the
   comparison set at that moment; b_t frozen in ``F_detail``; status ``judged``.
3. if F_first > 0: a second judgment with the same strategy (independent draw; few-shot: a fresh example
   draw; ranking: the idea joins a later, differently composed batch), then ``s_cal = calibrate(mean)`` and
   F recomputed against the frozen b_t.
4. transaction: F, F_detail, scored_seq (counter) set; status ``scored``.

``surrogate_scores`` / ``judge_outputs`` hold one entry per judgment (1 or 2). Non-basic strategies also
record ``judge_detail`` (one dict per judgment: strategy, samples / examples / batch). Ideas found in
``judged`` (process stopped between 2 and 4) are resumed from their frozen F_detail. ``F_detail = {"b_t",
"s_first", "s_final", "confirmed", "confirmation_call", "best_tag", "F_first"}``.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time

import numpy as np

from ar import anchors, judging, prompts, scoring
from ar import config as arconfig
from ar import journal as J

log = logging.getLogger("ar.judge")
WORKER = "judge"
POLL_S = 1.0
SNAP_KEYS = ("status", "description", "surrogate_scores", "F_detail", "summary")


def _eligible(idea: dict) -> bool:
    return idea.get("status") in ("tagged", "judged")


def _needs_call(idea: dict) -> bool:
    """tagged -> first judgment; judged with F_first > 0 and one score -> confirmation; else finalize only."""
    if idea.get("status") == "tagged":
        return True
    det = idea.get("F_detail") or {}
    return idea.get("status") == "judged" and det.get("F_first", 0) > 0 and len(idea.get("surrogate_scores") or []) < 2


class Judge:
    def __init__(self, journal: J.Journal, cfg: dict, llm, stop: asyncio.Event,
                 concurrency: int | None = None, poll_s: float = POLL_S):
        self.j, self.cfg, self.llm, self.stop = journal, cfg, llm, stop
        self.task_prompt = arconfig.task_prompt(cfg)
        jc = cfg["judge"]
        self.strategy = jc.get("strategy", "basic")
        self.facts = judging.judge_facts(cfg)
        self.seed = int(cfg["seed"])
        self.retries = int(jc["retries"])
        self.k = int(jc.get("k", 5))
        self.fs_source = jc.get("fewshot_source", "surrogate")
        self.fs_n = int(jc.get("fewshot_n", 6))
        self.rank_batch = int(jc.get("rank_batch", 5))
        self.rank_wait = float(jc.get("rank_max_wait_s", 60))
        self.rank_chars = jc.get("rank_max_idea_chars", 12000)
        self.refs = anchors.ranking_references(cfg["task_id"]) if self.strategy == "ranking" else None
        purposes = (cfg.get("llm") or {}).get("purposes") or {}
        self.rank_purpose = "judge_rank" if "judge_rank" in purposes else "judge"
        self.ext_real = (self._load_external_real(jc.get("fewshot_runs") or [])
                         if self.strategy == "fewshot" and self.fs_source == "real" else [])
        sc = cfg["scoring"]
        self.s_min, self.s_base, self.cal = float(sc["s_min"]), float(sc["s_base"]), sc["calibration"]
        self.timeout_s = float(cfg["claims"]["timeout_minutes"]) * 60
        # the LLM semaphore is the real cap; this only bounds how much we claim at once (in ideas)
        self.concurrency = int(concurrency or cfg["llm"].get("max_inflight", 64))
        self.poll_s = poll_s
        self.stats = {"judged": 0, "scored": 0, "confirmations": 0, "failed": 0, "length": 0}

    # ------------------------------------------------------------------ prompts
    def _system(self, iid: str) -> str:
        rng = np.random.default_rng([self.seed, int(iid.lstrip("i")), 104729])
        return prompts.judge_system(self.task_prompt, self.facts, rng)

    def _load_external_real(self, run_ids: list[str]) -> list[dict]:
        out = []
        for rid in run_ids:
            p = self.j.run_dir.parent / rid / J.JOURNAL
            try:
                st = json.loads(p.read_text())
            except (OSError, ValueError) as e:
                log.warning("fewshot_runs: cannot read %s: %r", p, e)
                continue
            if (st.get("config") or {}).get("task_id") != self.cfg["task_id"]:
                log.warning("fewshot_runs: %s is a different task; skipped", rid)
                continue
            for i in st["ideas"].values():
                if i.get("real_score") is not None and (i.get("real") or {}).get("value") is not None:
                    out.append({"id": f"{rid}/{i['id']}", "summary": i.get("summary"), "score": float(i["real_score"])})
        return out

    def _fs_select(self, st: dict, iid: str, draw: int) -> list[dict]:
        idea = st["ideas"][iid]
        if self.fs_source == "surrogate":
            cands = [{"id": i["id"], "summary": i.get("summary"), "score": float(i["s_cal"])}
                     for i in scoring.comparison_set(st, iid)]
        else:
            cands = [{"id": i["id"], "summary": i.get("summary"), "score": float(i["real_score"])}
                     for i in st["ideas"].values()
                     if i["id"] != iid and i.get("status") != "failed" and i.get("real_score") is not None
                     and (i.get("real") or {}).get("value") is not None] + self.ext_real
        rng = np.random.default_rng(judging.stable_int(self.seed, iid, draw, "fewshot"))
        chosen = judging.select_examples(cands, self.fs_n, rng, exclude_id=iid,
                                         exclude_texts=(idea.get("summary") or "",))
        return [{"id": c["id"], "summary": c["summary"], "score": c["score"]} for c in chosen]

    # ------------------------------------------------------------------ calls
    async def _call(self, messages: list[dict], purpose: str = "judge"):
        """-> (s_raw, raw_output, attempts, errors) or (None, last_output, attempts, errors)."""
        errors, last = [], ""
        for attempt in range(1 + self.retries):
            try:
                r = await self.llm.chat(messages, purpose)
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001
                errors.append(f"attempt {attempt + 1}: {e!r}"[:400])
                continue
            if r.finish_reason == "length":
                self.stats["length"] += 1
            last = r.raw
            s = scoring.parse_boxed(r.text)
            if s is not None:
                return s, r.raw, attempt + 1, errors
            errors.append(f"attempt {attempt + 1}: no parseable \\boxed{{}} "
                          f"(finish_reason={r.finish_reason})")
        return None, last, 1 + self.retries, errors

    async def _judge_call(self, system: str, description: str):
        return await self._call(judging.messages_single("basic", system, description))

    async def _judgment(self, iid: str, idea: dict, system: str, draw: int):
        """One judgment of one idea (J1-J4) -> (s, raw, attempts, errors, detail | None)."""
        desc = idea["description"]
        st = self.strategy
        if st == "basic":
            s, raw, att, errs = await self._judge_call(system, desc)
            return s, raw, att, errs, None
        if st == "rubric":
            s, raw, att, errs = await self._call(judging.messages_single("rubric", system, desc))
            return s, raw, att, errs, {"strategy": "rubric"}
        if st == "fewshot":
            chosen = (idea.get("_fs") or {}).get(draw) or []
            ex = judging.fewshot_examples(chosen)
            s, raw, att, errs = await self._call(judging.messages_single("fewshot", system, desc, ex))
            d = {"strategy": "fewshot", "source": self.fs_source,
                 "examples": [[c["id"], round(float(c["score"]), 4)] for c in chosen]}
            if not ex:
                d["fallback"] = "basic"
            return s, raw, att, errs, d
        if st == "basic_k":
            msgs = judging.messages_single("basic_k", system, desc)
            res = await asyncio.gather(*[self._call(msgs) for _ in range(self.k)])
            vals = [r[0] for r in res if r[0] is not None]
            errs = [e for r in res for e in r[3]]
            att = max(r[2] for r in res)
            parts = [r[1] if n == 0 else (r[1] or "")[-1500:] for n, r in enumerate(res)]
            raw = "".join((f"\n\n=== sample {n + 1}/{self.k} ===\n\n" if n else "") + p for n, p in enumerate(parts))
            if not vals:
                return None, raw, att, errs, None
            return float(np.mean(vals)), raw, att, errs, {"strategy": "basic_k", "samples": vals,
                                                          "failed": self.k - len(vals)}
        raise ValueError(f"_judgment: strategy {st!r}")

    # ------------------------------------------------------------------ journal writes
    def _fail(self, iid: str, msg: str, raw: str, stage: str) -> None:
        now = time.time()
        with self.j.transaction(WORKER, "fail") as st:
            idea = st["ideas"].get(iid)
            if idea is None or idea.get("status") == "failed":
                return
            if raw:
                idea.setdefault("judge_outputs", []).append(raw)
            idea.setdefault("retries", {})[stage] = self.retries
            J.fail(st, iid, msg, now)
        self.stats["failed"] += 1
        log.warning("judge failed %s: %s", iid, msg[:200])

    def _record_first(self, iid: str, s: float, raw: str, att: int, detail: dict | None,
                      keep_claim: bool) -> dict | None:
        """Transaction 2: first score, s_cal, first F, frozen b_t; status judged. -> idea copy or None."""
        now = time.time()
        with self.j.transaction(WORKER, "judged") as st:
            cur = st["ideas"].get(iid)
            if cur is None or cur.get("status") != "tagged":
                return None
            cur["surrogate_scores"] = [float(s)]
            cur["judge_outputs"] = [raw]
            if detail is not None:
                cur["judge_detail"] = [detail]
            cur["s_cal"] = scoring.calibrate(s, self.cal)
            F1, det = scoring.compute_F(st, iid, cur["s_cal"], self.s_min, self.s_base)
            if F1 is None:  # no tags: should not happen (tagger fails these)
                J.fail(st, iid, "judge: idea has no tags", now)
                self.stats["failed"] += 1
                return None
            cur["F_detail"] = {"b_t": det["b_t"], "s_first": cur["s_cal"], "s_final": None,
                               "F_first": F1, "best_tag": det.get("best_tag"),
                               "confirmed": False, "confirmation_call": F1 > 0}
            if att > 1:
                cur.setdefault("retries", {})["judge"] = att - 1
            J.set_status(st, iid, "judged", now)
            if keep_claim:
                cur["claim"] = {"worker": WORKER, "stage": "judge", "at": now}
            out = dict(cur)
        self.stats["judged"] += 1
        return out

    def _record_final(self, iid: str, new_scores: list[float], outputs_new: list[str],
                      details_new: list[dict]) -> None:
        """Transaction 3: confirmation (if any) folded in, F against the frozen b_t; status scored."""
        now = time.time()
        with self.j.transaction(WORKER, "scored") as st:
            cur = st["ideas"].get(iid)
            if cur is None or cur.get("status") != "judged":
                return
            det = cur["F_detail"]
            scores = list(cur.get("surrogate_scores") or []) + [float(x) for x in new_scores]
            cur["surrogate_scores"] = scores
            cur["judge_outputs"] = list(cur.get("judge_outputs") or []) + outputs_new
            if details_new:
                cur["judge_detail"] = list(cur.get("judge_detail") or []) + details_new
            s_final = scoring.calibrate(float(np.mean(scores)), self.cal)
            cur["s_cal"] = s_final
            if len(scores) > 1:
                F, d2 = scoring.compute_F(st, iid, s_final, self.s_min, self.s_base, b_t=det["b_t"])
                best = d2.get("best_tag")
            else:
                F, best = det["F_first"], det.get("best_tag")
            cur["F"] = float(F)
            cur["F_detail"] = {**det, "s_final": s_final, "best_tag": best,
                               "confirmation_call": len(scores) > 1,
                               "confirmed": len(scores) > 1 and F > 0}
            seq = int(st["counters"].get("scored_seq", 0))
            cur["scored_seq"] = seq
            st["counters"]["scored_seq"] = seq + 1
            J.set_status(st, iid, "scored", now)
        self.stats["scored"] += 1

    # ------------------------------------------------------------------ single-idea strategies (J1-J4)
    async def _process(self, iid: str, idea: dict) -> None:
        system = self._system(iid)
        if idea.get("status") == "tagged":
            s, raw, att, errs, detail = await self._judgment(iid, idea, system, 0)
            if s is None:
                self._fail(iid, "judge: " + "; ".join(errs), raw, "judge")
                return
            cur = self._record_first(iid, s, raw, att, detail, keep_claim=True)
            if cur is None:
                return
            idea = {**cur, "_fs": idea.get("_fs")}

        det = idea["F_detail"]
        new, outputs_new, details_new = [], [], []
        if det["F_first"] > 0 and len(idea["surrogate_scores"]) < 2:
            s2, raw2, _att2, errs2, d2 = await self._judgment(iid, idea, system, 1)
            if s2 is None:
                self._fail(iid, "judge confirmation: " + "; ".join(errs2), raw2, "judge_confirm")
                return
            new.append(float(s2))
            outputs_new.append(raw2)
            if d2 is not None:
                details_new.append(d2)
            self.stats["confirmations"] += 1
        self._record_final(iid, new, outputs_new, details_new)

    # ------------------------------------------------------------------ ranking strategy (J5)
    async def _process_batch(self, items: list[tuple[str, dict]]) -> None:
        snaps = dict(items)
        pending = list(items)
        results: dict[str, tuple] = {}
        errors: dict[str, list[str]] = {iid: [] for iid, _ in items}
        last_raw: dict[str, str] = {}
        ids_key = tuple(sorted(snaps))
        for attempt in range(1, 2 + self.retries):
            if not pending:
                break
            rng = np.random.default_rng(judging.stable_int(self.seed, ids_key, attempt, "rank"))
            order = [pending[int(i)] for i in rng.permutation(len(pending))]
            system = self._system(min(iid for iid, _ in order))
            msgs = judging.messages_ranking(system, [s["description"] for _, s in order], self.refs, self.rank_chars)
            try:
                r = await self.llm.chat(msgs, self.rank_purpose)
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001
                for iid, _ in pending:
                    errors[iid].append(f"attempt {attempt}: {e!r}"[:300])
                continue
            if r.finish_reason == "length":
                self.stats["length"] += 1
            vals = scoring.parse_ranked(r.text, len(order))
            still = []
            for pos, ((iid, snap), v) in enumerate(zip(order, vals), start=1):
                if v is None:
                    errors[iid].append(f"attempt {attempt}: no score for Idea {pos} (finish_reason={r.finish_reason})")
                    last_raw[iid] = r.raw
                    still.append((iid, snap))
                else:
                    results[iid] = (v, r.raw, attempt, {"strategy": "ranking", "batch": [i for i, _ in order],
                                                        "position": pos, "attempt": attempt})
            pending = still
        for iid, snap in pending:
            first = snap.get("status") == "tagged"
            self._fail(iid, ("judge ranking: " if first else "judge ranking confirmation: ") + "; ".join(errors[iid]),
                       last_raw.get(iid, ""), "judge" if first else "judge_confirm")
        for iid, (v, raw, att, det) in results.items():
            if snaps[iid].get("status") == "tagged":
                cur = self._record_first(iid, v, raw, att, det, keep_claim=False)
                if cur is not None and not cur["F_detail"]["F_first"] > 0:
                    self._record_final(iid, [], [], [])
                # F_first > 0: released in `judged`; a later batch makes the confirmation judgment
            else:
                self._record_final(iid, [v], [raw], [det])
                self.stats["confirmations"] += 1

    # ------------------------------------------------------------------ loop
    async def _guarded(self, coro, ids: list[str]) -> None:
        try:
            await coro
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            log.exception("judge crashed on %s", ids)
            for iid in ids:
                try:
                    self._fail(iid, f"judge crashed: {e!r}"[:500], "", "judge")
                except Exception:  # noqa: BLE001
                    pass

    def _snap(self, st: dict, iid: str) -> dict:
        idea = st["ideas"][iid]
        snap = {k: idea.get(k) for k in SNAP_KEYS}
        if self.strategy == "fewshot":
            draws = (0, 1) if idea.get("status") == "tagged" else (1,)
            snap["_fs"] = {d: self._fs_select(st, iid, d) for d in draws}
        return snap

    def _claim(self, limit: int) -> list[tuple[str, dict]]:
        with self.j.transaction(WORKER, "claim") as st:
            ids = J.claim(st, WORKER, "judge", _eligible, limit, time.time(), self.timeout_s)
            return [(i, self._snap(st, i)) for i in ids]

    def _claim_rank(self, limit: int) -> tuple[list[tuple[str, dict]], list[str]]:
        """-> (batch needing a ranking call, ids that only need finalizing). Waits for a full batch unless the
        oldest waiting idea has waited rank_max_wait_s."""
        now = time.time()
        with self.j.transaction(WORKER, "claim") as st:
            ready = [i for i in st["ideas"].values() if i.get("status") != "failed" and _eligible(i)
                     and J._claim_stale(i, now, self.timeout_s)]
            fin = [i["id"] for i in ready if not _needs_call(i)][:limit]
            calls = sorted((i for i in ready if _needs_call(i)), key=lambda i: i.get("created_at", 0.0))
            batch = []
            room = limit - len(fin)
            if calls and room > 0:
                since = min((i.get("times") or {}).get(i["status"], i.get("created_at", now)) for i in calls)
                if len(calls) >= self.rank_batch or now - since >= self.rank_wait:
                    batch = [i["id"] for i in calls[:min(self.rank_batch, room)]]
            for iid in fin + batch:
                st["ideas"][iid]["claim"] = {"worker": WORKER, "stage": "judge", "at": now}
            return [(i, self._snap(st, i)) for i in batch], fin

    def _new_work(self, room: int) -> list[tuple[object, list[str]]]:
        if self.strategy != "ranking":
            return [(self._process(iid, idea), [iid]) for iid, idea in self._claim(room)]
        out = []
        while room > 0:
            batch, fin = self._claim_rank(room)
            if not batch and not fin:
                break
            out += [(self._record_final_async(iid), [iid]) for iid in fin]
            if batch:
                out.append((self._process_batch(batch), [i for i, _ in batch]))
            room -= len(fin) + len(batch)
        return out

    async def _record_final_async(self, iid: str) -> None:
        self._record_final(iid, [], [], [])

    async def run(self) -> None:
        inflight: dict[asyncio.Task, int] = {}
        stop_wait = asyncio.ensure_future(self.stop.wait())
        try:
            while not self.stop.is_set():
                room = self.concurrency - sum(inflight.values())
                if room > 0:
                    for coro, ids in self._new_work(room):
                        inflight[asyncio.create_task(self._guarded(coro, ids))] = len(ids)
                waits = set(inflight) | {stop_wait}
                done, _ = await asyncio.wait(waits, timeout=self.poll_s,
                                             return_when=asyncio.FIRST_COMPLETED)
                for t in done:
                    if t is not stop_wait:
                        inflight.pop(t, None)
        finally:
            stop_wait.cancel()
            for t in inflight:
                t.cancel()
            if inflight:
                await asyncio.gather(*inflight, return_exceptions=True)
            # release claims of cancelled work so a resume picks them up immediately
            try:
                with self.j.transaction(WORKER, "release") as st:
                    for i in st["ideas"].values():
                        c = i.get("claim")
                        if c and c.get("worker") == WORKER and i.get("status") in ("tagged", "judged"):
                            i["claim"] = None
            except Exception:  # noqa: BLE001
                pass


async def run(journal: J.Journal, cfg: dict, llm, stop: asyncio.Event) -> None:
    await Judge(journal, cfg, llm, stop).run()
