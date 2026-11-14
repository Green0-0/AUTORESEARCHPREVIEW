"""Ideator worker (spec "Ideation", "Summarizing"; CONTRACTS.md "Workers").

Loop: keep ``ideation.concurrent_batches`` batches in flight. Each batch request is one
journal transaction that refits the posterior from the scored ideas (logged), computes
availability, draws B configs (logged), allocates idea ids + a wave number and snapshots the
RenderContext. Then, outside the lock, every idea of the batch is rendered with its own RNG
(seeded from (run seed, idea number)), run as a multi-turn chat under the ideator system
prompt, written as ``generated``, summarized (no system prompt) and written as ``summarized``.

Failures: each LLM step is retried once (on top of the client's transient retries); then the
idea is recorded as ``failed`` with its partial prompt record. A final answer that is empty
(e.g. truncated inside <think>) counts as a failure. ``finish_reason == "length"`` is counted
per idea (``prompt.finish_reasons``) and in ``counters.length_finish[purpose]``.

On start, ideas left in ``generated`` by a previous process (crash / stop) are summarized.
On stop, in-flight batches are cancelled; their unwritten ideas are dropped (ids are not reused).
"""
from __future__ import annotations

import asyncio
import logging
import time

import numpy as np

from ar import config as arconfig
from ar import journal as J
from ar import prompts, sampler
from ar.templating import RenderContext

log = logging.getLogger("ar.ideator")
WORKER = "ideator"


BACKLOG_STATUSES = ("generated", "summarized", "tagged", "judged")
BACKLOG_POLL_S = 15.0


def idea_number(idea_id: str) -> int:
    return int(idea_id.lstrip("i"))


def idea_rng(seed: int, idea_id: str) -> np.random.Generator:
    """Per-idea RNG for template draws: seeded from (run seed, idea number)."""
    return np.random.default_rng([int(seed), idea_number(idea_id)])


def _judged_ideas(state: dict) -> list[dict]:
    """Ideas that have a surrogate score (s_cal), not failed, creation order."""
    out = []
    for i in state["ideas"].values():
        if i.get("status") == "failed" or i.get("s_cal") is None or not i.get("summary"):
            continue
        out.append({"id": i["id"], "summary": i["summary"], "s_cal": i["s_cal"],
                    "real_score": i.get("real_score")})
    return out


def snapshot_context(state: dict, cfg: dict, task_prompt: str) -> RenderContext:
    return RenderContext(
        prompt=task_prompt,
        references=list(cfg.get("references") or []),
        facts=list(cfg.get("facts") or []),
        tags=[(t["name"], t.get("description", "")) for t in state.get("tags", {}).values()],
        judged_ideas=_judged_ideas(state),
    )


def _count_length(state: dict, purpose: str, n: int = 1) -> None:
    if n:
        d = state["counters"].setdefault("length_finish", {})
        d[purpose] = int(d.get(purpose, 0)) + n


def request_batch(state: dict, cfg: dict, task_prompt: str, now: float) -> dict | None:
    """Inside a transaction: refit, draw, allocate ids + wave, snapshot context.
    Returns None when stop.max_ideas is reached."""
    B = int(cfg["ideation"]["batch_size"])
    max_ideas = (cfg.get("stop") or {}).get("max_ideas")
    if max_ideas is not None:
        B = min(B, int(max_ideas) - int(state["counters"].get("next_idea", 0)))
        if B <= 0:
            return None
    sc = cfg["sampler"]
    X, y = sampler.observations(state)
    mu, Sigma = sampler.fit_posterior(X, y, sc["lambda0"], sc["lambda"], sc["sigma"], sc["gamma"])
    wave = int(state["counters"].get("wave", 0))
    state["counters"]["wave"] = wave + 1
    refit = {"t": now, "wave": wave, **sampler.refit_log(mu, Sigma, X.shape[0])}
    state["logs"]["refits"].append(refit)
    avail = sampler.availability(state)
    rng = np.random.default_rng([int(cfg["seed"]), wave, 7919])
    draws = sampler.draw_batch(mu, Sigma, avail, B, rng, mode=sc.get("mode", "thompson"))
    ids = [J.new_idea_id(state) for _ in draws]
    state["logs"]["batches"].append({
        "t": now, "wave": wave, "n_obs": int(X.shape[0]), "availability": avail,
        "draws": [{"idea_id": iid, "b_tilde": b, "config": c} for iid, (c, b) in zip(ids, draws)],
    })
    return {"wave": wave, "ids": ids, "draws": draws,
            "ctx": snapshot_context(state, cfg, task_prompt)}


class Ideator:
    def __init__(self, journal: J.Journal, cfg: dict, llm, stop: asyncio.Event):
        self._bp_logged = False
        self.j, self.cfg, self.llm, self.stop = journal, cfg, llm, stop
        self.task_prompt = arconfig.task_prompt(cfg)
        self.seed = int(cfg["seed"])
        self.arm = cfg.get("arm", "optimized")
        self.stats = {"batches": 0, "generated": 0, "summarized": 0, "failed": 0,
                      "length": {"ideation": 0, "summary": 0}}

    # ------------------------------------------------------------- llm steps
    async def _call(self, messages: list[dict], purpose: str):
        """One LLM step with one retry. Empty answers count as failures.
        -> (result | None, error strings, results of failed attempts)."""
        errors: list[str] = []
        bad = []
        for attempt in range(2):
            try:
                r = await self.llm.chat(messages, purpose)
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001
                errors.append(f"{purpose} attempt {attempt + 1}: {e!r}"[:500])
                continue
            if r.finish_reason == "length":
                self.stats["length"][purpose] = self.stats["length"].get(purpose, 0) + 1
            if r.text.strip():
                return r, errors, bad
            errors.append(f"{purpose} attempt {attempt + 1}: empty answer "
                          f"(finish_reason={r.finish_reason})")
            bad.append(r)   # keep the failed raw output for the record
        return None, errors, bad

    # --------------------------------------------------------------- one idea
    async def _idea(self, iid: str, wave: int, scfg: dict, b_tilde, ctx: RenderContext) -> None:
        rec_prompt = {"config": scfg, "b_tilde": b_tilde, "system": None, "turns": [],
                      "draws": [], "raw_outputs": [], "finish_reasons": []}
        try:
            asm = prompts.assemble(scfg, ctx, idea_rng(self.seed, iid))
        except Exception as e:  # noqa: BLE001  (render bug: record and move on)
            self._write_failed(iid, wave, rec_prompt, f"render: {e!r}")
            return
        rec_prompt.update(system=asm["system"], turns=list(asm["turns"]), draws=list(asm["draws"]))
        messages = [{"role": "system", "content": asm["system"]}]
        result = None
        n_len = 0
        for k, turn in enumerate(asm["turns"]):
            messages.append({"role": "user", "content": turn})
            result, errors, bad = await self._call(messages, "ideation")
            n_len += sum(b.finish_reason == "length" for b in bad)
            if bad:  # failed attempts are kept separately so raw_outputs stays one per turn
                rec_prompt.setdefault("failed_raw_outputs", []).extend(
                    {"turn": k + 1, "raw": b.raw, "finish_reason": b.finish_reason} for b in bad)
            if errors:
                rec_prompt.setdefault("retried_turns", []).append(k + 1)
            if result is None:
                msg = "; ".join(errors)
                self._write_failed(iid, wave, rec_prompt, f"ideation turn {k + 1}: {msg}",
                                   n_len=n_len)
                return
            rec_prompt["raw_outputs"].append(result.raw)
            rec_prompt["finish_reasons"].append(result.finish_reason)
            n_len += result.finish_reason == "length"
            messages.append({"role": "assistant", "content": result.text})
        description = result.text

        now = time.time()
        with self.j.transaction(WORKER, "generated") as st:
            st["ideas"][iid] = self._record(iid, wave, rec_prompt, description, now)
            J.set_status(st, iid, "generated", now)
            _count_length(st, "ideation", n_len)
        self.stats["generated"] += 1
        await self._summarize(iid, description)

    async def _summarize(self, iid: str, description: str) -> None:
        r, errors, _ = await self._call([{"role": "user", "content": prompts.summary_prompt(description)}],
                                     "summary")
        now = time.time()
        with self.j.transaction(WORKER, "summarized" if r else "fail_summary") as st:
            idea = st["ideas"].get(iid)
            if idea is None or idea.get("status") != "generated":
                return
            if r is None:
                msg = "; ".join(errors)
                idea.setdefault("retries", {})["summary"] = 1
                J.fail(st, iid, f"summary: {msg}", now)
                self.stats["failed"] += 1
                return
            idea["summary"] = r.text
            idea["summary_raw"] = r.raw
            idea["prompt"].setdefault("summary_finish_reason", r.finish_reason)
            if r.finish_reason == "length":
                _count_length(st, "summary")
            if errors:
                idea.setdefault("retries", {})["summary"] = 1
            J.set_status(st, iid, "summarized", now)
        self.stats["summarized"] += 1

    def _record(self, iid, wave, rec_prompt, description, now) -> dict:
        return {
            "id": iid, "created_at": now, "wave": wave, "arm": self.arm,
            "prompt": rec_prompt, "description": description, "summary": None, "tags": [],
            "surrogate_scores": [], "judge_outputs": [], "s_cal": None,
            "F": None, "F_detail": None, "scored_seq": None,
            "real_score": None, "real": None,
            "status": "generated", "times": {}, "claim": None, "error": None, "retries": {},
        }

    def _write_failed(self, iid, wave, rec_prompt, error, n_len=0) -> None:
        now = time.time()
        with self.j.transaction(WORKER, "fail_ideation") as st:
            st["ideas"][iid] = self._record(iid, wave, rec_prompt, "", now)
            st["ideas"][iid]["retries"] = {"ideation": 1}
            J.fail(st, iid, error, now)
            _count_length(st, "ideation", n_len)
        self.stats["failed"] += 1
        log.warning("idea %s failed: %s", iid, error[:200])

    # ------------------------------------------------------------------ batch
    async def _batch(self, req: dict) -> None:
        await asyncio.gather(*[self._idea(iid, req["wave"], c, b, req["ctx"])
                               for iid, (c, b) in zip(req["ids"], req["draws"])])
        self.stats["batches"] += 1

    def _backlogged(self) -> bool:
        """Backpressure (added 2026-11-11): pause ideation while more than ideation.max_backlog ideas are
        waiting for summary/tagging/judging, so F feedback to the sampler stays current. None = off."""
        mb = self.cfg["ideation"].get("max_backlog")
        if mb is None:
            return False
        st = self.j.read()
        n = sum(1 for i in st["ideas"].values() if i.get("status") in BACKLOG_STATUSES)
        if n >= int(mb):
            if not self._bp_logged:
                log.info("ideator: backpressure, %d unscored ideas >= max_backlog %s", n, mb)
                self._bp_logged = True
            return True
        self._bp_logged = False
        return False

    def _request(self) -> dict | None:
        now = time.time()
        with self.j.transaction(WORKER, "request_batch") as st:
            return request_batch(st, self.cfg, self.task_prompt, now)

    async def _resume_generated(self) -> None:
        """Summarize ideas left in `generated` (previous process stopped between the steps)."""
        timeout_s = float(self.cfg["claims"]["timeout_minutes"]) * 60
        with self.j.transaction(WORKER, "claim_summary") as st:
            ids = J.claim(st, WORKER, "summary", lambda i: i.get("status") == "generated",
                          10**9, time.time(), timeout_s)
            descs = {i: st["ideas"][i]["description"] for i in ids}
        if ids:
            await asyncio.gather(*[self._summarize(i, d) for i, d in descs.items()])

    async def run(self) -> None:
        await self._resume_generated()
        inflight: set[asyncio.Task] = set()
        exhausted = False
        stop_wait = asyncio.ensure_future(self.stop.wait())
        try:
            while not self.stop.is_set():
                while not exhausted and len(inflight) < int(self.cfg["ideation"]["concurrent_batches"]):
                    if self._backlogged():
                        break
                    req = self._request()
                    if req is None:
                        exhausted = True
                        break
                    inflight.add(asyncio.create_task(self._batch(req)))
                if not inflight and not exhausted:
                    # backpressure: too many unscored ideas; wait for tagger/judge to catch up
                    try:
                        await asyncio.wait_for(self.stop.wait(), timeout=BACKLOG_POLL_S)
                    except asyncio.TimeoutError:
                        pass
                    continue
                if not inflight:
                    # max_ideas reached and everything written: idle until stopped (returning
                    # early would make a supervisor restart us in a loop)
                    log.info("ideator: stop.max_ideas reached; idle until stop")
                    await self.stop.wait()
                    break
                done, _ = await asyncio.wait(inflight | {stop_wait},
                                             return_when=asyncio.FIRST_COMPLETED)
                for t in done:
                    if t is stop_wait:
                        continue
                    inflight.discard(t)
                    if not t.cancelled() and t.exception() is not None:
                        log.error("ideator batch crashed: %r", t.exception())
        finally:
            stop_wait.cancel()
            for t in inflight:
                t.cancel()
            if inflight:
                await asyncio.gather(*inflight, return_exceptions=True)


async def run(journal: J.Journal, cfg: dict, llm, stop: asyncio.Event) -> None:
    await Ideator(journal, cfg, llm, stop).run()
