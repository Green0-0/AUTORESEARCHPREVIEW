"""Tagger worker (spec "Tagging"; plan §2 tagging notes).

One tagging cycle:
  1. trigger check on a lock-free snapshot (count of summarized ideas >= trigger_count, or
     the oldest waiting idea has waited >= trigger_minutes), then claim up to batch_limit
     under a transaction;
  2. creation, outside the lock: partition -> Generator (parallel) -> Merger rounds
     (N lists at a time until <= K remain) -> Writer (last ```json block, one retry);
  3. pool update in a transaction (names already in the pool are dropped);
  4. assignment, outside the lock: one call per idea (parallel) against the updated pool,
     one retry if no valid tag;
  5. results + the cycle log entry written in a final transaction.

If the writer fails twice (or any creation call raises), the whole batch is released
untagged and the failure is logged; the ideas are picked up again in the next cycle.
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import re
import time
from typing import Any

import numpy as np

from ar import journal as J
from ar import prompts

log = logging.getLogger(__name__)

WORKER = "tagger"
STAGE = "tagger"
POLL_SECONDS = 5.0  # overridable via cfg["tagging"]["poll_seconds"]

_now = time.time  # monkeypatched in tests


# --------------------------------------------------------------------------- helpers

def _tcfg(cfg: dict) -> dict:
    d = {"trigger_count": 20, "trigger_minutes": 10, "batch_limit": 100,
         "S": 10, "r": 2, "N": 4, "K": 4}
    d.update(cfg.get("tagging") or {})
    return d


def _task_prompt(cfg: dict) -> str:
    from ar import config as C
    return C.task_prompt(cfg)


def _pool(state: dict) -> list[tuple[str, str]]:
    return [(t["name"], t.get("description", "")) for t in state.get("tags", {}).values()]


def _msgs(system: str, user: str) -> list[dict]:
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _waiting(state: dict, now: float, timeout_s: float) -> list[dict]:
    """Summarized ideas that are claimable (unclaimed or stale claim)."""
    out = []
    for idea in state.get("ideas", {}).values():
        if idea.get("status") != "summarized":
            continue
        c = idea.get("claim")
        if c and now - float(c.get("at", 0)) < timeout_s:
            continue
        out.append(idea)
    return out


def _waited_since(idea: dict) -> float:
    return float((idea.get("times") or {}).get("summarized", idea.get("created_at", 0.0)))


def should_trigger(state: dict, cfg: dict, now: float) -> bool:
    t = _tcfg(cfg)
    timeout_s = float((cfg.get("claims") or {}).get("timeout_minutes", 30)) * 60.0
    w = _waiting(state, now, timeout_s)
    if not w:
        return False
    if len(w) >= int(t["trigger_count"]):
        return True
    oldest = min(_waited_since(i) for i in w)
    return now - oldest >= float(t["trigger_minutes"]) * 60.0


# --------------------------------------------------------------------------- partition

def partition(ids: list[str], S: int, r: int, rng: np.random.Generator) -> list[list[str]]:
    """Split ids into subsets of at most S, each id in exactly r distinct subsets.

    Number of subsets m = max(r, ceil(n*r/S)); sizes are balanced. Deterministic given rng.
    """
    n = len(ids)
    if n == 0:
        return []
    S, r = int(S), int(r)
    if S < 1 or r < 1:
        raise ValueError("S and r must be >= 1")
    m = max(r, math.ceil(n * r / S))
    # Balanced capacities (differ by at most one), all <= S.
    base, extra = divmod(n * r, m)
    caps = [base + (1 if j < extra else 0) for j in range(m)]

    for _ in range(50):  # randomized greedy; well-mixed subsets
        subsets: list[list[str]] = [[] for _ in range(m)]
        members: list[set[str]] = [set() for _ in range(m)]
        ok = True
        for _c in range(r):
            for k in rng.permutation(n):
                x = ids[int(k)]
                cands = [j for j in range(m) if x not in members[j] and len(subsets[j]) < caps[j]]
                if not cands:
                    ok = False
                    break
                slack = max(caps[j] - len(subsets[j]) for j in cands)
                best = [j for j in cands if caps[j] - len(subsets[j]) == slack]
                j = best[int(rng.integers(len(best)))]
                subsets[j].append(x)
                members[j].add(x)
            if not ok:
                break
        if ok:
            return subsets

    # Guaranteed construction: r consecutive copies of each id dealt round-robin over m >= r
    # subsets land in r distinct subsets; sizes are balanced.
    order = [ids[int(k)] for k in rng.permutation(n)]
    subsets = [[] for _ in range(m)]
    seq = [x for x in order for _ in range(r)]
    for p, x in enumerate(seq):
        subsets[p % m].append(x)
    return subsets


# --------------------------------------------------------------------------- parsing

_JSON_BLOCK = re.compile(r"```json[^\n]*\n?(.*?)```", re.DOTALL | re.IGNORECASE)


def parse_writer(text: str) -> list[dict] | None:
    """Parse the LAST ```json fenced block -> [{"name","description"}], or None if malformed."""
    blocks = _JSON_BLOCK.findall(text or "")
    if not blocks:
        # tolerate an unterminated final block (e.g. truncated closing fence)
        i = (text or "").lower().rfind("```json")
        if i < 0:
            return None
        blocks = [text[i + len("```json"):]]
    try:
        obj = json.loads(blocks[-1].strip())
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(obj, dict) or not isinstance(obj.get("new_tags"), list):
        return None
    out = []
    for item in obj["new_tags"]:
        if not isinstance(item, dict):
            return None
        name = item.get("name")
        desc = item.get("description", "")
        if not isinstance(name, str) or not name.strip():
            return None
        if not isinstance(desc, str):
            desc = str(desc)
        out.append({"name": name.strip(), "description": desc.strip()})
    return out


def normalize(name: str) -> str:
    return re.sub(r"[^0-9a-z]+", "", name.lower())


_BULLET = re.compile(r"^\s*(?:[-*+•‣◦>]+|\(?\d+[.)]|\(?[a-zA-Z][.)](?=\s))\s*")
_STRIP_CHARS = " \t`'\"*_“”‘’[]"


def _clean(s: str) -> str:
    s = s.strip()
    prev = None
    while prev != s:
        prev = s
        s = _BULLET.sub("", s, count=1).strip()
        s = s.strip(_STRIP_CHARS).strip()
        s = re.sub(r"[.,;]+$", "", s).strip()
    return s


def _candidates(line: str) -> list[str]:
    """Candidate tag names in one output line, most specific first."""
    raw = line.strip()
    if not raw:
        return []
    cands = [_clean(raw)]
    # "name: description", "name - description", "name — description"
    for sep in (":", " - ", " — ", " – "):
        if sep in raw:
            cands.append(_clean(raw.split(sep, 1)[0]))
    # backticked / quoted / bold names anywhere in the line
    for m in re.finditer(r"`([^`]+)`|\*\*([^*]+)\*\*|\"([^\"]+)\"", raw):
        cands.append(_clean(next(g for g in m.groups() if g)))
    return [c for c in dict.fromkeys(cands) if c]


def parse_tag_names(text: str, pool_names: list[str]) -> tuple[list[str], int, list[str]]:
    """Match names in an assignment reply against the pool.

    Returns (matched names in reply order, number matched only after normalization,
    unmatched candidate strings). Exact match first, then normalized (case/whitespace/
    punctuation-insensitive) match; ambiguous normalized forms are not matched.
    """
    exact = set(pool_names)
    norm: dict[str, str | None] = {}
    for n in pool_names:
        k = normalize(n)
        norm[k] = None if (k in norm and norm[k] != n) else n

    matched: list[str] = []
    n_norm = 0
    unmatched: list[str] = []
    for line in (text or "").splitlines():
        cands = _candidates(line)
        if not cands:
            continue
        hits = []
        h, vn = _match(cands, exact, norm)
        if h is not None:
            hits.append((h, vn))
        elif "," in line:
            # comma-separated list on one line ("a, b, c" or "Tags: a, b, c")
            body = line.split(":", 1)[1] if ":" in line else line
            for part in body.split(","):
                h2, vn2 = _match(_candidates(part), exact, norm)
                if h2 is not None:
                    hits.append((h2, vn2))
        if not hits:
            unmatched.append(cands[0])
        for h, vn in hits:
            if h not in matched:
                matched.append(h)
                n_norm += int(vn)
    return matched, n_norm, unmatched


def _match(cands, exact, norm) -> tuple[str | None, bool]:
    for c in cands:
        if c in exact:
            return c, False
    for c in cands:
        h = norm.get(normalize(c))
        if h:
            return h, True
    return None, False


# --------------------------------------------------------------------------- creation

class CreationFailed(Exception):
    pass


# ---- context-budget guards (added 2026-11-11: in the 24 h test runs the pool grew to 194-331 tags and every
# creation call after cycle ~10-26 overflowed the 32k context, so no idea was tagged/judged afterwards).
POOL_DESC_WORDS = 15          # tag descriptions shown to the LLM are cut to this many words
MAX_CONTEXT_SUMMARIES = 40    # merger/writer prompts show at most this many idea summaries (creation order)
SUMMARY_CHARS = 600           # each summary shown is cut to this many characters
PROMPT_TOKEN_BUDGET = 24000   # estimated system+user tokens per call (32k context minus ~8k for output)
CHARS_PER_TOKEN = 3.3


def compact_pool(pool: list[tuple[str, str]]) -> list[tuple[str, str]]:
    out = []
    for name, desc in pool:
        w = (desc or "").split()
        out.append((name, " ".join(w[:POOL_DESC_WORDS]) + (" ..." if len(w) > POOL_DESC_WORDS else "")))
    return out


def _est_tokens(*texts: str) -> int:
    return int(sum(len(t) for t in texts) / CHARS_PER_TOKEN)


async def _create(llm, system: str, ideas: dict[str, dict], subsets: list[list[str]],
                  N: int, K: int, stats: dict) -> tuple[list[dict], list[str]]:
    """Returns (parsed new tags, raw writer outputs)."""
    summ = {i: (ideas[i].get("summary") or ideas[i].get("description") or "")[:SUMMARY_CHARS] for i in ideas}

    def summaries(cover, cap: int = MAX_CONTEXT_SUMMARIES) -> list[str]:
        return [summ[i] for i in ideas if i in cover][:cap]  # creation order, capped

    def fit(build, cover) -> str:
        """Largest prompt (fewest summaries dropped) that fits the token budget with the system prompt."""
        for cap in (MAX_CONTEXT_SUMMARIES, 20, 10, 5, 0):
            u = build(summaries(cover, cap))
            if _est_tokens(system, u) <= PROMPT_TOKEN_BUDGET:
                return u
        return u

    async def call(user: str) -> str:
        stats["calls"] += 1
        res = await llm.chat(_msgs(system, user), "tagging")
        return res.text

    # Generate
    gen = await asyncio.gather(*[call(prompts.generator_prompt([summ[i] for i in s])) for s in subsets])
    stats["generator_calls"] = len(subsets)
    lists: list[tuple[str, set[str]]] = [(t, set(s)) for t, s in zip(gen, subsets)]

    # Merge rounds
    N = max(2, int(N))
    K = max(1, int(K))
    rounds = 0
    merge_calls = 0
    while len(lists) > K:
        groups = [lists[p:p + N] for p in range(0, len(lists), N)]
        todo = [(gi, g) for gi, g in enumerate(groups) if len(g) > 1]
        outs = await asyncio.gather(*[
            call(fit(lambda ss, g=g: prompts.merger_prompt(ss, [t for t, _ in g]), set().union(*[c for _, c in g])))
            for _, g in todo])
        merge_calls += len(todo)
        merged = {gi: (o, set().union(*[c for _, c in groups[gi]])) for (gi, _), o in zip(todo, outs)}
        lists = [merged[gi] if gi in merged else groups[gi][0] for gi in range(len(groups))]
        rounds += 1
    stats["merge_rounds"] = rounds
    stats["merge_calls"] = merge_calls
    stats["lists_to_writer"] = len(lists)

    # Write (one retry on malformed JSON)
    cover = set().union(*[c for _, c in lists]) if lists else set()
    wp = fit(lambda ss: prompts.writer_prompt(ss, [t for t, _ in lists]), cover)
    raws = []
    for attempt in range(2):
        txt = await call(wp)
        raws.append(txt)
        stats["writer_calls"] = attempt + 1
        parsed = parse_writer(txt)
        if parsed is not None:
            return parsed, raws
    raise CreationFailed("writer returned malformed JSON twice")


# --------------------------------------------------------------------------- assignment

async def _assign(llm, system: str, idea: dict, pool_names: list[str]) -> dict:
    outs = []
    n_norm = 0
    unmatched: list[str] = []
    errors: list[str] = []
    for _attempt in range(2):
        try:
            res = await llm.chat(_msgs(system, prompts.assign_prompt(idea.get("description") or "")), "assign")
        except Exception as e:  # counts as a failed attempt
            errors.append(f"{type(e).__name__}: {e}")
            continue
        outs.append(res.text)
        names, nn, um = parse_tag_names(res.text, pool_names)
        unmatched.extend(um)
        if names:
            return {"tags": names, "normalized": nn, "unmatched": unmatched,
                    "outputs": outs, "calls": len(outs) + len(errors), "error": None}
    return {"tags": [], "normalized": 0, "unmatched": unmatched, "outputs": outs,
            "calls": len(outs) + len(errors),
            "error": "assignment: no valid tag after retry" + (f" ({errors[-1]})" if errors else "")}


# --------------------------------------------------------------------------- cycle

async def tagging_cycle(journal, cfg: dict, llm) -> dict | None:
    """Run one cycle if the trigger fires. Returns the log entry, or None if not triggered."""
    t = _tcfg(cfg)
    timeout_s = float((cfg.get("claims") or {}).get("timeout_minutes", 30)) * 60.0
    now = _now()
    if not should_trigger(journal.read(), cfg, now):
        return None

    with journal.transaction(WORKER, "claim") as state:
        ids = J.claim(state, WORKER, STAGE, lambda i: i.get("status") == "summarized",
                      int(t["batch_limit"]), now, timeout_s)
        cycle = len(state["logs"]["tagging"])
        seed = int(state.get("config", cfg).get("seed", cfg.get("seed", 0)) or 0)
        ideas = {i: dict(state["ideas"][i]) for i in ids}
        pool_before = _pool(state)
    if not ids:
        return None

    t0 = _now()
    entry: dict[str, Any] = {
        "t": t0, "cycle": cycle, "ok": False, "n_ideas": len(ids), "ids": ids,
        "pool_size_before": len(pool_before), "n_subsets": 0, "calls": {},
        "new_tags": [], "dropped_existing": [], "pool_size_after": len(pool_before),
        "failures": [], "normalized_matches": 0, "unmatched_names": 0, "error": None,
    }
    task_prompt = _task_prompt(cfg)
    rng = np.random.default_rng([seed, cycle])
    subsets = partition(ids, t["S"], t["r"], rng)
    entry["n_subsets"] = len(subsets)
    entry["subsets"] = subsets
    stats = {"calls": 0}

    # ---- creation (no lock): pool shown is the pre-creation pool
    try:
        new_tags, writer_raw = await _create(llm, prompts.tagging_system(task_prompt, compact_pool(pool_before)),
                                             ideas, subsets, t["N"], t["K"], stats)
    except Exception as e:
        # Do not strand the batch (that is what stalled the 24 h test runs): skip creation for this cycle and
        # assign against the existing pool. Only an empty pool still releases the ideas for a later cycle.
        err = f"creation failed: {type(e).__name__}: {e}"
        entry["creation_error"] = err[:2000]
        if not pool_before:
            log.warning("tagging cycle %d: %s; empty pool, releasing %d ideas", cycle, err, len(ids))
            entry.update(error=err, calls={k: v for k, v in stats.items()}, duration_s=_now() - t0)
            with journal.transaction(WORKER, "release") as state:
                for i in ids:
                    idea = state["ideas"].get(i)
                    if idea is None:
                        continue
                    c = idea.get("claim") or {}
                    if c.get("worker") == WORKER and idea.get("status") == "summarized":
                        J.release(state, i)
                        r = idea.setdefault("retries", {})
                        r[STAGE] = r.get(STAGE, 0) + 1
                state["logs"]["tagging"].append(entry)
            return entry
        log.warning("tagging cycle %d: %s; assigning %d ideas against the existing pool", cycle, err, len(ids))
        new_tags, writer_raw = [], []

    # ---- pool update
    with journal.transaction(WORKER, "add_tags") as state:
        tags = state["tags"]
        existing_norm = {normalize(n) for n in tags}
        added, dropped = [], []
        ts = _now()
        for tg in new_tags:
            nm = tg["name"]
            if nm in tags or normalize(nm) in existing_norm:
                dropped.append(nm)
                continue
            tags[nm] = {"name": nm, "description": tg["description"], "created_at": ts, "ideas": []}
            existing_norm.add(normalize(nm))
            added.append(nm)
        pool_after = _pool(state)
    entry["new_tags"] = added
    entry["dropped_existing"] = dropped
    entry["writer_outputs"] = writer_raw
    entry["pool_size_after"] = len(pool_after)

    # ---- assignment (no lock): full updated pool
    sys_assign = prompts.tagging_system(task_prompt, compact_pool(pool_after))
    pool_names = [n for n, _ in pool_after]
    results = await asyncio.gather(*[_assign(llm, sys_assign, ideas[i], pool_names) for i in ids])
    assign_calls = sum(r["calls"] for r in results)

    # ---- write results
    with journal.transaction(WORKER, "assign") as state:
        ts = _now()
        written = []
        for i, res in zip(ids, results):
            idea = state["ideas"].get(i)
            if idea is None:
                continue
            c = idea.get("claim") or {}
            if idea.get("status") != "summarized" or c.get("worker") != WORKER:
                continue  # claim was reaped / taken over meanwhile; leave it alone
            idea["assign_outputs"] = res["outputs"]
            if res["error"]:
                J.fail(state, i, res["error"], ts)
                entry["failures"].append(i)
                continue
            idea["tags"] = list(res["tags"])
            for nm in res["tags"]:
                lst = state["tags"][nm].setdefault("ideas", [])
                if i not in lst:
                    lst.append(i)
            J.set_status(state, i, "tagged", ts)
            written.append(i)
        entry.update(
            ok=True, tagged=len(written),
            normalized_matches=sum(r["normalized"] for r in results),
            unmatched_names=sum(len(r["unmatched"]) for r in results),
            pool_size_after=len(state["tags"]),
            calls={"generator": stats.get("generator_calls", 0),
                   "merger": stats.get("merge_calls", 0),
                   "writer": stats.get("writer_calls", 0),
                   "assign": assign_calls,
                   "total": stats["calls"] + assign_calls},
            merge_rounds=stats.get("merge_rounds", 0),
            lists_to_writer=stats.get("lists_to_writer", 0),
            duration_s=_now() - t0,
        )
        state["logs"]["tagging"].append(entry)
    log.info("tagging cycle %d: %d ideas, %d subsets, +%d tags (pool %d), %d failed, %d normalized",
             cycle, len(ids), len(subsets), len(added), entry["pool_size_after"],
             len(entry["failures"]), entry["normalized_matches"])
    return entry


async def run(journal, cfg: dict, llm, stop: asyncio.Event) -> None:
    poll = float(_tcfg(cfg).get("poll_seconds", POLL_SECONDS))
    while not stop.is_set():
        entry = None
        try:
            entry = await tagging_cycle(journal, cfg, llm)
        except Exception:  # keep the worker alive; claims time out and get reaped
            log.exception("tagging cycle crashed")
        if entry is not None and entry.get("ok"):
            continue  # more work may be waiting; re-check immediately
        try:
            await asyncio.wait_for(stop.wait(), timeout=poll)
        except asyncio.TimeoutError:
            pass
