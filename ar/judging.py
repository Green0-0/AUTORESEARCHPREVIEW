"""Judge-strategy helpers shared by the pipeline judge (ar/workers/judge.py) and the offline study
(analysis/judge_study.py). Plan §9.1; prompts in ar/prompts.py, parsers in ar/scoring.py.

Strategies: basic (J1, K=1), basic_k (J2, mean of K samples), rubric (J3), fewshot (J4), ranking (J5).

* Facts are judge-only: ``judge_facts(cfg)`` = ``judge.facts`` (null -> the top-level ``facts``, the legacy
  behaviour) + ``judge.facts_file`` + the task's anchor facts if ``judge.anchor_facts``. Ideators only ever see
  the top-level ``facts``.
* Few-shot examples (``select_examples``): candidates are filtered (the idea itself and near-duplicates of it
  are removed), sorted by score, cut into ``n`` contiguous equal-count bins (np.array_split) and one example is
  drawn uniformly from each bin; the chosen examples are shown in a random order. The rng is seeded per
  (seed, idea, draw), so a judgment is reproducible and its confirmation draw sees a different selection.
* Ranking batches for the offline study (``ranking_batches``): R repetitions; in each the ideas are shuffled
  and split into ceil(n / b) near-equal batches; for repetition r >= 1 the shuffle (out of 50 seeded
  candidates) that repeats the fewest co-batched pairs of earlier repetitions is used.
"""
from __future__ import annotations

import hashlib
import math
import re
from itertools import combinations

import numpy as np

from ar import anchors
from ar import config as arconfig
from ar import prompts

STRATEGIES = arconfig.JUDGE_STRATEGIES


def judge_facts(cfg: dict) -> list[str]:
    jc = cfg.get("judge") or {}
    base = cfg.get("facts") if jc.get("facts") is None else jc.get("facts")
    facts = list(base or [])
    extra = list(arconfig.load_facts_file(jc["facts_file"])) if jc.get("facts_file") else []
    if jc.get("anchor_facts"):
        extra += anchors.anchor_facts(cfg["task_id"])
    facts += [x for x in extra if x not in facts]
    return facts


def stable_int(*parts) -> int:
    """Deterministic 63-bit integer from arbitrary parts (for rng seeds)."""
    return int(hashlib.sha256(repr(parts).encode()).hexdigest()[:15], 16)


def truncate(text: str, max_chars: int | None) -> str:
    text = str(text or "")
    if not max_chars or len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n[... description truncated ...]"


_WORD = re.compile(r"[a-z0-9]+")


def _words(s: str) -> set[str]:
    return set(_WORD.findall(str(s or "").lower()))


def near_duplicate(a: str, b: str, threshold: float = 0.6) -> bool:
    """Word-set Jaccard of two summaries/descriptions >= threshold (or identical text)."""
    if not a or not b:
        return False
    if " ".join(str(a).split()) == " ".join(str(b).split()):
        return True
    wa, wb = _words(a), _words(b)
    if not wa or not wb:
        return False
    return len(wa & wb) / len(wa | wb) >= threshold


def select_examples(cands: list[dict], n: int, rng: np.random.Generator, *,
                    exclude_id=None, exclude_texts: tuple[str, ...] = ()) -> list[dict]:
    """Stratified few-shot selection (see module doc).

    cands = [{"id", "summary", "score", optional "sort"}]: ``score`` is what the prompt shows, ``sort`` (default
    score) orders the bins. Returns up to n candidates in presentation order."""
    pool = [c for c in cands if c.get("id") != exclude_id and c.get("summary")
            and not any(near_duplicate(c["summary"], t) for t in exclude_texts if t)]
    if not pool or n <= 0:
        return []
    pool.sort(key=lambda c: (float(c.get("sort", c["score"])), str(c["id"])))
    n = min(n, len(pool))
    bins = np.array_split(np.arange(len(pool)), n)
    chosen = [pool[int(b[rng.integers(len(b))])] for b in bins]
    order = rng.permutation(len(chosen))
    return [chosen[int(i)] for i in order]


def fewshot_examples(chosen: list[dict]) -> list[tuple[str, float]]:
    return [(c["summary"], float(c["score"])) for c in chosen]


def messages_single(strategy: str, system: str, description: str, examples: list[tuple[str, float]] | None = None
                    ) -> list[dict]:
    """Chat messages for one idea under J1-J4 (fewshot with no examples falls back to the basic prompt)."""
    if strategy in ("basic", "basic_k"):
        user = prompts.judge_prompt(description)
    elif strategy == "rubric":
        user = prompts.rubric_prompt(description)
    elif strategy == "fewshot":
        user = prompts.fewshot_prompt(description, examples) if examples else prompts.judge_prompt(description)
    else:
        raise ValueError(f"not a single-idea strategy: {strategy!r}")
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def messages_ranking(system: str, descriptions: list[str], references: list[dict],
                     max_chars: int | None) -> list[dict]:
    user = prompts.ranking_prompt([truncate(d, max_chars) for d in descriptions], references)
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def ranking_batches(keys: list, batch_size: int, R: int, seed: int, candidates: int = 50) -> list[list[list]]:
    """-> [rep][batch] = list of keys. Each key appears once per repetition (R times overall)."""
    keys = list(keys)
    if not keys:
        return [[] for _ in range(R)]
    nb = max(1, math.ceil(len(keys) / batch_size))
    seen: set[frozenset] = set()
    reps = []
    for r in range(R):
        best, best_cost = None, None
        for c in range(1 if r == 0 else candidates):
            rng = np.random.default_rng(stable_int(seed, "rank", r, c))
            perm = rng.permutation(len(keys))
            batches = [[keys[int(i)] for i in b] for b in np.array_split(perm, nb)]
            cost = sum(1 for b in batches for p in combinations(b, 2) if frozenset(p) in seen)
            if best_cost is None or cost < best_cost:
                best, best_cost = batches, cost
            if best_cost == 0:
                break
        for b in best:
            seen.update(frozenset(p) for p in combinations(b, 2))
        reps.append(best)
    return reps
