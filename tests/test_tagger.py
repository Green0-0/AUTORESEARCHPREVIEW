"""Tests for ar/workers/tagger.py with a fake LLM (offline, fast)."""
from __future__ import annotations

import asyncio
import json
import re
import sys
import types
from collections import Counter
from dataclasses import dataclass

import numpy as np
import pytest

# ar.prompts is written by another agent; fall back to a minimal stand-in that uses the
# spec's verbatim marker phrases so the fake LLM below can recognise each call type.
try:
    from ar import prompts as _real_prompts  # noqa: F401
except Exception:  # pragma: no cover - only before prompts.py lands
    _fp = types.ModuleType("ar.prompts")

    def _lines(xs):
        return "\n".join(f"- {x}" for x in xs)

    _fp.tagging_system = lambda prompt, tags: (
        "Your role is to act as a tagging agent ...\n" + prompt + "\n"
        + "\n".join(f"- {n}: {d}" for n, d in tags))
    _fp.generator_prompt = lambda s: (
        f"Here is a new list of ideas that are being considered:\n{_lines(s)}\n\n"
        "Output all the tags missing from the current list")
    _fp.merger_prompt = lambda s, ls: (
        f"Here is a new list of ideas that are being considered:\n{_lines(s)}\n\n"
        f"Here are {len(ls)} proposed lists of tags\n" + "\n\n".join(ls)
        + "\nDeduplicate, merge and refine")
    _fp.writer_prompt = lambda s, ls: (
        f"Here is a new list of ideas that are being considered:\n{_lines(s)}\n\n"
        "Here is a list of proposed tags: " + "\n\n".join(ls) + "\nOutput a json")
    _fp.assign_prompt = lambda idea: (
        f"Here is a new idea being considered: {idea}\n\n"
        "Output the list of tags by name, exact, from the source list")
    sys.modules["ar.prompts"] = _fp
    import ar
    ar.prompts = _fp

from ar import journal as J  # noqa: E402
from ar.workers import tagger  # noqa: E402


# ----------------------------------------------------------------------------- fakes

@dataclass
class R:
    text: str
    reasoning: str = ""
    finish_reason: str = "stop"
    prompt_tokens: int = 0
    completion_tokens: int = 0


def kind(messages, purpose):
    u = messages[-1]["content"]
    if purpose == "assign":
        return "assign"
    if "Output a json" in u:
        return "writer"
    if "Deduplicate, merge and refine" in u:
        return "merger"
    if "Output all the tags missing" in u:
        return "generator"
    raise AssertionError(f"unknown call: {u[:80]}")


class FakeLLM:
    """writer: callable(n) -> text; assign: callable(description, system, attempt) -> text."""

    def __init__(self, writer=None, assign=None):
        self.calls = []
        self.inflight = 0
        self.max_inflight = 0
        self.writer = writer or (lambda n: '```json\n{"new_tags":[{"name":"alpha","description":"A"},'
                                           '{"name":"beta","description":"B"}]}\n```')
        self.assign = assign or (lambda desc, system, attempt: "- alpha\n- beta")
        self.assign_attempts = Counter()

    async def chat(self, messages, purpose):
        k = kind(messages, purpose)
        self.calls.append((k, messages))
        self.inflight += 1
        self.max_inflight = max(self.max_inflight, self.inflight)
        await asyncio.sleep(0.001)
        self.inflight -= 1
        if k == "generator":
            return R("- alpha: A\n- beta: B")
        if k == "merger":
            return R("- alpha: A merged")
        if k == "writer":
            n = sum(1 for c, _ in self.calls if c == "writer")
            return R(self.writer(n))
        desc = messages[-1]["content"]
        self.assign_attempts[desc] += 1
        return R(self.assign(desc, messages[0]["content"], self.assign_attempts[desc]))

    def count(self, k):
        return sum(1 for c, _ in self.calls if c == k)


def make_cfg(**tagging):
    t = {"trigger_count": 20, "trigger_minutes": 10, "batch_limit": 100, "S": 10, "r": 2, "N": 4, "K": 4}
    t.update(tagging)
    return {"seed": 7, "tagging": t, "claims": {"timeout_minutes": 30}}


@pytest.fixture(autouse=True)
def _patch(monkeypatch):
    monkeypatch.setattr(tagger, "_task_prompt", lambda cfg: "TASK PROMPT")


def make_journal(tmp_path, n, cfg, t_summ=1000.0, tags=None):
    j = J.Journal.init(tmp_path / "run", cfg)
    with j.transaction("test", "seed") as st:
        for k in range(n):
            iid = J.new_idea_id(st)
            st["ideas"][iid] = {
                "id": iid, "created_at": float(k), "wave": 0, "arm": "optimized",
                "description": f"DESC {iid}", "summary": f"SUM {iid}", "tags": [],
                "status": "summarized", "times": {"summarized": t_summ}, "claim": None,
                "error": None, "retries": {},
            }
        for nm, d in (tags or {}).items():
            st["tags"][nm] = {"name": nm, "description": d, "created_at": 0.0, "ideas": []}
    return j


def run_cycle(j, cfg, llm, now=2000.0, monkeypatch=None):
    tagger._now = lambda: now
    try:
        return asyncio.run(tagger.tagging_cycle(j, cfg, llm))
    finally:
        tagger._now = __import__("time").time


# ----------------------------------------------------------------------------- partition

@pytest.mark.parametrize("n,S,r", [(1, 10, 2), (3, 10, 2), (20, 10, 2), (37, 10, 2), (100, 10, 2),
                                   (25, 4, 3), (7, 3, 1), (11, 2, 2), (50, 10, 5)])
def test_partition_properties(n, S, r):
    ids = [f"i{k:03d}" for k in range(n)]
    subs = tagger.partition(ids, S, r, np.random.default_rng([1, 0]))
    assert all(1 <= len(s) <= S for s in subs)
    assert all(len(set(s)) == len(s) for s in subs)            # no duplicate in a subset
    c = Counter(x for s in subs for x in s)
    assert set(c) == set(ids) and all(v == r for v in c.values())
    assert len(subs) == max(r, -(-n * r // S))


def test_partition_deterministic():
    ids = [f"i{k}" for k in range(40)]
    a = tagger.partition(ids, 10, 2, np.random.default_rng([7, 3]))
    b = tagger.partition(ids, 10, 2, np.random.default_rng([7, 3]))
    c = tagger.partition(ids, 10, 2, np.random.default_rng([7, 4]))
    assert a == b and a != c


# ----------------------------------------------------------------------------- parsing

def test_parse_writer_last_block_and_malformed():
    good = ('Draft:\n```json\n{"new_tags":[{"name":"old","description":"x"}]}\n```\n'
            'Final:\n```json\n{"new_tags":[{"name":" new ","description":"d"}]}\n```')
    assert tagger.parse_writer(good) == [{"name": "new", "description": "d"}]
    assert tagger.parse_writer('```json\n{"new_tags": []}\n```') == []
    for bad in ["no json here", "```json\n{not json}\n```", '```json\n{"tags": []}\n```',
                '```json\n{"new_tags":[{"description":"no name"}]}\n```',
                '```json\n{"new_tags":"x"}\n```', '```json\n[1,2]\n```']:
        assert tagger.parse_writer(bad) is None, bad
    # unterminated final fence is tolerated
    assert tagger.parse_writer('```json\n{"new_tags":[{"name":"a","description":"b"}]}') == \
        [{"name": "a", "description": "b"}]


POOL = ["Curriculum Learning", "loss-based ordering", "data_pruning", "Label Noise"]


@pytest.mark.parametrize("text,expect,n_norm", [
    ("Curriculum Learning\nlabel noise", ["Curriculum Learning", "Label Noise"], 1),
    ("- Curriculum Learning\n* data_pruning", ["Curriculum Learning", "data_pruning"], 0),
    ("1. `loss-based ordering`\n2) \"data_pruning\"", ["loss-based ordering", "data_pruning"], 0),
    ("- Curriculum Learning: orders examples easy to hard\n- Label Noise - noisy labels",
     ["Curriculum Learning", "Label Noise"], 0),
    ("**Curriculum Learning**\n'Label Noise'.", ["Curriculum Learning", "Label Noise"], 0),
    ("Tags: Curriculum Learning, data_pruning", ["Curriculum Learning", "data_pruning"], 0),
    ("- Loss Based Ordering\n- DATA PRUNING", ["loss-based ordering", "data_pruning"], 2),
    ("Here are the tags:\n- Nonexistent Tag\n- Curriculum Learning\n- Curriculum Learning",
     ["Curriculum Learning"], 0),
    ("nothing relevant\n- made up", [], 0),
])
def test_parse_tag_names_variants(text, expect, n_norm):
    got, nn, _unmatched = tagger.parse_tag_names(text, POOL)
    assert got == expect
    assert nn == n_norm


def test_parse_tag_names_ambiguous_normalized_not_matched():
    got, nn, _ = tagger.parse_tag_names("data pruning", ["Data-Pruning", "data_pruning"])
    assert got == [] and nn == 0
    got, nn, _ = tagger.parse_tag_names("data_pruning", ["Data-Pruning", "data_pruning"])
    assert got == ["data_pruning"] and nn == 0


# ----------------------------------------------------------------------------- trigger

def test_trigger_count_and_age(tmp_path):
    cfg = make_cfg(trigger_count=5, trigger_minutes=10)
    j = make_journal(tmp_path, 3, cfg, t_summ=1000.0)
    st = j.read()
    assert not tagger.should_trigger(st, cfg, 1000.0 + 599)
    assert tagger.should_trigger(st, cfg, 1000.0 + 600)
    cfg2 = make_cfg(trigger_count=3, trigger_minutes=10)
    assert tagger.should_trigger(st, cfg2, 1000.0)
    llm = FakeLLM()
    assert run_cycle(j, cfg, llm, now=1100.0) is None and llm.calls == []


# ----------------------------------------------------------------------------- full cycle

def test_cycle_merge_rounds_and_pool_growth(tmp_path):
    cfg = make_cfg(trigger_count=1, batch_limit=100, S=4, r=2, N=3, K=2)
    j = make_journal(tmp_path, 40, cfg, tags={"alpha": "existing"})
    llm = FakeLLM()
    e = run_cycle(j, cfg, llm)
    assert e["ok"] and e["n_ideas"] == 40
    assert e["n_subsets"] == 20 and llm.count("generator") == 20
    # 20 lists -> 7 -> 3 -> 1 (N=3, K=2): merges 7 + 2 + 1 calls (single leftovers pass through)
    assert e["lists_to_writer"] <= 2
    assert e["merge_rounds"] == 3 and llm.count("merger") == e["calls"]["merger"] == 10
    assert llm.count("writer") == 1
    assert e["new_tags"] == ["beta"] and e["dropped_existing"] == ["alpha"]
    assert e["pool_size_before"] == 1 and e["pool_size_after"] == 2
    assert e["calls"]["assign"] == 40 and e["calls"]["total"] == 20 + 10 + 1 + 40
    st = j.read()
    assert st["logs"]["tagging"][-1]["pool_size_after"] == 2
    assert all(i["status"] == "tagged" and i["tags"] == ["alpha", "beta"] and i["claim"] is None
               for i in st["ideas"].values())
    assert len(st["tags"]["beta"]["ideas"]) == 40 and "created_at" in st["tags"]["beta"]
    # creation used the pre-creation pool; assignment the updated pool
    gen_sys = next(m for k, m in llm.calls if k == "generator")[0]["content"]
    asg_sys = next(m for k, m in llm.calls if k == "assign")[0]["content"]
    assert "beta" not in gen_sys and "beta" in asg_sys and "TASK PROMPT" in gen_sys
    # parallelism
    assert llm.max_inflight > 1


def test_merge_covers_union_of_summaries(tmp_path):
    cfg = make_cfg(trigger_count=1, S=2, r=1, N=2, K=1)
    j = make_journal(tmp_path, 4, cfg)
    llm = FakeLLM()
    e = run_cycle(j, cfg, llm)
    assert e["ok"] and e["n_subsets"] == 2 and llm.count("merger") == 1
    merger_user = next(m for k, m in llm.calls if k == "merger")[-1]["content"]
    assert all(f"SUM i00000{k}" in merger_user for k in range(4))
    writer_user = next(m for k, m in llm.calls if k == "writer")[-1]["content"]
    assert all(f"SUM i00000{k}" in writer_user for k in range(4))


def test_no_merge_when_few_lists(tmp_path):
    cfg = make_cfg(trigger_count=1)
    j = make_journal(tmp_path, 15, cfg)          # 3 subsets <= K=4
    llm = FakeLLM()
    e = run_cycle(j, cfg, llm)
    assert e["n_subsets"] == 3 and llm.count("merger") == 0 and e["merge_rounds"] == 0


def test_batch_limit(tmp_path):
    cfg = make_cfg(trigger_count=1, batch_limit=7)
    j = make_journal(tmp_path, 10, cfg)
    e = run_cycle(j, cfg, FakeLLM())
    assert e["n_ideas"] == 7
    st = j.read()
    assert sum(i["status"] == "tagged" for i in st["ideas"].values()) == 7
    assert [i["id"] for i in st["ideas"].values() if i["status"] == "summarized"] == \
        ["i000007", "i000008", "i000009"]


def test_writer_retry_then_success(tmp_path):
    cfg = make_cfg(trigger_count=1)
    j = make_journal(tmp_path, 5, cfg)
    llm = FakeLLM(writer=lambda n: "oops {" if n == 1 else
                  '```json\n{"new_tags":[{"name":"gamma","description":"G"}]}\n```')
    e = run_cycle(j, cfg, llm, monkeypatch=None)
    assert e["ok"] and llm.count("writer") == 2 and e["calls"]["writer"] == 2
    assert e["new_tags"] == ["gamma"]


def test_writer_double_failure_releases_batch(tmp_path):
    cfg = make_cfg(trigger_count=1)
    j = make_journal(tmp_path, 5, cfg)
    llm = FakeLLM(writer=lambda n: "not json")
    e = run_cycle(j, cfg, llm)
    assert not e["ok"] and "malformed" in e["error"]
    assert llm.count("writer") == 2 and llm.count("assign") == 0
    st = j.read()
    assert st["tags"] == {}
    assert all(i["status"] == "summarized" and i["claim"] is None and i["retries"]["tagger"] == 1
               for i in st["ideas"].values())
    assert st["logs"]["tagging"][-1]["error"] == e["error"]
    # next cycle uses a different partition seed (cycle index advanced) and succeeds
    e2 = run_cycle(j, cfg, FakeLLM())
    assert e2["ok"] and e2["cycle"] == 1
    assert all(i["status"] == "tagged" for i in j.read()["ideas"].values())


def test_creation_exception_releases_batch(tmp_path):
    cfg = make_cfg(trigger_count=1)
    j = make_journal(tmp_path, 3, cfg)

    class Boom(FakeLLM):
        async def chat(self, messages, purpose):
            if kind(messages, purpose) == "generator":
                raise RuntimeError("server down")
            return await super().chat(messages, purpose)

    e = run_cycle(j, cfg, Boom())
    assert not e["ok"] and "server down" in e["error"]
    assert all(i["status"] == "summarized" and i["claim"] is None for i in j.read()["ideas"].values())


def test_assignment_retry_failure_and_normalized(tmp_path):
    cfg = make_cfg(trigger_count=1)
    j = make_journal(tmp_path, 4, cfg)

    def assign(desc, system, attempt):
        if "i000000" in desc:
            return "- not a tag"                         # always invalid -> failed
        if "i000001" in desc:
            return "garbage" if attempt == 1 else "- alpha"   # valid on retry
        if "i000002" in desc:
            return "- ALPHA\n- Beta."                     # normalized matches
        return "1. `alpha`: A\n2. beta"

    llm = FakeLLM(assign=assign)
    e = run_cycle(j, cfg, llm)
    assert e["ok"] and e["failures"] == ["i000000"]
    assert e["normalized_matches"] == 2
    assert e["calls"]["assign"] == 2 + 2 + 1 + 1
    st = j.read()
    ideas = st["ideas"]
    assert ideas["i000000"]["status"] == "failed" and "no valid tag" in ideas["i000000"]["error"]
    assert ideas["i000001"]["tags"] == ["alpha"] and ideas["i000001"]["status"] == "tagged"
    assert ideas["i000002"]["tags"] == ["alpha", "beta"]
    assert ideas["i000003"]["tags"] == ["alpha", "beta"]
    assert st["tags"]["alpha"]["ideas"] == ["i000001", "i000002", "i000003"]
    assert "i000000" not in st["tags"]["beta"]["ideas"]


def test_llm_never_called_under_lock(tmp_path):
    cfg = make_cfg(trigger_count=1)
    j = make_journal(tmp_path, 6, cfg)
    entry = J._entry(j._key)

    class Checking(FakeLLM):
        async def chat(self, messages, purpose):
            assert entry["owner"] is None, "LLM call while journal lock held"
            return await super().chat(messages, purpose)

    assert run_cycle(j, cfg, Checking())["ok"]


def test_run_loop_stops(tmp_path):
    cfg = make_cfg(trigger_count=1, poll_seconds=0.01)
    j = make_journal(tmp_path, 3, cfg)
    llm = FakeLLM()

    async def main():
        stop = asyncio.Event()
        task = asyncio.create_task(tagger.run(j, cfg, llm, stop))
        for _ in range(200):
            await asyncio.sleep(0.01)
            if all(i["status"] == "tagged" for i in j.read()["ideas"].values()):
                break
        stop.set()
        await asyncio.wait_for(task, 2)

    asyncio.run(main())
    st = j.read()
    assert all(i["status"] == "tagged" for i in st["ideas"].values())
    assert len(st["logs"]["tagging"]) == 1


def test_creation_failure_with_pool_still_assigns(tmp_path):
    """2026-11-11 fix: a failed creation step must not strand the batch when a pool exists."""
    cfg = make_cfg(trigger_count=1)
    j = make_journal(tmp_path, 3, cfg)
    with j.transaction("test", "seed_pool") as st:
        st["tags"]["Existing tag"] = {"name": "Existing tag", "description": "an existing tag", "created_at": 0.0, "ideas": []}

    class Boom(FakeLLM):
        async def chat(self, messages, purpose):
            if kind(messages, purpose) == "generator":
                raise RuntimeError("context length exceeded")
            return await super().chat(messages, purpose)

    e = run_cycle(j, cfg, Boom())
    assert "context length" in e.get("creation_error", "")
    assert all(i["status"] != "summarized" for i in j.read()["ideas"].values())


def test_compact_pool_and_prompt_budget():
    from ar.workers import tagger as T
    pool = [(f"Tag {k}", " ".join(["word"] * 60)) for k in range(400)]
    cp = T.compact_pool(pool)
    assert all(len(d.split()) <= T.POOL_DESC_WORDS + 1 for _, d in cp)
    from ar import prompts
    sysp = prompts.tagging_system("task " * 200, cp)
    assert T._est_tokens(sysp) < T.PROMPT_TOKEN_BUDGET
