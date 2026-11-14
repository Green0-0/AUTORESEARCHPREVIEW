import asyncio
import time

import numpy as np
import pytest

from ar import config as arconfig
from ar import prompts
from ar.journal import Journal
from ar.llm import MockLLM
from ar.workers import ideator

TASK = "Train a CIFAR-100 classifier in 15 minutes; only the data ordering may change."


def make_cfg(tmp_path, **over):
    tp = tmp_path / "TASK.md"
    tp.write_text(TASK)
    user = {"run_id": "t", "task_id": "t", "seed": 5, "task_prompt_file": str(tp),
            "references": ["Ref A: curriculum by loss.", "Ref B: anti-curriculum."],
            "facts": ["A standard ResNet-18 gets 60% here."],
            "ideation": {"batch_size": 4, "concurrent_batches": 2},
            "stop": {"max_hours": 1, "max_ideas": 12}}
    for k, v in over.items():
        user[k] = v
    return arconfig.from_dict(user)


def run_ideator(j, cfg, llm, timeout=20, expect_all=True):
    """Run until stop.max_ideas ideas are all written and past `generated`, then stop."""
    async def go():
        stop = asyncio.Event()
        t = asyncio.create_task(ideator.run(j, cfg, llm, stop))
        t0 = time.time()
        while True:
            await asyncio.sleep(0.02)
            assert not t.done(), t.exception()
            st = j.read()
            n = st["counters"]["next_idea"]
            seeded = sum(1 for k in st["ideas"] if k.startswith("i999"))
            if (n >= cfg["stop"]["max_ideas"] and (not expect_all or len(st["ideas"]) - seeded == n)
                    and all(i["status"] != "generated" for i in st["ideas"].values())):
                break
            assert time.time() - t0 < timeout
        stop.set()
        await asyncio.wait_for(t, 5)
    asyncio.run(go())


def seed_state(j):
    """Give the run a tag pool and one judged idea so tag/idea appends become available."""
    now = time.time() - 100
    with j.transaction("test", "seed") as st:
        st["tags"]["curriculum"] = {"name": "curriculum", "description": "orders data by difficulty",
                                    "created_at": now, "ideas": ["i999000"]}
        st["ideas"]["i999000"] = {
            "id": "i999000", "created_at": now, "wave": -1, "arm": "optimized",
            "prompt": {"config": {"base": "Generic", "tag_slot": None, "NoIdea": False,
                                  "References": False, "RefIdeas": False},
                       "b_tilde": None, "turns": [], "draws": [], "raw_outputs": []},
            "description": "Seed idea {{prompt}} with braces", "summary": "Seed summary {{tags}}",
            "tags": ["curriculum"], "surrogate_scores": [0.6], "judge_outputs": ["\\boxed{0.6}"],
            "s_cal": 0.6, "F": 0.05, "F_detail": None, "scored_seq": 0, "real_score": None,
            "real": None, "status": "scored", "times": {}, "claim": None, "error": None, "retries": {}}
        st["counters"]["scored_seq"] = 1


def test_ideator_produces_summarized_ideas(tmp_path):
    cfg = make_cfg(tmp_path)
    j = Journal.init(tmp_path / "run", cfg)
    seed_state(j)
    llm = MockLLM(cfg["llm"], tmp_path / "run" / "llm_usage.jsonl", seed=1)
    run_ideator(j, cfg, llm)
    st = j.read()
    new = [i for k, i in st["ideas"].items() if k != "i999000"]
    assert len(new) == 12                                     # stop.max_ideas
    assert all(i["status"] == "summarized" for i in new)
    assert st["counters"]["wave"] == 3
    assert sorted({i["wave"] for i in new}) == [0, 1, 2]
    assert len(st["logs"]["refits"]) == 3 and st["logs"]["refits"][0]["n"] == 1
    assert len(st["logs"]["batches"]) == 3
    assert all(len(b["draws"]) == 4 and len(b["draws"][0]["b_tilde"]) == 12 for b in st["logs"]["batches"])
    for i in new:
        p = i["prompt"]
        assert set(p) >= {"config", "b_tilde", "turns", "draws", "raw_outputs", "system"}
        assert TASK in p["system"] and "ResNet-18" in p["system"]
        assert len(p["turns"]) == len(p["raw_outputs"]) == len(p["finish_reasons"]) >= 1
        assert len(p["turns"]) == len(prompts.BASE_PROMPTS[p["config"]["base"]])
        assert "<think>" in p["raw_outputs"][-1]               # raw kept
        assert "<think>" not in i["description"] and i["description"].startswith("Idea (turn")
        assert i["summary"].startswith("Summary:")
        assert i["times"]["generated"] <= i["times"]["summarized"]
        assert i["created_at"] == i["times"]["generated"]
    # with tags/judged ideas/references available, at least some appends were used
    cfgs = [i["prompt"]["config"] for i in new]
    assert any(c["References"] or c["RefIdeas"] or c["NoIdea"] or c["tag_slot"] for c in cfgs)
    # multi-turn: the final turn was sent after the model's answer to turn 1
    multi = [i for i in new if len(i["prompt"]["turns"]) > 1]
    ideation_calls = [m for p, m in llm.calls if p == "ideation"]
    for i in multi:
        last = [m for m in ideation_calls if m[-1]["content"] == i["prompt"]["turns"][-1]
                and m[0]["content"] == i["prompt"]["system"]]
        assert last and last[0][-2]["role"] == "assistant" and "<think>" not in last[0][-2]["content"]
    # summary call has no system prompt
    assert all(m[0]["role"] == "user" and len(m) == 1 for p, m in llm.calls if p == "summary")
    assert (tmp_path / "run" / "llm_usage.jsonl").exists()


def test_naive_arm_samples_prompt_pool_uniformly(tmp_path):
    cfg = make_cfg(tmp_path, arm="naive")
    assert cfg["sampler"]["mode"] == "uniform"
    j = Journal.init(tmp_path / "run", cfg)
    seed_state(j)
    run_ideator(j, cfg, MockLLM(cfg["llm"], seed=2))
    new = [i for k, i in j.read()["ideas"].items() if k != "i999000"]
    assert len(new) == 12
    for i in new:
        assert i["prompt"]["b_tilde"] is None          # no posterior draw: the naive sampler never learns
        assert i["arm"] == "naive"
    assert len({i["prompt"]["config"]["base"] for i in new}) > 1   # not Generic-only


def test_idea_rng_deterministic():
    a = ideator.idea_rng(5, "i000123").random(3)
    b = ideator.idea_rng(5, "i000123").random(3)
    c = ideator.idea_rng(5, "i000124").random(3)
    assert np.allclose(a, b) and not np.allclose(a, c)


def test_truncation_fails_after_one_retry(tmp_path):
    cfg = make_cfg(tmp_path, stop={"max_hours": 1, "max_ideas": 4})
    j = Journal.init(tmp_path / "run", cfg)
    llm = MockLLM(cfg["llm"], seed=0, length_p=1.0)          # every thinking answer truncated
    run_ideator(j, cfg, llm)
    st = j.read()
    assert len(st["ideas"]) == 4
    for i in st["ideas"].values():
        assert i["status"] == "failed" and "empty answer" in i["error"]
        assert len(i["prompt"]["failed_raw_outputs"]) == 2    # first try + one retry
        assert i["prompt"]["raw_outputs"] == []
    assert st["counters"]["length_finish"]["ideation"] == 8
    assert sum(1 for p, _ in llm.calls if p == "ideation") == 8
    assert not any(p == "summary" for p, _ in llm.calls)


def test_summary_failure_marks_failed(tmp_path):
    cfg = make_cfg(tmp_path, stop={"max_hours": 1, "max_ideas": 4})
    j = Journal.init(tmp_path / "run", cfg)

    class SumFails(MockLLM):
        async def _request(self, messages, purpose, p):
            if purpose == "summary":
                raise ValueError("bad request")
            return await super()._request(messages, purpose, p)
    run_ideator(j, cfg, SumFails(cfg["llm"], seed=0))
    st = j.read()
    assert all(i["status"] == "failed" and i["error"].startswith("summary") for i in st["ideas"].values())
    assert all(i["description"] for i in st["ideas"].values())


def test_stop_event_and_resume(tmp_path):
    cfg = make_cfg(tmp_path, stop={"max_hours": 1, "max_ideas": None})
    j = Journal.init(tmp_path / "run", cfg)
    llm = MockLLM(cfg["llm"], seed=0, latency_s=0.05)

    async def go():
        stop = asyncio.Event()
        t = asyncio.create_task(ideator.run(j, cfg, llm, stop))
        await asyncio.sleep(0.6)
        stop.set()
        await asyncio.wait_for(t, 5)
    asyncio.run(go())
    st = j.read()
    assert len(st["ideas"]) > 0
    assert st["counters"]["wave"] >= 2
    # simulate a crash between generated and summarized, then resume
    with j.transaction("test", "unsummarize") as s:
        iid = next(iter(s["ideas"]))
        s["ideas"][iid].update(status="generated", summary=None)
    cfg2 = dict(cfg, stop={"max_hours": 1, "max_ideas": st["counters"]["next_idea"]})
    run_ideator(j, cfg2, MockLLM(cfg["llm"], seed=0), expect_all=False)
    assert j.read()["ideas"][iid]["status"] == "summarized"


def test_backpressure_pauses_when_backlog_full(tmp_path):
    cfg = make_cfg(tmp_path)
    cfg["ideation"]["max_backlog"] = 1
    j = Journal.init(tmp_path / "run", cfg)
    seed_state(j)
    with j.transaction("test", "add") as st:
        for k in range(2):
            iid = f"i88800{k}"
            st["ideas"][iid] = {"id": iid, "status": "summarized", "created_at": 0.0}
    ide = ideator.Ideator(j, cfg, MockLLM(cfg["llm"], seed=1), asyncio.Event())
    assert ide._backlogged()
    cfg["ideation"]["max_backlog"] = None
    assert not ide._backlogged()
