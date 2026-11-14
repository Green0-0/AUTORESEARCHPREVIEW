import asyncio
import re

import pytest

from ar import config as arconfig
from ar.journal import Journal
from ar.llm import LLM, MockLLM
from ar.workers import judge

TASK = "Train a CIFAR-100 classifier in 15 minutes."


def make_cfg(tmp_path, **over):
    tp = tmp_path / "TASK.md"
    tp.write_text(TASK)
    user = {"run_id": "t", "task_id": "t", "seed": 3, "task_prompt_file": str(tp),
            "facts": ["Standard ResNet-18 gets 60%."], "judge": {"retries": 2}}
    user.update(over)
    return arconfig.from_dict(user)


def add_ideas(j, specs):
    """specs: list of (tags, status, s_cal). created_at increasing in list order."""
    with j.transaction("test", "seed") as st:
        for n, (tags, status, s_cal) in enumerate(specs):
            iid = f"i{n:06d}"
            st["ideas"][iid] = {
                "id": iid, "created_at": 1000.0 + n, "wave": 0, "arm": "optimized",
                "prompt": {"config": {"base": "Generic", "tag_slot": None, "NoIdea": False,
                                      "References": False, "RefIdeas": False},
                           "b_tilde": None, "turns": ["t"], "draws": [], "raw_outputs": ["r"]},
                "description": f"Idea number {n}", "summary": f"summary {n}", "tags": tags,
                "surrogate_scores": [] if s_cal is None else [s_cal], "judge_outputs": [],
                "s_cal": s_cal, "F": None, "F_detail": None, "scored_seq": None,
                "real_score": None, "real": None, "status": status, "times": {},
                "claim": None, "error": None, "retries": {}}
            for t in tags:
                st["tags"].setdefault(t, {"name": t, "description": t, "created_at": 0.0, "ideas": []})
                st["tags"][t]["ideas"].append(iid)
            st["counters"]["next_idea"] = n + 1


def run_until_done(j, cfg, llm, timeout=20):
    async def go():
        stop = asyncio.Event()
        t = asyncio.create_task(judge.Judge(j, cfg, llm, stop, poll_s=0.02).run())
        t0 = asyncio.get_running_loop().time()
        while True:
            await asyncio.sleep(0.02)
            st = j.read()
            if all(i["status"] in ("scored", "failed") for i in st["ideas"].values()):
                break
            assert asyncio.get_running_loop().time() - t0 < timeout
        stop.set()
        await asyncio.wait_for(t, 5)
    asyncio.run(go())
    return j.read()


class Scripted(LLM):
    """Judge answers from a per-idea script of strings (keyed by 'Idea number n')."""
    def __init__(self, script):
        super().__init__({}, client=object(), backoff_s=0)
        self.script = {k: list(v) for k, v in script.items()}
        self.n = {}

    async def _request(self, messages, purpose, p):
        assert purpose == "judge"
        assert messages[0]["role"] == "system" and TASK in messages[0]["content"]
        assert "Standard ResNet-18" in messages[0]["content"]
        k = int(re.search(r"Idea number (\d+)", messages[1]["content"]).group(1))
        self.n[k] = self.n.get(k, 0) + 1
        return {"content": self.script[k].pop(0), "reasoning": "", "finish_reason": "stop",
                "prompt_tokens": 1, "completion_tokens": 1}


def test_confirmation_logic(tmp_path):
    cfg = make_cfg(tmp_path)
    j = Journal.init(tmp_path / "run", cfg)
    add_ideas(j, [
        (["a"], "scored", 0.70),          # 0: earlier best in tag a
        (["a"], "tagged", None),          # 1: 0.9 then 0.5 -> F1=.15>0, mean .7 -> F=0 vs b_a=.7
        (["b"], "tagged", None),          # 2: new tag: 0.9 then 0.7 -> F1=.35, final .8 -> F=.25
        (["a", "b"], "tagged", None),     # 3: 0.52 -> b_a=.7, b_b depends on 2's s_cal at that time
        (["c"], "tagged", None),          # 4: garbage x3 -> failed (retries=2)
        (["c"], "tagged", None),          # 5: garbage, garbage, then 0.4 -> F=0, one call path
    ])
    script = {1: ["\\boxed{0.9}", "\\boxed{0.5}"],
              2: ["<think>x</think>\\boxed{0.9}", "\\boxed{0.7}"],
              3: ["\\boxed{0.52}"],
              4: ["no score"] * 3,
              5: ["nope", "still nope", "Answer: \\boxed{0.4}"]}
    llm = Scripted(script)
    st = run_until_done(j, cfg, llm)
    I = st["ideas"]
    i1, i2, i4, i5 = I["i000001"], I["i000002"], I["i000004"], I["i000005"]
    assert i1["status"] == "scored" and i1["surrogate_scores"] == [0.9, 0.5]
    assert i1["F_detail"]["b_t"] == {"a": 0.7} and i1["F_detail"]["F_first"] == pytest.approx(0.15)
    assert i1["s_cal"] == pytest.approx(0.7) and i1["F"] == 0.0
    assert i1["F_detail"]["confirmation_call"] and not i1["F_detail"]["confirmed"]
    assert i1["F_detail"]["s_first"] == pytest.approx(0.9) and i1["F_detail"]["s_final"] == pytest.approx(0.7)
    assert len(i1["judge_outputs"]) == 2
    assert i2["F_detail"]["b_t"] == {"b": 0.5}
    assert i2["F"] == pytest.approx(0.25) and i2["F_detail"]["confirmed"]
    assert "<think>" in i2["judge_outputs"][0]                      # raw kept
    assert i4["status"] == "failed" and llm.n[4] == 3 and "boxed" in i4["error"]
    assert i5["status"] == "scored" and i5["surrogate_scores"] == [0.4] and llm.n[5] == 3
    assert i5["F"] == 0.0 and not i5["F_detail"]["confirmation_call"]
    assert i5["retries"]["judge"] == 2
    i3 = I["i000003"]
    assert i3["surrogate_scores"] == [0.52] and llm.n[3] == 1 and i3["F"] == 0.0
    assert i3["F_detail"]["b_t"]["a"] == 0.7
    # scored_seq: unique, consecutive, set only on scored ideas
    seqs = sorted(i["scored_seq"] for i in I.values() if i["status"] == "scored" and i["id"] != "i000000")
    assert seqs == list(range(len(seqs)))
    assert st["counters"]["scored_seq"] == len(seqs)
    assert all(i["claim"] is None for i in I.values())


def test_calibration_applied(tmp_path):
    cfg = make_cfg(tmp_path, scoring={"s_min": 0.05, "s_base": 0.5, "calibration": {"a": 0.1, "b": 0.5}})
    j = Journal.init(tmp_path / "run", cfg)
    add_ideas(j, [(["a"], "tagged", None)])
    st = run_until_done(j, cfg, Scripted({0: ["\\boxed{0.8}"]}))
    i = st["ideas"]["i000000"]
    assert i["surrogate_scores"] == [0.8] and i["s_cal"] == pytest.approx(0.5)  # .1+.5*.8
    assert i["F"] == 0.0 and len(i["judge_outputs"]) == 1


def test_mock_many_ideas_frozen_F(tmp_path):
    cfg = make_cfg(tmp_path)
    j = Journal.init(tmp_path / "run", cfg)
    tags = ["a", "b", "c"]
    add_ideas(j, [([tags[n % 3]] + ([tags[(n + 1) % 3]] if n % 4 == 0 else []), "tagged", None)
                  for n in range(40)])
    llm = MockLLM(cfg["llm"], tmp_path / "run" / "llm_usage.jsonl", seed=7)
    st = run_until_done(j, cfg, llm)
    ideas = list(st["ideas"].values())
    assert all(i["status"] == "scored" for i in ideas)
    sm, cal = cfg["scoring"]["s_min"], cfg["scoring"]["calibration"]
    n_conf = 0
    for i in ideas:
        d = i["F_detail"]
        assert set(d) >= {"b_t", "s_first", "s_final", "confirmed"}
        assert set(d["b_t"]) == set(i["tags"])
        assert all(0.3 <= s <= 0.9 for s in i["surrogate_scores"])
        expect_first = max(0.0, max(d["s_first"] - b - sm for b in d["b_t"].values()))
        assert d["F_first"] == pytest.approx(expect_first)
        if d["F_first"] > 0:
            n_conf += 1
            assert len(i["surrogate_scores"]) == 2
            mean = sum(i["surrogate_scores"]) / 2
            assert i["s_cal"] == pytest.approx(mean)
        else:
            assert len(i["surrogate_scores"]) == 1 and i["F"] == 0.0
        # final F is against the frozen b_t
        assert i["F"] == pytest.approx(max(0.0, max(i["s_cal"] - b - sm for b in d["b_t"].values())))
    assert n_conf > 0
    assert sorted(i["scored_seq"] for i in ideas) == list(range(40))


def test_resume_judged(tmp_path):
    """An idea left in `judged` (crash before confirmation) is finished from its frozen b_t."""
    cfg = make_cfg(tmp_path)
    j = Journal.init(tmp_path / "run", cfg)
    add_ideas(j, [(["a"], "judged", 0.9)])
    with j.transaction("test", "x") as st:
        st["ideas"]["i000000"]["judge_outputs"] = ["\\boxed{0.9}"]
        st["ideas"]["i000000"]["F_detail"] = {"b_t": {"a": 0.6}, "s_first": 0.9, "s_final": None,
                                              "F_first": 0.25, "best_tag": "a", "confirmed": False,
                                              "confirmation_call": True}
    llm = Scripted({0: ["\\boxed{0.7}"]})
    st = run_until_done(j, cfg, llm)
    i = st["ideas"]["i000000"]
    assert i["status"] == "scored" and i["surrogate_scores"] == [0.9, 0.7]
    assert i["F"] == pytest.approx(0.8 - 0.6 - 0.05) and i["F_detail"]["b_t"] == {"a": 0.6}
    assert llm.n[0] == 1
