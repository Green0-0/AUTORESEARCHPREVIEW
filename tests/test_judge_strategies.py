"""Judge strategies J1-J5 (plan §9.1): prompts, parsers, config, selection helpers and each pipeline judge
mode with MockLLM / scripted LLMs."""
import asyncio
import re
from pathlib import Path

import numpy as np
import pytest

from ar import anchors, judging, prompts as P, scoring
from ar import config as arconfig
from ar.journal import Journal
from ar.llm import LLM, MockLLM
from ar.workers import ideator, judge

from test_judge import TASK, add_ideas, run_until_done

NOTE = Path(__file__).resolve().parents[1] / "class_notes" / "Autoresearch Draft Spec Graph Note.md"


# ----------------------------------------------------------------------------- prompts
def _sections():
    t = NOTE.read_text()
    out = {}
    for m in re.finditer(r"^## (.+?)\n\n(.*?)(?=^## |\Z)", t, re.S | re.M):
        out.setdefault(m.group(1).strip(), []).append(m.group(2).strip())
    return out


@pytest.mark.skipif(not NOTE.exists(), reason="graph note not present")
def test_judge_prompts_verbatim_from_graph_note():
    s = _sections()
    assert P.ARBITRARY_RUBRIC == s["Arbitrary Rubric"][0]
    assert P.BASIC_FS == s["Basic FS"][0]
    assert P.BASIC_SURROGATE in s["Basic"]
    assert P.JUDGE_SYSTEM == s["General System Prompt"][0]
    # the ranking prompt keeps the spec's description sentence
    assert "with the goal of correctly ranking everything" in P.RANKING
    assert "relatively in mind" in s["Ranking Based"][0] and "relatively in mind" in P.RANKING


def test_fewshot_and_ranking_prompt_format():
    fs = P.fewshot_prompt("NEW {{prompt}} idea", [("first\n summary", 0.3), ("second", 1)])
    assert fs.startswith("Here is a new idea: NEW {{prompt}} idea\n\n")       # inert substitution
    assert "- first summary (judging result: 0.30)\n- second (judging result: 1.00).\n\n" in fs
    assert fs.endswith("within \\boxed{}")
    refs = anchors.ranking_references("t07_activation_tuning")
    rk = P.ranking_prompt(["desc one", "desc two", "desc three"], refs)
    assert "Here are 3 new ideas:" in rk
    assert "Reference A (known judging result: 0.00):\nStarter baseline" in rk
    assert "Reference B (known judging result: 1.00):\nNormalized learnable activations" in rk
    assert re.findall(r"^Idea (\d+):$", rk, re.M) == ["1", "2", "3"]
    assert "from Idea 1 to Idea 3" in rk and rk.endswith("Idea 1: \\boxed{0.xx}")
    assert P.rubric_prompt("X").startswith("Here is a new idea:\nX\n\nCreate a relevant rubric")


def test_anchor_data_complete():
    refs = anchors.reference_ideas()
    for t in anchors.STARTER_IDEAS:
        assert t in refs and t in anchors.ANCHOR_SHORT
    for t in ("t07_activation_tuning", "t10_label_noise", "t15_lwf", "t09_tokenizer_transfer",
              "t13_robust_noadv", "t16_synthetic_data", "t18_anytime"):
        f = anchors.anchor_facts(t)
        assert len(f) == 2 and "score 0" in f[0] and "score 1.0" in f[1]
    with pytest.raises(ValueError):
        anchors.anchor_facts("t19_data_ordering")      # ref is null


# ----------------------------------------------------------------------------- parsers
@pytest.mark.parametrize("text,n,want", [
    ("Idea 1: \\boxed{0.4}\nIdea 2: \\boxed{0.7}", 2, [0.4, 0.7]),
    ("Idea 2 beats Idea 1.\n**Idea 1**: \\boxed{0.2}\n- **Idea 2** \u2014 \\boxed{65\\%}", 2, [0.2, 0.65]),
    ("1. \\boxed{0.1}\n2) \\boxed{0.3}\n[3]: \\boxed{0.5}", 3, [0.1, 0.3, 0.5]),
    ("### Idea 1\nlong reasoning\n\\boxed{0.3}\n### Idea 2\nmore\n\\boxed{0.8}", 2, [0.3, 0.8]),
    ("Idea 1: 0.35\nIdea 2: 45%", 2, [0.35, 0.45]),
    ("\\boxed{0.1} and \\boxed{0.9}", 2, [0.1, 0.9]),                      # unlabelled, count matches
    ("\\boxed{0.1} and \\boxed{0.9}", 3, [None, None, None]),               # unlabelled, count mismatch
    ("Idea 1: \\boxed{0.4}\nlater: Idea 1: \\boxed{0.5}", 2, [0.5, None]),  # last answer wins; 2 missing
    ("Reference A: \\boxed{0.0}\nIdea 1: \\boxed{\\frac{1}{4}}", 1, [0.25]),
    ("Idea 1: \\boxed{1.7}\nIdea 2: \\boxed{-0.2}\nIdea 9: \\boxed{0.5}", 2, [1.0, 0.0]),  # clamped; 9 ignored
    ("Idea 1: \\boxed{0.xx}\nIdea 2: \\boxed{0.6}", 2, [None, 0.6]),
    ("", 2, [None, None]),
    ("Idea 1: \\boxed{0.4", 1, [None]),                                        # unterminated
])
def test_parse_ranked(text, n, want):
    got = scoring.parse_ranked(text, n)
    assert len(got) == n
    for g, w in zip(got, want):
        assert (g is None and w is None) or g == pytest.approx(w)


def test_parse_boxed_rubric_takes_final():
    t = "Novelty: \\boxed{0.9}\nFeasibility: \\boxed{0.3}\nAverage = \\boxed{0.6}"
    assert scoring.parse_boxed(t) == pytest.approx(0.6)


# ----------------------------------------------------------------------------- config
def _user(tmp_path, judge_cfg=None, task_id="t07_activation_tuning", **over):
    tp = tmp_path / "TASK.md"
    tp.write_text(TASK)
    u = {"run_id": "r", "task_id": task_id, "seed": 1, "task_prompt_file": str(tp)}
    if judge_cfg is not None:
        u["judge"] = judge_cfg
    u.update(over)
    return u


def test_config_judge_defaults(tmp_path):
    cfg = arconfig.from_dict(_user(tmp_path))
    jc = cfg["judge"]
    assert jc["strategy"] == "basic" and jc["facts"] is None and jc["anchor_facts"] is False
    assert jc["k"] == 5 and jc["fewshot_source"] == "surrogate" and jc["fewshot_n"] == 6
    assert jc["rank_batch"] == 5 and jc["retries"] == 2


@pytest.mark.parametrize("jc,msg", [
    ({"strategy": "pairwise"}, "judge.strategy"),
    ({"fewshot_source": "oracle"}, "fewshot_source"),
    ({"k": 0}, "judge.k"),
    ({"facts": "one fact"}, "judge.facts"),
    ({"rank_batch": 2.5}, "rank_batch"),
    ({"bogus": 1}, "unknown key"),
    ({"facts_file": "nope/missing.yaml"}, "facts_file not found"),
])
def test_config_judge_rejects(tmp_path, jc, msg):
    with pytest.raises(arconfig.ConfigError, match=msg):
        arconfig.from_dict(_user(tmp_path, jc))


def test_config_ranking_and_anchor_need_task_data(tmp_path):
    arconfig.from_dict(_user(tmp_path, {"strategy": "ranking", "anchor_facts": True}))
    with pytest.raises(arconfig.ConfigError, match="no reference ideas"):
        arconfig.from_dict(_user(tmp_path, {"strategy": "ranking"}, task_id="t_unknown"))
    with pytest.raises(arconfig.ConfigError, match="anchor_facts"):
        arconfig.from_dict(_user(tmp_path, {"anchor_facts": True}, task_id="t19_data_ordering"))


def test_judge_facts_are_judge_only(tmp_path):
    ff = tmp_path / "jf.yaml"
    ff.write_text("- file fact\n")
    legacy = arconfig.from_dict(_user(tmp_path, facts=["ideation fact"]))
    assert judging.judge_facts(legacy) == ["ideation fact"]          # null = legacy: inherit top-level facts
    cfg = arconfig.from_dict(_user(tmp_path, {"facts": ["judge fact"], "facts_file": str(ff), "anchor_facts": True},
                                   facts=["ideation fact"]))
    jf = judging.judge_facts(cfg)
    assert jf[:2] == ["judge fact", "file fact"] and len(jf) == 4 and "ideation fact" not in jf
    assert "0.323" in jf[2]
    ctx = ideator.snapshot_context({"tags": {}, "ideas": {}}, cfg, TASK)
    assert ctx.facts == ["ideation fact"]                            # ideators never see judge facts
    sysmsg = P.judge_system(TASK, jf, np.random.default_rng(0))
    assert "judge fact" in sysmsg and "ideation fact" not in sysmsg


# ----------------------------------------------------------------------------- selection helpers
def test_select_examples_stratified_and_loo():
    cands = [{"id": f"c{i}", "summary": f"unique summary number {i} about topic{i}", "score": i / 19} for i in range(20)]
    cands.append({"id": "dup", "summary": "the new idea summary text here", "score": 0.5})
    rng = np.random.default_rng(1)
    ch = judging.select_examples(cands, 5, rng, exclude_id="c3", exclude_texts=("the new idea summary text here",))
    ids = [c["id"] for c in ch]
    assert len(ids) == 5 and "c3" not in ids and "dup" not in ids
    scores = sorted(c["score"] for c in ch)
    # one per quintile bin of the 19 remaining
    assert scores[0] < 0.25 and scores[-1] > 0.75
    again = judging.select_examples(cands, 5, np.random.default_rng(1), exclude_id="c3",
                                    exclude_texts=("the new idea summary text here",))
    assert [c["id"] for c in again] == ids
    assert len(judging.select_examples(cands[:3], 6, rng)) == 3
    assert judging.select_examples([], 6, rng) == []


def test_ranking_batches_cover_each_key_R_times():
    keys = [("run", f"i{i}") for i in range(19)]
    reps = judging.ranking_batches(keys, 5, 2, seed=0)
    assert len(reps) == 2
    for rep in reps:
        assert sorted(k for b in rep for k in b) == sorted(keys)
        assert len(rep) == 4 and all(4 <= len(b) <= 5 for b in rep)
    pairs = lambda rep: {frozenset((a, b)) for bt in rep for a in bt for b in bt if a != b}  # noqa: E731
    assert len(pairs(reps[0]) & pairs(reps[1])) < len(pairs(reps[0])) / 2
    assert judging.ranking_batches(keys, 5, 2, seed=0) == reps


# ----------------------------------------------------------------------------- judge modes (worker)
def _cfg(tmp_path, judge_cfg, task_id="t19_data_ordering", **over):
    return arconfig.from_dict(_user(tmp_path, {"retries": 2, **judge_cfg}, task_id=task_id,
                                    facts=["Standard ResNet-18 gets 60%."], **over))


def _check_scored(st, cfg):
    sm = cfg["scoring"]["s_min"]
    for i in st["ideas"].values():
        if i["status"] != "scored" or i["F_detail"] is None:
            continue
        d = i["F_detail"]
        n = 2 if d["F_first"] > 0 else 1
        assert len(i["surrogate_scores"]) == n == len(i["judge_outputs"])
        assert i["s_cal"] == pytest.approx(float(np.mean(i["surrogate_scores"])))
        assert i["F"] == pytest.approx(max(0.0, max(i["s_cal"] - b - sm for b in d["b_t"].values())))


def _tagged(n, start_scored=0):
    tags = ["a", "b", "c"]
    specs = [([tags[k % 3]], "scored", 0.2 + 0.05 * k) for k in range(start_scored)]
    return specs + [([tags[k % 3]], "tagged", None) for k in range(n)]


def test_mode_basic_k(tmp_path):
    cfg = _cfg(tmp_path, {"strategy": "basic_k", "k": 3})
    j = Journal.init(tmp_path / "run", cfg)
    add_ideas(j, _tagged(8))
    llm = MockLLM(cfg["llm"], seed=1)
    st = run_until_done(j, cfg, llm)
    assert all(i["status"] == "scored" for i in st["ideas"].values())
    n_judgments = sum(len(i["surrogate_scores"]) for i in st["ideas"].values())
    assert len(llm.calls) == 3 * n_judgments
    for i in st["ideas"].values():
        for v, d in zip(i["surrogate_scores"], i["judge_detail"]):
            assert d["strategy"] == "basic_k" and len(d["samples"]) == 3 and v == pytest.approx(np.mean(d["samples"]))
    _check_scored(st, cfg)


def test_mode_rubric(tmp_path):
    cfg = _cfg(tmp_path, {"strategy": "rubric"})
    j = Journal.init(tmp_path / "run", cfg)
    add_ideas(j, _tagged(6))
    llm = MockLLM(cfg["llm"], seed=2)
    st = run_until_done(j, cfg, llm)
    assert all(i["status"] == "scored" for i in st["ideas"].values())
    assert all("Create a relevant rubric" in m[1]["content"] for _, m in llm.calls)
    assert all(d["strategy"] == "rubric" for i in st["ideas"].values() for d in i["judge_detail"])
    _check_scored(st, cfg)


def test_mode_fewshot_surrogate(tmp_path):
    cfg = _cfg(tmp_path, {"strategy": "fewshot", "fewshot_n": 3})
    j = Journal.init(tmp_path / "run", cfg)
    add_ideas(j, _tagged(6, start_scored=5))
    llm = MockLLM(cfg["llm"], seed=3)
    st = run_until_done(j, cfg, llm)
    new = [i for i in st["ideas"].values() if int(i["id"][1:]) >= 5]
    assert all(i["status"] == "scored" for i in new)
    for i in new:
        d = i["judge_detail"][0]
        assert d["strategy"] == "fewshot" and d["source"] == "surrogate" and len(d["examples"]) == 3
        assert i["id"] not in [e[0] for e in d["examples"]]
        created = {x["id"]: x["created_at"] for x in st["ideas"].values()}
        assert all(created[e] < i["created_at"] for e, _ in d["examples"])     # only earlier ideas
    fs_users = [m[1]["content"] for _, m in llm.calls]
    assert all("Here are some previous ideas, each with their judging result:" in u for u in fs_users)
    assert all("(judging result: 0." in u for u in fs_users)
    _check_scored(st, cfg)


def test_mode_fewshot_real_and_fallback(tmp_path):
    cfg = _cfg(tmp_path, {"strategy": "fewshot", "fewshot_source": "real", "fewshot_n": 2,
                          "fewshot_runs": ["other"]})
    # another run of the same task with real results
    other = Journal.init(tmp_path / "other", cfg)
    add_ideas(other, [(["a"], "real_judged", 0.4), (["a"], "real_judged", 0.6)])
    with other.transaction("t", "real") as s2:
        for n, iid in enumerate(s2["ideas"]):
            s2["ideas"][iid].update(real_score=0.1 + 0.7 * n, real={"value": 0.5 + n})
            s2["ideas"][iid]["summary"] = f"external real idea {n} with distinct words {n * 7}"
    j = Journal.init(tmp_path / "run", cfg)
    add_ideas(j, _tagged(4))
    llm = MockLLM(cfg["llm"], seed=4)
    st = run_until_done(j, cfg, llm)
    for i in st["ideas"].values():
        assert i["status"] == "scored"
        ex = i["judge_detail"][0]["examples"]
        assert sorted(e[0] for e in ex) == ["other/i000000", "other/i000001"]
        assert sorted(e[1] for e in ex) == [pytest.approx(0.1), pytest.approx(0.8)]
    # no examples available at all -> Basic prompt fallback
    (tmp_path / "x").mkdir()
    cfg2 = _cfg(tmp_path / "x", {"strategy": "fewshot", "fewshot_source": "real"})
    j2 = Journal.init(tmp_path / "run2", cfg2)
    add_ideas(j2, _tagged(2))
    llm2 = MockLLM(cfg2["llm"], seed=5)
    st2 = run_until_done(j2, cfg2, llm2)
    for i in st2["ideas"].values():
        assert i["status"] == "scored" and i["judge_detail"][0].get("fallback") == "basic"
    assert all("Output the expected performance of the idea" in m[1]["content"] for _, m in llm2.calls)


def test_mode_ranking_batches_and_confirmation(tmp_path):
    cfg = _cfg(tmp_path, {"strategy": "ranking", "rank_batch": 4, "rank_max_wait_s": 0.2, "anchor_facts": False})
    j = Journal.init(tmp_path / "run", cfg)
    add_ideas(j, _tagged(10))
    llm = MockLLM(cfg["llm"], seed=6)
    st = run_until_done(j, cfg, llm)
    ideas = list(st["ideas"].values())
    assert all(i["status"] == "scored" for i in ideas)
    for _, m in llm.calls:
        u = m[1]["content"]
        n = int(re.search(r"Here are (\d+) new ideas", u).group(1))
        assert 1 <= n <= 4 and len(re.findall(r"^Idea \d+:$", u, re.M)) == n
        assert "Reference A (known judging result: 0.00)" in u and "uniform random reshuffle" in u
        assert "Reference B (known judging result: 1.00)" in u and "Online batch selection" in u
        assert "Standard ResNet-18" in m[0]["content"]
    assert len(llm.calls) < sum(len(i["surrogate_scores"]) for i in ideas)   # batched
    confirmed = [i for i in ideas if len(i["surrogate_scores"]) == 2]
    assert confirmed, "expected some F>0 confirmations"
    for i in confirmed:
        b1, b2 = (d["batch"] for d in i["judge_detail"])
        assert i["id"] in b1 and i["id"] in b2
    _check_scored(st, cfg)


class RankScript(LLM):
    """Ranking answers that omit one idea id per call; records batch sizes."""
    def __init__(self, omit_ids=(2,)):
        super().__init__({}, client=object(), backoff_s=0)
        self.sizes, self.omit = [], set(omit_ids)

    async def _request(self, messages, purpose, p):
        ids = [int(k) for k in re.findall(r"^Idea (\d+):$", messages[1]["content"], re.M)]
        self.sizes.append(len(ids))
        text = "\n".join(f"Idea {k}: \\boxed{{0.{10 + k}}}" for k in ids if k not in self.omit)
        return {"content": text, "reasoning": "", "finish_reason": "stop", "prompt_tokens": 1, "completion_tokens": 1}


def test_ranking_missing_ids_are_reasked_then_failed(tmp_path):
    cfg = _cfg(tmp_path, {"strategy": "ranking", "rank_batch": 3, "rank_max_wait_s": 0.0, "retries": 1},
               scoring={"s_min": 0.05, "s_base": 0.9, "calibration": {"a": 0.0, "b": 1.0}})   # F = 0: no confirmations
    # Idea 2 missing: re-asked alone (as "Idea 1" of a 1-idea batch) and answered
    llm = RankScript(omit_ids=(2,))
    j = Journal.init(tmp_path / "run", cfg)
    add_ideas(j, _tagged(3))
    st = run_until_done(j, cfg, llm)
    assert all(i["status"] == "scored" for i in st["ideas"].values())
    assert llm.sizes == [3, 1]
    assert sorted(i["judge_detail"][0]["attempt"] for i in st["ideas"].values()) == [1, 1, 2]
    # Idea 1 always missing: retries exhausted -> that idea fails, the others are scored
    llm2 = RankScript(omit_ids=(1,))
    j2 = Journal.init(tmp_path / "run2", cfg)
    add_ideas(j2, _tagged(3))
    st2 = run_until_done(j2, cfg, llm2)
    status = [i["status"] for i in st2["ideas"].values()]
    assert status.count("scored") == 2 and status.count("failed") == 1 and llm2.sizes == [3, 1]
    failed = [i for i in st2["ideas"].values() if i["status"] == "failed"][0]
    assert "no score for Idea 1" in failed["error"] and failed["error"].startswith("judge ranking:")


def test_default_strategy_is_unchanged_basic(tmp_path):
    """Defaults: one Basic call per judgment, no judge_detail key, legacy facts in the system prompt."""
    cfg = _cfg(tmp_path, {})
    j = Journal.init(tmp_path / "run", cfg)
    add_ideas(j, _tagged(5))
    llm = MockLLM(cfg["llm"], seed=8)
    st = run_until_done(j, cfg, llm)
    assert len(llm.calls) == sum(len(i["surrogate_scores"]) for i in st["ideas"].values())
    for _, m in llm.calls:
        assert m[1]["content"] == P.judge_prompt(re.search(r"Idea number \d+", m[1]["content"]).group(0))
        assert "Standard ResNet-18" in m[0]["content"]
    assert all("judge_detail" not in i for i in st["ideas"].values())


# ----------------------------------------------------------------------------- e2e: orchestrator + every judge mode
@pytest.mark.parametrize("jc", [
    {"strategy": "basic_k", "k": 2},
    {"strategy": "rubric", "anchor_facts": False, "facts": ["judge-only fact XYZ"]},
    {"strategy": "fewshot", "fewshot_n": 3},
    {"strategy": "ranking", "rank_batch": 3, "rank_max_wait_s": 0.5},
], ids=lambda d: d["strategy"])
def test_e2e_mock_each_judge_mode(tmp_path, jc):
    import json
    import yaml
    import test_e2e_mock as E
    p = E._cfg(tmp_path, f"e2e_{jc['strategy']}", "optimized", 10)
    d = yaml.safe_load(p.read_text())
    d["judge"] = jc
    p.write_text(yaml.safe_dump(d))
    assert E._run(E._argv(tmp_path, [p])) == 0
    rd = tmp_path / "runs" / f"e2e_{jc['strategy']}"
    st = E._check_run(rd, 10)
    scored = [i for i in st["ideas"].values() if i["status"] == "scored"]
    assert all(i["judge_detail"][0]["strategy"] == jc["strategy"] for i in scored)
    if "facts" in jc:   # judge-only facts reach the judge, never the ideator
        for i in st["ideas"].values():
            assert all("judge-only fact XYZ" not in t for t in i["prompt"]["turns"])
        assert "judge-only fact XYZ" in json.dumps(st["config"]["judge"])
