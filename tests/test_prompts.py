import re
from pathlib import Path

import numpy as np
import pytest

from ar import prompts as P
from ar.templating import RenderContext

SPEC = Path(__file__).resolve().parents[1] / "class_notes" / "Autoresearch Preview Spec.md"


def spec_blocks():
    t = SPEC.read_text()
    return [b for _, lang, b in re.findall(r"^(`{3,4})(\w*)\n(.*?)\n\1$", t, re.S | re.M) if lang == ""]


@pytest.fixture(scope="module")
def blocks():
    if not SPEC.exists():
        pytest.skip("spec not present")
    return spec_blocks()


def test_all_strings_verbatim_against_spec(blocks):
    b = P.BASE_PROMPTS
    expected = [
        P.IDEATOR_SYSTEM, None, P.SUMMARY,
        b["Generic"][0], b["BetterGeneric"][0],
        *b["BestSample1"], *b["BestSample2"], *b["RandSample"], *b["NewSampleCritique"], *b["TopicIdea"],
        P.APPENDS["References"]["top"], P.APPENDS["References"]["bottom"],
        P.APPENDS["RefIdeas"]["top"], P.APPENDS["RefIdeas"]["bottom"],
        P.APPENDS["NoTag"]["bottom"], P.APPENDS["UseTag"]["bottom"], P.APPENDS["NoIdea"]["bottom"],
        P.TAGGING_SYSTEM, P.GENERATOR_GENERIC, P.MERGER_GENERIC, P.WRITER_GENERIC, P.BASIC_ASSIGNMENT,
        P.JUDGE_SYSTEM, P.BASIC_SURROGATE,
    ]
    assert len(blocks) == len(expected)
    for got, want in zip(expected, blocks):
        if got is not None:      # block 1 is the assembly layout, not a prompt
            assert got == want


def test_exact_strings():
    assert P.BASE_PROMPTS["Generic"] == ["Come up with a new idea. Output the full idea description document and nothing else."]
    assert P.BASE_PROMPTS["RandSample"][1] == "Take idea #{{rangeint;1;5}}, discard everything else, and write the full idea description document and nothing else."
    assert P.APPENDS["NoTag"]["bottom"] == "Ensure any ideas do not relate with the following tags {{tags;{{rangefloat;0.1;1}}}}."
    assert P.BASIC_SURROGATE.endswith("Output the expected performance of the idea as a float between 0.0 and 1.0 within \\boxed{}")
    assert '{"new_tags": \n[\n{"name":"new_tag_1_name_here", "description":"new_tag_1_description_here"}, \n' in P.WRITER_GENERIC
    assert P.BASES == ["Generic", "BetterGeneric", "BestSample1", "BestSample2", "RandSample", "NewSampleCritique", "TopicIdea"]
    assert set(P.BASES) == set(P.BASE_PROMPTS)


def ctx():
    return RenderContext(
        prompt="THE TASK",
        references=["paper A", "paper B", "paper C", "paper D"],
        facts=["f1", "f2"],
        tags=[(f"t{i}", f"d{i}") for i in range(6)],
        judged_ideas=[{"id": f"i{i}", "summary": f"s{i}", "s_cal": 0.5, "real_score": None} for i in range(4)],
    )


def cfg(base="Generic", tag_slot=None, NoIdea=False, References=False, RefIdeas=False):
    return {"base": base, "tag_slot": tag_slot, "NoIdea": NoIdea, "References": References, "RefIdeas": RefIdeas}


def test_generic_no_appends():
    out = P.assemble(cfg(), ctx(), np.random.default_rng(0))
    assert out["turns"] == [P.BASE_PROMPTS["Generic"][0]]
    assert out["system"].startswith(P.IDEATOR_SYSTEM.split("{{prompt}}")[0] + "THE TASK\n")
    assert "- f1\n- f2" in out["system"]
    assert [d["where"] for d in out["draws"]] == ["system"]


@pytest.mark.parametrize("tag_slot", ["NoTag", "UseTag"])
def test_all_appends_spec_order(tag_slot):
    out = P.assemble(cfg("BestSample1", tag_slot, True, True, True), ctx(), np.random.default_rng(3))
    t1 = out["turns"][0]
    parts = t1.split("\n\n")
    assert parts[0].startswith("Here are some scientific papers that may or may not be relevant:\n- paper")
    assert parts[1].startswith("Here are some previously generated ideas for reference, with their evaluation results:\n- s")
    assert re.fullmatch(r"Come up with [2-5] new ideas, one after another\. They should all be meaningfully different\.", parts[2])
    assert parts[3] == P.APPENDS["References"]["bottom"]
    assert parts[4] == P.APPENDS["RefIdeas"]["bottom"]
    assert parts[5].startswith(P.APPENDS[tag_slot]["bottom"].split("{{")[0] + "- t")
    assert parts[6].startswith("Ensure any idea does not relate with the following ideas - s") and parts[6].endswith(".")
    assert len(parts) == 7
    assert out["turns"][1] == P.BASE_PROMPTS["BestSample1"][1]     # appends on turn 1 only
    utils = [(d["where"], d["utility"]) for d in out["draws"]]
    inner = "rangeint" if tag_slot == "UseTag" else "rangefloat"
    assert utils == [("system", "facts"),
                     ("turn1", "rangeint"), ("turn1", "references"),
                     ("turn1", "rangeint"), ("turn1", "idea_summaries_and_critiques"),
                     ("turn1", "rangeint"),                      # BestSample1 count
                     ("turn1", inner), ("turn1", "tags"),
                     ("turn1", "rangefloat"), ("turn1", "idea_summaries")]


def test_randsample_turn2_draw_recorded():
    out = P.assemble(cfg("RandSample"), ctx(), np.random.default_rng(7))
    d = [x for x in out["draws"] if x["where"] == "turn2"]
    assert len(d) == 1 and d[0]["utility"] == "rangeint" and 1 <= d[0]["value"] <= 5
    assert out["turns"][1] == f"Take idea #{d[0]['value']}, discard everything else, and write the full idea description document and nothing else."
    assert out["turns"][0] == P.BASE_PROMPTS["RandSample"][0]


def test_assemble_deterministic():
    c = cfg("TopicIdea", "UseTag", True, True, True)
    a = P.assemble(c, ctx(), np.random.default_rng(42))
    b = P.assemble(c, ctx(), np.random.default_rng(42))
    assert a == b


def test_unavailable_append_raises():
    c = ctx()
    c.tags = []
    with pytest.raises(ValueError):
        P.assemble(cfg(tag_slot="NoTag"), c, np.random.default_rng(0))
    with pytest.raises(ValueError):
        P.assemble(cfg(base="Nope"), ctx(), np.random.default_rng(0))


def test_helpers():
    evil = "idea with {{prompt}} and }} {{"
    assert P.summary_prompt(evil) == P.SUMMARY.replace("{{idea}}", evil)
    assert P.assign_prompt("X") == "Here is a new idea being considered: X\n\nOutput the list of tags by name, exact, from the source list"
    assert P.judge_prompt(evil) == P.BASIC_SURROGATE.replace("{{idea}}", evil)
    ts = P.tagging_system("TASK", [("a", "desc a"), ("b", "desc b")])
    assert ts.endswith("(new ones are added over time):\n- a: desc a\n- b: desc b") and "prompt:\nTASK\n" in ts
    g = P.generator_prompt(["s1", "s2\nmore"])
    assert g.startswith("Here is a new list of ideas that are being considered:\n- s1\n- s2 more\n\nOutput all")
    m = P.merger_prompt(["s1"], ["- x: y", "- z: w"])
    assert "Here are 2 proposed lists of tags" in m and "List 1:\n- x: y\n\nList 2:\n- z: w\n\nDeduplicate" in m
    w = P.writer_prompt(["s1"], ["- x: y"])
    assert "Here is a list of proposed tags: \nList 1:\n- x: y\n\nOutput a json" in w
    js = P.judge_system("TASK", ["f1"], np.random.default_rng(0))
    assert js.endswith("prompt:\nTASK\n\nAdditionally, reference these known facts from previous research rounds:\n- f1")


@pytest.mark.parametrize("base", P.BASES)
def test_every_base_renders_without_braces(base):
    out = P.assemble(cfg(base, "UseTag", True, True, True), ctx(), np.random.default_rng(1))
    assert len(out["turns"]) == len(P.BASE_PROMPTS[base])
    for t in [out["system"], *out["turns"]]:
        assert "{{" not in t and "}}" not in t
