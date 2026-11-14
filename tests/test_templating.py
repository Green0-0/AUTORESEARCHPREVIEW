import numpy as np
import pytest

from ar.templating import RenderContext, TemplateError, list_count, parse, render, Node, Text


def ctx(**kw):
    base = dict(
        prompt="TASK PROMPT",
        references=[f"ref {i}" for i in range(6)],
        facts=["fact a", "fact b", "fact c"],
        tags=[(f"tag{i}", f"desc {i}") for i in range(10)],
        judged_ideas=[{"id": f"i{i:06d}", "summary": f"summary {i}", "s_cal": i / 10,
                       "real_score": (0.41 if i == 2 else None)} for i in range(5)],
    )
    base.update(kw)
    return RenderContext(**base)


def rng(seed=0):
    return np.random.default_rng(seed)


def test_parse_nested_ast():
    ast = parse("a {{references;{{rangeint;1;3}}}} b")
    assert isinstance(ast[0], Text) and ast[0].text == "a "
    node = ast[1]
    assert isinstance(node, Node) and node.name == "references"
    inner = node.args[0][0]
    assert inner.name == "rangeint" and [a[0].text for a in inner.args] == ["1", "3"]
    assert ast[2].text == " b"


@pytest.mark.parametrize("bad", ["{{prompt", "x }} y", "{{}}", "{{a;{{b}}"])
def test_unbalanced_template_errors(bad):
    with pytest.raises(TemplateError):
        render(bad, ctx(), rng())


def test_unknown_utility_errors():
    with pytest.raises(TemplateError, match="unknown template utility"):
        render("{{nope;3}}", ctx(), rng())


def test_nested_references_rangeint():
    for seed in range(20):
        out, draws = render("{{references;{{rangeint;1;3}}}}", ctx(), rng(seed))
        assert draws[0]["utility"] == "rangeint"
        k = draws[0]["value"]
        assert 1 <= k <= 3
        assert draws[1]["utility"] == "references" and draws[1]["args"] == [str(k)]
        ids = draws[1]["sampled_ids"]
        assert len(ids) == k and ids == sorted(ids)            # creation order
        assert out == "\n".join(f"- ref {i}" for i in ids)


def test_nested_tags_rangefloat_fraction():
    for seed in range(20):
        out, draws = render("{{tags;{{rangefloat;0.1;1}}}}", ctx(), rng(seed))
        f = draws[0]["value"]
        assert 0.1 <= f <= 1 and "." in draws[1]["args"][0]
        names = draws[1]["sampled_ids"]
        assert len(names) == max(1, int(f * 10 + 0.5))
        lines = out.split("\n")
        assert lines == [f"- {n}: desc {n[3:]}" for n in names]
        assert [int(n[3:]) for n in names] == sorted(int(n[3:]) for n in names)


def test_tags_no_arg_full_pool():
    out, draws = render("{{tags}}", ctx(), rng())
    assert out.split("\n") == [f"- tag{i}: desc {i}" for i in range(10)]
    assert draws[0]["sampled_ids"] == [f"tag{i}" for i in range(10)]


def test_list_count_semantics():
    assert list_count("3", 10) == 3
    assert list_count("30", 10) == 10          # capped
    assert list_count("0.25", 10) == 3         # round half up
    assert list_count("0.01", 10) == 1         # min 1
    assert list_count("1.0", 4) == 4           # decimal = fraction
    assert list_count("1", 4) == 1             # integer
    assert list_count("5", 0) == 0
    assert list_count(None, 7) == 7
    with pytest.raises(TemplateError):
        list_count("abc", 3)


def test_facts_1000_all_in_order_and_empty():
    out, d = render("{{facts;1000}}", ctx(), rng())
    assert out == "- fact a\n- fact b\n- fact c"
    out, d = render("F:\n{{facts;1000}}", ctx(facts=[]), rng())
    assert out == "F:\n" and d[0]["sampled_ids"] == []


def test_critiques_format():
    out, d = render("{{idea_summaries_and_critiques;10}}", ctx(), rng())
    lines = out.split("\n")
    assert lines[0] == "- summary 0 (surrogate: 0.00)"
    assert lines[2] == "- summary 2 (surrogate: 0.20; real: 0.41)"
    assert d[0]["sampled_ids"] == [f"i{i:06d}" for i in range(5)]
    out, _ = render("{{idea_summaries;2}}", ctx(), rng(1))
    assert all(l.startswith("- summary ") for l in out.split("\n")) and len(out.split("\n")) == 2


def test_inert_substitution():
    evil = "use {{prompt}} and {{rangeint;1;9}} and }} unbalanced {{ x ; y"
    c = ctx(prompt=evil, tags=[("t{{prompt}}", "d}}{{")],
            judged_ideas=[{"id": "i1", "summary": "{{facts;3}} }}", "s_cal": 0.5, "real_score": None}])
    out, draws = render("P={{prompt}}|{{tags}}|{{idea_summaries;1}}", c, rng())
    assert out == f"P={evil}|- t{{{{prompt}}}}: d}}}}{{{{|- {{{{facts;3}}}} }}}}"
    assert [d["utility"] for d in draws] == ["tags", "idea_summaries"]   # nothing re-parsed
    out, _ = render("X {{idea}} Y", c, None, variables={"idea": "{{prompt}} {{"})
    assert out == "X {{prompt}} {{ Y"


def test_choice_and_rangeint_bounds():
    seen = set()
    for s in range(50):
        out, d = render("{{choice;a;b;c}}-{{rangeint;2;5}}", ctx(), rng(s))
        c, n = out.split("-")
        assert c in "abc" and 2 <= int(n) <= 5
        assert d[0] == {"utility": "choice", "args": ["a", "b", "c"], "value": c}
        seen.add(int(n))
    assert seen == {2, 3, 4, 5}


def test_determinism_seeded():
    t = "{{references;{{rangeint;1;3}}}}\n{{tags;{{rangefloat;0.1;1}}}}\n{{idea_summaries;0.5}}\n{{choice;x;y}}"
    a = render(t, ctx(), rng(123))
    b = render(t, ctx(), rng(123))
    assert a == b
    outs = {render(t, ctx(), rng(s))[0] for s in range(10)}
    assert len(outs) > 1


def test_no_rng_needed_when_deterministic():
    out, _ = render("{{prompt}} {{tags}} {{facts;1000}}", ctx(), None)
    assert out.startswith("TASK PROMPT - tag0")
    with pytest.raises(TemplateError):
        render("{{rangeint;1;2}}", ctx(), None)


def test_fractional_list_size_is_capped():
    from ar.templating import list_count, MAX_FRACTION_ITEMS
    assert list_count("1.0000", 1000) == MAX_FRACTION_ITEMS
    assert list_count("0.1000", 100) == 10          # below the cap: unchanged
    assert list_count("500", 1000) == 500           # integer N is not capped
    assert list_count(None, 1000) == 1000           # full list is not capped
