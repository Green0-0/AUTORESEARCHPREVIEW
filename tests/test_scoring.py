import pytest

from ar.scoring import parse_boxed, calibrate, compute_F


@pytest.mark.parametrize("text,val", [
    (r"score \boxed{0.7}", 0.7),
    (r"\boxed{ .65 }", 0.65),
    (r"\boxed{0.70/1}", 0.70),
    (r"first \boxed{0.2} then final \boxed{0.9}", 0.9),
    (r"\boxed{1.5}", 1.0),
    (r"\boxed{-0.3}", 0.0),
    (r"\boxed{70\%}", 0.7),
    (r"\boxed{\frac{3}{4}}", 0.75),
    (r"\boxed {0.4}", 0.4),
    (r"\boxed{\text{0.55}}", 0.55),
    (r"\boxed{1}", 1.0),
])
def test_parse_boxed_ok(text, val):
    assert parse_boxed(text) == pytest.approx(val)


@pytest.mark.parametrize("text", ["", "no box 0.7", r"\boxed{}", r"\boxed{high}", r"\boxed{0.8} \boxed{n/a}",
                                  r"\boxed{0.5", r"\boxed{\frac{1}{0}}", None])
def test_parse_boxed_none(text):
    assert parse_boxed(text) is None


def test_calibrate():
    assert calibrate(0.6, {"a": 0.0, "b": 1.0}) == pytest.approx(0.6)
    assert calibrate(0.6, {"a": 0.1, "b": 0.5}) == pytest.approx(0.4)
    assert calibrate(0.9, {"a": 0.5, "b": 1.0}) == 1.0
    assert calibrate(0.1, {"a": -0.5, "b": 1.0}) == 0.0


def mk(ideas):
    return {"ideas": {i["id"]: i for i in ideas}}


def I(id, t, tags, s_cal, status="scored"):
    return {"id": id, "created_at": t, "tags": tags, "s_cal": s_cal, "status": status}


def test_no_earlier_ideas_uses_s_base():
    st = mk([I("a", 1.0, ["x", "y"], None, "judged")])
    F, d = compute_F(st, "a", 0.7, 0.05, 0.5)
    assert F == pytest.approx(0.15) and d["b_t"] == {"x": 0.5, "y": 0.5}
    F, _ = compute_F(st, "a", 0.52, 0.05, 0.5)
    assert F == 0.0


def test_multiple_tags_max():
    st = mk([I("p", 1.0, ["x"], 0.8), I("q", 2.0, ["y"], 0.6), I("q2", 2.5, ["y"], 0.65),
             I("a", 3.0, ["x", "y", "z"], None, "judged")])
    F, d = compute_F(st, "a", 0.75, 0.05, 0.5)
    assert d["b_t"] == {"x": 0.8, "y": 0.65, "z": 0.5}
    assert F == pytest.approx(0.2) and d["best_tag"] == "z"   # z: 0.75-0.5-0.05
    st["ideas"]["a"]["tags"] = ["x", "y"]
    F, d = compute_F(st, "a", 0.75, 0.05, 0.5)
    assert F == pytest.approx(0.05) and d["best_tag"] == "y"


def test_excludes_failed_later_unscored_self():
    st = mk([I("f", 1.0, ["x"], 0.95, "failed"), I("u", 1.5, ["x"], None, "tagged"),
             I("a", 2.0, ["x"], 0.9, "judged"), I("later", 3.0, ["x"], 0.99),
             I("same", 2.0, ["x"], 0.99)])
    F, d = compute_F(st, "a", 0.9, 0.05, 0.5)
    assert d["b_t"] == {"x": 0.5}
    assert F == pytest.approx(0.35)


def test_frozen_b_t_reuse():
    st = mk([I("p", 1.0, ["x"], 0.6), I("a", 2.0, ["x"], None, "judged")])
    F1, d1 = compute_F(st, "a", 0.8, 0.05, 0.5)
    assert F1 == pytest.approx(0.15)
    # meanwhile an earlier-created idea gets a higher score; frozen b_t must ignore it
    st["ideas"]["p"]["s_cal"] = 0.95
    F2, d2 = compute_F(st, "a", 0.7, 0.05, 0.5, b_t=d1["b_t"])
    assert d2["b_t"] == {"x": 0.6} and F2 == pytest.approx(0.05)
    F3, _ = compute_F(st, "a", 0.7, 0.05, 0.5)
    assert F3 == 0.0


def test_no_tags_gives_none():
    st = mk([I("a", 1.0, [], None, "judged")])
    F, d = compute_F(st, "a", 0.9, 0.05, 0.5)
    assert F is None and d["b_t"] == {}
