import numpy as np
import pytest

from ar import sampler as S
from ar.sampler import FEATURES, BASES, featurize, fit_posterior, draw_batch, observations, availability, refit_log

ALL = {"tags": True, "judged_ideas": True, "references": True}
NONE = {"tags": False, "judged_ideas": False, "references": False}


def prior(l0=0.1, l=0.1):
    return fit_posterior(np.zeros((0, 12)), np.zeros(0), l0, l, 0.1, 0.99)


def test_featurize():
    x = featurize({"base": "Generic", "tag_slot": None, "NoIdea": False, "References": False, "RefIdeas": False})
    assert x.shape == (12,) and x.sum() == 1 and x[0] == 1
    x = featurize({"base": "TopicIdea", "tag_slot": "UseTag", "NoIdea": True, "References": True, "RefIdeas": True})
    on = {FEATURES[i] for i in np.nonzero(x)[0]}
    assert on == {"intercept", "TopicIdea", "UseTag", "NoIdea", "References", "RefIdeas"}
    with pytest.raises(ValueError):
        featurize({"base": "Nope", "tag_slot": None})


def test_prior_only():
    mu, Sig = prior(0.2, 0.3)
    assert np.allclose(mu, 0)
    assert np.allclose(np.diag(Sig), [0.04] + [0.09] * 11)
    log = refit_log(mu, Sig, 0)
    assert log["n"] == 0 and len(log["mu"]) == 12 and np.isclose(log["sd"][1], 0.3)


def test_prior_draws_uniform():
    mu, Sig = prior()
    rng = np.random.default_rng(0)
    N = 14000
    res = draw_batch(mu, Sig, ALL, N, rng)
    bases = [c["base"] for c, _ in res]
    counts = np.array([bases.count(b) for b in BASES])
    # P(Generic) = P(all 6 draws <= 0) = 1/64; others share the rest equally.
    assert abs(counts[0] / N - 1 / 64) < 0.006
    others = counts[1:] / N
    assert np.all(np.abs(others - (1 - 1 / 64) / 6) < 0.015)
    tags = [c["tag_slot"] for c, _ in res]
    # none w.p. 1/4, NoTag 3/8, UseTag 3/8 -> slot used 75%; each option ~50% "drawn positive"
    assert abs(tags.count(None) / N - 0.25) < 0.015
    assert abs(tags.count("NoTag") / N - 0.375) < 0.015
    for k in ("NoIdea", "References", "RefIdeas"):
        assert abs(np.mean([c[k] for c, _ in res]) - 0.5) < 0.015
    assert all(len(b) == 12 for _, b in res)


def test_availability_masks():
    mu = np.full(12, 5.0)  # everything strongly positive
    Sig = np.eye(12) * 1e-6
    rng = np.random.default_rng(1)
    for c, _ in draw_batch(mu, Sig, NONE, 200, rng):
        assert c["tag_slot"] is None and not c["NoIdea"] and not c["References"] and not c["RefIdeas"]
        assert c["base"] != "Generic"
    for c, _ in draw_batch(mu, Sig, {"tags": True, "judged_ideas": False, "references": False}, 50, rng):
        assert c["tag_slot"] is not None and not c["NoIdea"] and not c["RefIdeas"] and not c["References"]
    for c, _ in draw_batch(*prior(1, 1), {"tags": False, "judged_ideas": True, "references": False}, 500, rng):
        assert c["tag_slot"] is None and not c["References"]
    for c, _ in draw_batch(mu, Sig, ALL, 50, rng):
        assert c["NoIdea"] and c["References"] and c["RefIdeas"]


def test_argmax_and_threshold():
    mu = np.zeros(12)
    mu[FEATURES.index("BestSample2")] = 1.0
    mu[FEATURES.index("RandSample")] = 0.5
    mu[FEATURES.index("UseTag")] = 0.3
    mu[FEATURES.index("NoTag")] = 0.6
    mu[FEATURES.index("NoIdea")] = -1
    mu[FEATURES.index("References")] = -1
    mu[FEATURES.index("RefIdeas")] = 1
    for c, b in draw_batch(mu, np.eye(12) * 1e-8, ALL, 5, np.random.default_rng(2)):
        assert c == {"base": "BestSample2", "tag_slot": "NoTag", "NoIdea": False, "References": False, "RefIdeas": True}
    neg = -np.ones(12)
    c, _ = draw_batch(neg, np.eye(12) * 1e-8, ALL, 1, np.random.default_rng(2))[0]
    assert c == S.generic_config()


def test_fixed_generic():
    res = draw_batch(np.ones(12), np.eye(12), ALL, 4, np.random.default_rng(0), mode="fixed_generic")
    assert res == [(S.generic_config(), None)] * 4


def _synthetic(n, b_true, rng, noise=0.05):
    cfgs, _ = zip(*draw_batch(*prior(1, 1), ALL, n, rng))
    X = np.stack([featurize(c) for c in cfgs])
    y = X @ b_true + noise * rng.standard_normal(n)
    return X, y


def test_posterior_recovers_effects():
    rng = np.random.default_rng(3)
    b_true = rng.uniform(-0.15, 0.15, 12)
    X, y = _synthetic(3000, b_true, rng)
    mu_small, Sig_small = fit_posterior(X[:30], y[:30], 0.1, 0.1, 0.1, 1.0)
    mu, Sig = fit_posterior(X, y, 0.1, 0.1, 0.1, 1.0)
    assert np.max(np.abs(mu - b_true)) < 0.03
    sd, sd_small, sd0 = (np.sqrt(np.diag(s)) for s in (Sig, Sig_small, prior()[1]))
    assert np.all(sd < sd_small) and np.all(sd_small < sd0)
    # matches closed form computed independently
    A = X.T @ X / 0.01 + np.eye(12) / 0.01
    assert np.allclose(mu, np.linalg.solve(A, X.T @ y / 0.01))


def test_gamma_weighting():
    rng = np.random.default_rng(4)
    b_old, b_new = np.zeros(12), np.zeros(12)
    b_old[FEATURES.index("UseTag")] = 0.3
    b_new[FEATURES.index("UseTag")] = -0.3
    X1, y1 = _synthetic(800, b_old, rng, 0.01)
    X2, y2 = _synthetic(800, b_new, rng, 0.01)
    X, y = np.vstack([X1, X2]), np.concatenate([y1, y2])
    k = FEATURES.index("UseTag")
    mu_flat, _ = fit_posterior(X, y, 0.1, 1.0, 0.1, 1.0)
    mu_disc, Sig_disc = fit_posterior(X, y, 0.1, 1.0, 0.1, 0.98)
    assert abs(mu_flat[k]) < 0.05          # average of both regimes
    assert abs(mu_disc[k] - (-0.3)) < 0.03  # recent regime dominates
    # explicit weights: gamma^(n-1-j) equals an explicitly weighted fit
    n = len(y); w = 0.98 ** (n - 1 - np.arange(n))
    lam = np.diag([0.01] + [1.0] * 11)
    A = X.T @ (w[:, None] * X) / 0.01 + np.linalg.inv(lam)
    assert np.allclose(mu_disc, np.linalg.solve(A, X.T @ (w * y) / 0.01))
    assert np.allclose(Sig_disc, np.linalg.inv(A))


def _idea(i, F, seq, status="scored", base="Generic", s_cal=0.5):
    return {"id": f"i{i:06d}", "created_at": float(i), "status": status, "F": F, "scored_seq": seq, "s_cal": s_cal,
            "prompt": {"config": {"base": base, "tag_slot": None, "NoIdea": False, "References": False, "RefIdeas": False}}}


def test_observations_and_availability():
    ideas = [_idea(0, 0.1, 2, base="TopicIdea"), _idea(1, 0.2, 0, base="RandSample"),
             _idea(2, None, None, status="tagged", s_cal=None), _idea(3, 0.3, 1, status="failed"),
             _idea(4, 0.0, 1, base="BestSample1")]
    st = {"ideas": {d["id"]: d for d in ideas}, "tags": {}, "config": {"references": []}}
    X, y = observations(st)
    assert list(y) == [0.2, 0.0, 0.1]
    assert X[0, FEATURES.index("RandSample")] == 1 and X[2, FEATURES.index("TopicIdea")] == 1
    assert availability(st) == {"tags": False, "judged_ideas": True, "references": False}
    X0, y0 = observations({"ideas": {}})
    assert X0.shape == (0, 12) and y0.shape == (0,)
    st2 = {"ideas": {"a": _idea(2, None, None, status="failed", s_cal=0.4)}, "tags": {"t": {}}, "config": {"references": ["p"]}}
    assert availability(st2) == {"tags": True, "judged_ideas": False, "references": True}


def test_uniform_naive_matches_spec_start_distribution():
    rng = np.random.default_rng(0)
    res = draw_batch(np.ones(12), np.eye(12), ALL, 14000, rng, mode="uniform")
    cfgs = [c for c, b in res]
    assert all(b is None for _, b in res)
    from collections import Counter
    bases = Counter(c["base"] for c in cfgs)
    assert set(bases) == set(BASES) and all(abs(v / 14000 - 1 / 7) < 0.02 for v in bases.values())
    slots = Counter(c["tag_slot"] for c in cfgs)
    assert abs(slots[None] / 14000 - 0.5) < 0.02 and abs(slots["NoTag"] / 14000 - 0.25) < 0.02
    for k in ("NoIdea", "References", "RefIdeas"):
        assert abs(sum(c[k] for c in cfgs) / 14000 - 0.5) < 0.02
    none = draw_batch(np.ones(12), np.eye(12), {"tags": False, "judged_ideas": False, "references": False},
                      200, rng, mode="uniform")
    assert all(c["tag_slot"] is None and not (c["NoIdea"] or c["References"] or c["RefIdeas"]) for c, _ in none)
