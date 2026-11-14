"""Prompt sampler: features, gamma-weighted Bayesian linear posterior, Thompson draws.

Spec: "Prompt sampler" section; interfaces per CONTRACTS.md.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve

FEATURES = ["intercept", "BetterGeneric", "BestSample1", "BestSample2", "RandSample",
            "NewSampleCritique", "TopicIdea", "NoTag", "UseTag", "NoIdea", "References", "RefIdeas"]
BASES = ["Generic", "BetterGeneric", "BestSample1", "BestSample2", "RandSample",
         "NewSampleCritique", "TopicIdea"]
NON_GENERIC_BASES = BASES[1:]
TAG_OPTIONS = ["NoTag", "UseTag"]
_IDX = {f: i for i, f in enumerate(FEATURES)}
D = len(FEATURES)


def generic_config() -> dict:
    return {"base": "Generic", "tag_slot": None, "NoIdea": False, "References": False, "RefIdeas": False}


def featurize(cfg: dict) -> np.ndarray:
    """0/1 feature vector (12,). Generic is the reference base (no feature)."""
    x = np.zeros(D)
    x[0] = 1.0
    base = cfg.get("base", "Generic")
    if base not in BASES:
        raise ValueError(f"unknown base {base!r}")
    if base != "Generic":
        x[_IDX[base]] = 1.0
    slot = cfg.get("tag_slot")
    if slot is not None:
        if slot not in TAG_OPTIONS:
            raise ValueError(f"unknown tag_slot {slot!r}")
        x[_IDX[slot]] = 1.0
    for k in ("NoIdea", "References", "RefIdeas"):
        if cfg.get(k):
            x[_IDX[k]] = 1.0
    return x


def observations(state: dict) -> tuple[np.ndarray, np.ndarray]:
    """X (n,12), y (n,) from ideas with F not None and not failed, ordered by scored_seq."""
    obs = [i for i in state.get("ideas", {}).values()
           if i.get("F") is not None and i.get("status") != "failed"]
    obs.sort(key=lambda i: (i.get("scored_seq") if i.get("scored_seq") is not None else float("inf")))
    if not obs:
        return np.zeros((0, D)), np.zeros(0)
    X = np.stack([featurize(i["prompt"]["config"]) for i in obs])
    y = np.array([float(i["F"]) for i in obs])
    return X, y


def fit_posterior(X, y, lambda0: float, lam: float, sigma: float, gamma: float):
    """Weighted Bayesian linear regression posterior. Observation j (0-based, oldest first)
    gets weight gamma^(n-1-j). Returns mu (12,), Sigma (12,12). n=0 gives the prior."""
    X = np.asarray(X, dtype=float).reshape(-1, D)
    y = np.asarray(y, dtype=float).reshape(-1)
    n = X.shape[0]
    w = gamma ** np.arange(n - 1, -1, -1, dtype=float) if n else np.zeros(0)
    prior_var = np.full(D, lam ** 2, dtype=float)
    prior_var[0] = lambda0 ** 2
    s2 = sigma ** 2
    A = (X.T * w) @ X / s2 + np.diag(1.0 / prior_var)
    rhs = (X.T * w) @ y / s2
    c = cho_factor(A, lower=True)
    Sigma = cho_solve(c, np.eye(D))
    Sigma = 0.5 * (Sigma + Sigma.T)
    mu = cho_solve(c, rhs)
    return mu, Sigma


def refit_log(mu, Sigma, n: int) -> dict:
    """Journal log entry for one refit."""
    return {"n": int(n), "mu": [float(v) for v in mu],
            "sd": [float(v) for v in np.sqrt(np.clip(np.diag(Sigma), 0, None))]}


def availability(state: dict) -> dict:
    judged = any(i.get("s_cal") is not None and i.get("status") != "failed"
                 for i in state.get("ideas", {}).values())
    return {"tags": len(state.get("tags", {})) > 0,
            "judged_ideas": judged,
            "references": len((state.get("config") or {}).get("references") or []) > 0}


def _chol_psd(Sigma: np.ndarray) -> np.ndarray:
    S = 0.5 * (Sigma + Sigma.T)
    jitter = 0.0
    for _ in range(8):
        try:
            return np.linalg.cholesky(S + jitter * np.eye(S.shape[0]))
        except np.linalg.LinAlgError:
            jitter = max(jitter * 10, 1e-12 * max(1.0, float(np.trace(S)) / S.shape[0]))
    raise np.linalg.LinAlgError("Sigma not positive definite")


def config_from_draw(b: np.ndarray, avail: dict) -> dict:
    """Map one sampled effect vector to a SamplerConfig, honoring availability."""
    best_base, best_val = "Generic", 0.0
    for name in NON_GENERIC_BASES:  # all bases are always available
        v = b[_IDX[name]]
        if v > best_val:
            best_base, best_val = name, v
    tag_slot = None
    if avail.get("tags"):
        best = 0.0
        for opt in TAG_OPTIONS:
            if b[_IDX[opt]] > best:
                tag_slot, best = opt, b[_IDX[opt]]
    judged = bool(avail.get("judged_ideas"))
    return {"base": best_base, "tag_slot": tag_slot,
            "NoIdea": bool(judged and b[_IDX["NoIdea"]] > 0),
            "References": bool(avail.get("references") and b[_IDX["References"]] > 0),
            "RefIdeas": bool(judged and b[_IDX["RefIdeas"]] > 0)}


def uniform_config(avail: dict, rng: np.random.Generator) -> dict:
    """Naive-arm draw from the same prompt pool, never learning: the spec's intended starting distribution
    (every base equally likely; each optional slot used half the time, subject to availability)."""
    judged = bool(avail.get("judged_ideas"))
    tag_slot = None
    if avail.get("tags") and rng.random() < 0.5:
        tag_slot = TAG_OPTIONS[int(rng.integers(len(TAG_OPTIONS)))]
    return {"base": BASES[int(rng.integers(len(BASES)))], "tag_slot": tag_slot,
            "NoIdea": bool(judged and rng.random() < 0.5),
            "References": bool(avail.get("references") and rng.random() < 0.5),
            "RefIdeas": bool(judged and rng.random() < 0.5)}


def draw_batch(mu, Sigma, avail: dict, n: int, rng: np.random.Generator,
               mode: str = "thompson") -> list[tuple[dict, list[float] | None]]:
    """n prompt configs. thompson: each from its own posterior draw b~ = mu + chol(Sigma) z.
    uniform (naive arm): uniform over the same prompt pool, no learning. fixed_generic: Generic only
    (kept for reference; it confounds the arm comparison with prompt choice, so it is not the naive arm)."""
    if mode == "fixed_generic":
        return [(generic_config(), None) for _ in range(n)]
    if mode == "uniform":
        return [(uniform_config(avail, rng), None) for _ in range(n)]
    if mode != "thompson":
        raise ValueError(f"unknown sampler mode {mode!r}")
    mu = np.asarray(mu, dtype=float)
    L = _chol_psd(np.asarray(Sigma, dtype=float))
    out = []
    for _ in range(n):
        b = mu + L @ rng.standard_normal(D)
        out.append((config_from_draw(b, avail), [float(v) for v in b]))
    return out
