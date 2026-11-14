"""Run configuration: YAML -> validated dict with defaults (see CONTRACTS.md, "Config")."""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(os.environ.get("AR_REPO") or Path(__file__).resolve().parents[1])

ARMS = ("optimized", "naive")
SAMPLER_MODES = ("thompson", "uniform", "fixed_generic")
JUDGE_STRATEGIES = ("basic", "basic_k", "rubric", "fewshot", "ranking")   # J1, J2, J3, J4, J5 (plan §9.1)
FEWSHOT_SOURCES = ("surrogate", "real")

DEFAULTS: dict[str, Any] = {
    "run_id": None,                 # required
    "task_id": None,                # required
    "arm": "optimized",
    "seed": 1,
    "task_prompt_file": None,       # default tasks/<task_id>/TASK.md
    "references": [],
    "facts": [],
    "facts_file": None,
    "llm": {
        "base_url": "http://127.0.0.1:8000/v1",
        "model": "qwen",
        "max_inflight": 64,
        "purposes": {
            "ideation": {"thinking": True, "max_tokens": 16000, "temperature": 0.8},
            "summary": {"thinking": False, "max_tokens": 400, "temperature": 0.3},
            "tagging": {"thinking": False, "max_tokens": 4000, "temperature": 0.3},
            "assign": {"thinking": False, "max_tokens": 800, "temperature": 0.2},
            "judge": {"thinking": True, "max_tokens": 12000, "temperature": 0.7},
        },
    },
    "sampler": {"mode": "thompson", "lambda0": 0.1, "lambda": 0.1, "sigma": 0.1, "gamma": 0.99},
    "scoring": {"s_min": 0.05, "s_base": 0.5, "calibration": {"a": 0.0, "b": 1.0}},
    "ideation": {"batch_size": 8, "concurrent_batches": 6, "max_backlog": None},
    "tagging": {"trigger_count": 20, "trigger_minutes": 10, "batch_limit": 100,
                "S": 10, "r": 2, "N": 4, "K": 4, "poll_seconds": 5},
    # judge strategy (plan §9.1). The defaults reproduce the original J1 judge exactly.
    "judge": {
        "retries": 2,
        "strategy": "basic",        # basic (J1) | basic_k (J2) | rubric (J3) | fewshot (J4) | ranking (J5)
        "k": 5,                     # basic_k: samples per judgment (score = their mean)
        "facts": None,              # judge-only facts; null = the top-level `facts` (legacy behaviour)
        "facts_file": None,         # optional YAML list of strings appended to the judge facts
        "anchor_facts": False,      # append the task's starter=0 / reference=1 anchor facts (task.yaml floor/ref)
        "fewshot_source": "surrogate",   # fewshot: surrogate (same-run earlier ideas, s_cal) | real (real results)
        "fewshot_n": 6,             # fewshot: examples per prompt (stratified over the score range)
        "fewshot_runs": [],         # fewshot real: extra run ids (same runs root) whose real results are examples
        "rank_batch": 5,            # ranking: new ideas per call
        "rank_max_wait_s": 60,      # ranking: judge a smaller batch once the oldest waiting idea is this old
        "rank_max_idea_chars": 12000,    # ranking: per-idea description cap (null = none) so 5 ideas fit the context
    },
    "claims": {"timeout_minutes": 30},
    "stop": {"max_hours": 24, "max_ideas": None},
}

# Type specs for leaves. "num" = int or float (not bool); "pos" suffix = > 0; "nn" = >= 0.
_SPEC: dict[str, Any] = {
    "run_id": "str", "task_id": "str", "arm": "str", "seed": "int",
    "task_prompt_file": "str", "references": "strlist", "facts": "strlist",
    "facts_file": "str?",
    "llm": {"base_url": "str", "model": "str", "max_inflight": "int_pos",
            "purposes": "purposes"},
    "sampler": {"mode": "str", "lambda0": "num_pos", "lambda": "num_pos",
                "sigma": "num_pos", "gamma": "num_pos"},
    "scoring": {"s_min": "num_nn", "s_base": "num_nn",
                "calibration": {"a": "num", "b": "num"}},
    "ideation": {"batch_size": "int_pos", "concurrent_batches": "int_pos", "max_backlog": "int_pos?"},
    "tagging": {"trigger_count": "int_pos", "trigger_minutes": "num_pos",
                "batch_limit": "int_pos", "S": "int_pos", "r": "int_pos",
                "N": "int_pos", "K": "int_pos", "poll_seconds": "num_pos"},
    "judge": {"retries": "int_nn", "strategy": "str", "k": "int_pos", "facts": "strlist?",
              "facts_file": "str?", "anchor_facts": "bool", "fewshot_source": "str", "fewshot_n": "int_pos",
              "fewshot_runs": "strlist", "rank_batch": "int_pos", "rank_max_wait_s": "num_nn",
              "rank_max_idea_chars": "int_pos?"},
    "claims": {"timeout_minutes": "num_pos"},
    "stop": {"max_hours": "num_pos?", "max_ideas": "int_pos?"},
}
_PURPOSE_SPEC = {"thinking": "bool", "max_tokens": "int_pos", "temperature": "num_nn"}


class ConfigError(ValueError):
    pass


def _deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _check_leaf(path: str, v: Any, spec: str) -> None:
    optional = spec.endswith("?")
    spec = spec.rstrip("?")
    if v is None:
        if optional:
            return
        raise ConfigError(f"{path}: required, got null")
    kind, _, bound = spec.partition("_")
    if kind == "str":
        ok = isinstance(v, str) and v != ""
    elif kind == "bool":
        ok = isinstance(v, bool)
    elif kind == "int":
        ok = isinstance(v, int) and not isinstance(v, bool)
    elif kind == "num":
        ok = isinstance(v, (int, float)) and not isinstance(v, bool)
    elif kind == "strlist":
        ok = isinstance(v, list) and all(isinstance(x, str) for x in v)
    else:  # pragma: no cover
        raise AssertionError(spec)
    if not ok:
        raise ConfigError(f"{path}: expected {spec}{' or null' if optional else ''}, got {v!r}")
    if bound == "pos" and not v > 0:
        raise ConfigError(f"{path}: must be > 0, got {v!r}")
    if bound == "nn" and not v >= 0:
        raise ConfigError(f"{path}: must be >= 0, got {v!r}")


def _validate(cfg: Any, spec: dict, prefix: str = "") -> None:
    if not isinstance(cfg, dict):
        raise ConfigError(f"{prefix or 'config'}: expected a mapping, got {type(cfg).__name__}")
    unknown = sorted(set(cfg) - set(spec))
    if unknown:
        raise ConfigError(f"{prefix or 'config'}: unknown key(s) {unknown}; allowed: {sorted(spec)}")
    for k, s in spec.items():
        path = f"{prefix}.{k}" if prefix else k
        v = cfg.get(k)
        if isinstance(s, dict):
            _validate(v, s, path)
        elif s == "purposes":
            if not isinstance(v, dict):
                raise ConfigError(f"{path}: expected a mapping")
            for name, p in v.items():
                _validate(p, _PURPOSE_SPEC, f"{path}.{name}")
        else:
            _check_leaf(path, v, s)


def resolve_path(p: str | Path) -> Path:
    p = Path(p).expanduser()
    return p if p.is_absolute() else (REPO_ROOT / p)


def load(path: str | Path) -> dict:
    """Load YAML, deep-merge with DEFAULTS, validate, resolve paths, load facts_file."""
    path = Path(path)
    try:
        with open(path, "r", encoding="utf-8") as f:
            user = yaml.safe_load(f)
    except FileNotFoundError:
        raise ConfigError(f"config file not found: {path}") from None
    except yaml.YAMLError as e:
        raise ConfigError(f"{path}: invalid YAML: {e}") from None
    if user is None:
        user = {}
    if not isinstance(user, dict):
        raise ConfigError(f"{path}: top level must be a mapping")
    try:
        return from_dict(user)
    except ConfigError as e:
        raise ConfigError(f"{path}: {e}") from None


def from_dict(user: dict) -> dict:
    """Same as load() but from an already-parsed dict."""
    # Check unknown keys before merging so typos are reported against the user's file.
    _validate_unknown(user, DEFAULTS, "")
    user_mode = (user.get("sampler") or {}).get("mode") if isinstance(user.get("sampler"), dict) else None
    cfg = _deep_merge(DEFAULTS, user)
    if cfg.get("task_id") and not cfg.get("task_prompt_file"):
        cfg["task_prompt_file"] = f"tasks/{cfg['task_id']}/TASK.md"
    if cfg["arm"] == "naive" and user_mode is None:
        cfg["sampler"]["mode"] = "uniform"
    _validate(cfg, _SPEC)

    if cfg["arm"] not in ARMS:
        raise ConfigError(f"arm: must be one of {ARMS}, got {cfg['arm']!r}")
    if cfg["sampler"]["mode"] not in SAMPLER_MODES:
        raise ConfigError(f"sampler.mode: must be one of {SAMPLER_MODES}, got {cfg['sampler']['mode']!r}")
    expected = ("uniform", "fixed_generic") if cfg["arm"] == "naive" else ("thompson",)
    if cfg["sampler"]["mode"] not in expected:
        raise ConfigError(f"arm {cfg['arm']!r} requires sampler.mode in {expected!r}, got {cfg['sampler']['mode']!r}")
    if not cfg["sampler"]["gamma"] <= 1:
        raise ConfigError(f"sampler.gamma: must be in (0, 1], got {cfg['sampler']['gamma']!r}")
    for k in ("ideation", "summary", "tagging", "assign", "judge"):
        if k not in cfg["llm"]["purposes"]:
            raise ConfigError(f"llm.purposes.{k}: missing")

    _validate_judge(cfg)

    cfg["task_prompt_file"] = str(resolve_path(cfg["task_prompt_file"]))
    if cfg["facts_file"]:
        fp = resolve_path(cfg["facts_file"])
        cfg["facts_file"] = str(fp)
        extra = load_facts_file(fp)
        cfg["facts"] = list(cfg["facts"]) + [x for x in extra if x not in cfg["facts"]]
    return cfg


def load_facts_file(fp: str | Path) -> list[str]:
    fp = resolve_path(fp)
    try:
        with open(fp, "r", encoding="utf-8") as f:
            extra = yaml.safe_load(f)
    except FileNotFoundError:
        raise ConfigError(f"facts_file not found: {fp}") from None
    except yaml.YAMLError as e:
        raise ConfigError(f"facts_file {fp}: invalid YAML: {e}") from None
    if extra is None:
        extra = []
    if not (isinstance(extra, list) and all(isinstance(x, str) for x in extra)):
        raise ConfigError(f"facts_file {fp}: must be a YAML list of strings")
    return extra


def _validate_judge(cfg: dict) -> None:
    jc = cfg["judge"]
    if jc["strategy"] not in JUDGE_STRATEGIES:
        raise ConfigError(f"judge.strategy: must be one of {JUDGE_STRATEGIES}, got {jc['strategy']!r}")
    if jc["fewshot_source"] not in FEWSHOT_SOURCES:
        raise ConfigError(f"judge.fewshot_source: must be one of {FEWSHOT_SOURCES}, got {jc['fewshot_source']!r}")
    if jc["facts_file"]:
        jc["facts_file"] = str(resolve_path(jc["facts_file"]))
        load_facts_file(jc["facts_file"])
    from ar import anchors  # noqa: PLC0415  (lazy: anchors reads task files)
    if jc["anchor_facts"]:
        try:
            anchors.anchor_facts(cfg["task_id"])
        except (OSError, KeyError, ValueError) as e:
            raise ConfigError(f"judge.anchor_facts: no anchor facts for task {cfg['task_id']!r}: {e}") from None
    if jc["strategy"] == "ranking":
        try:
            anchors.ranking_references(cfg["task_id"])
        except (OSError, KeyError) as e:
            raise ConfigError(f"judge.strategy ranking: no reference ideas for task {cfg['task_id']!r}: {e}") from None


def _validate_unknown(user: Any, defaults: dict, prefix: str) -> None:
    if not isinstance(user, dict):
        return
    unknown = sorted(set(user) - set(defaults))
    if unknown:
        raise ConfigError(f"{prefix or 'config'}: unknown key(s) {unknown}; allowed: {sorted(defaults)}")
    for k, v in user.items():
        d = defaults.get(k)
        if isinstance(d, dict) and isinstance(v, dict) and not (prefix == "llm" and k == "purposes"):
            _validate_unknown(v, d, f"{prefix}.{k}" if prefix else k)


def task_prompt(cfg: dict) -> str:
    """Text of the task prompt file ({{prompt}})."""
    p = resolve_path(cfg["task_prompt_file"])
    try:
        return p.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        raise ConfigError(f"task_prompt_file not found: {p}") from None
