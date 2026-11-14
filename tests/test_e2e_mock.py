"""End-to-end: the real orchestrator + real workers + MockLLM on a tmp run dir (~20 ideas).

Skips (with the missing module named) if any pipeline module is not importable yet."""
from __future__ import annotations

import asyncio
import importlib
import json
from pathlib import Path

import pytest
import yaml

MODULES = ["ar.config", "ar.journal", "ar.llm", "ar.prompts", "ar.templating", "ar.sampler", "ar.scoring",
           "ar.workers.ideator", "ar.workers.tagger", "ar.workers.judge"]
_missing = []
for _m in MODULES:
    try:
        importlib.import_module(_m)
    except Exception as e:  # noqa: BLE001
        _missing.append(f"{_m} ({type(e).__name__}: {e})")
pytestmark = pytest.mark.skipif(bool(_missing), reason="pipeline modules not ready: " + "; ".join(_missing))

from ar import orchestrator as orch  # noqa: E402


def _cfg(tmp_path: Path, run_id: str, arm: str, max_ideas: int) -> Path:
    task = tmp_path / "TASK.md"
    task.write_text("Improve CIFAR-100 top-1 of a fixed ResNet-18 in a 15-minute A5000 budget "
                    "by changing only the order in which training examples are presented.\n")
    cfg = {
        "run_id": run_id, "task_id": "t19_data_ordering", "arm": arm, "seed": 3,
        "task_prompt_file": str(task),
        "references": ["Curriculum learning orders examples from easy to hard."],
        "facts": ["Under this budget a standard ResNet-18 reaches about 60% top-1 (score 0.5)."],
        "llm": {"base_url": "http://127.0.0.1:9/v1", "model": "qwen", "max_inflight": 16},
        "sampler": {"mode": "thompson" if arm == "optimized" else "uniform",
                    "lambda0": 0.1, "lambda": 0.1, "sigma": 0.1, "gamma": 0.99},
        "ideation": {"batch_size": 4, "concurrent_batches": 1},
        "tagging": {"trigger_count": 6, "trigger_minutes": 0.02, "batch_limit": 100, "S": 4, "r": 2, "N": 2, "K": 2},
        "stop": {"max_hours": 0.1, "max_ideas": max_ideas},
    }
    p = tmp_path / f"{run_id}.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


def _deps():
    deps = orch._deps()
    load = deps.load_config

    def fast_load(path):
        cfg = load(path)
        cfg["tagging"]["poll_seconds"] = 0.2       # tagger poll (config key tagging.poll_seconds)
        return cfg

    deps.load_config = fast_load
    return deps


def _run(argv, timeout=25):
    args = orch.parse_args(argv)
    return asyncio.run(asyncio.wait_for(orch.amain(args, _deps()), timeout=timeout))


def _argv(tmp_path, cfgs, *extra):
    a = []
    for c in cfgs:
        a += ["--config", str(c)]
    return a + ["--mock", "--runs-root", str(tmp_path / "runs"), "--poll-seconds", "0.2",
                "--drain-minutes", "0.3", "--grace", "10", *extra]


def _check_run(rd: Path, min_ideas: int) -> dict:
    st = json.loads((rd / "journal.json").read_text())
    ideas = list(st["ideas"].values())
    scored = [i for i in ideas if i["status"] == "scored"]
    assert len(ideas) >= min_ideas
    assert len(scored) >= 0.8 * min_ideas, {i["id"]: (i["status"], i.get("error")) for i in ideas}
    for i in scored:
        assert i["summary"] and i["tags"], i["id"]
        assert i["s_cal"] is not None and i["F"] is not None and i["scored_seq"] is not None
        assert 1 <= len(i["surrogate_scores"]) <= 2
        if i["F_detail"] and i["F_detail"].get("confirmed"):
            assert len(i["surrogate_scores"]) == 2
    assert st["tags"], "tag pool is empty"
    assert all(set(t["ideas"]) <= set(st["ideas"]) for t in st["tags"].values())
    assert st["logs"]["refits"], "no sampler refits logged"
    assert st["logs"]["tagging"], "no tagging cycles logged"
    hb = json.loads((rd / "heartbeat.json").read_text())
    assert hb["finished"] and hb["stopping"]["reason"] == "max_ideas"
    assert (rd / "events.jsonl").stat().st_size > 0
    assert (rd / "config.yaml").exists()
    assert (rd / "llm_usage.jsonl").exists()
    return st


def test_e2e_mock_two_runs_then_resume(tmp_path):
    a = _cfg(tmp_path, "e2e_opt", "optimized", 20)
    b = _cfg(tmp_path, "e2e_naive", "naive", 12)
    assert _run(_argv(tmp_path, [a, b])) == 0
    st = _check_run(tmp_path / "runs" / "e2e_opt", 20)
    _check_run(tmp_path / "runs" / "e2e_naive", 12)
    # naive arm: prompts drawn uniformly from the same pool (not Generic-only)
    stn = json.loads((tmp_path / "runs" / "e2e_naive" / "journal.json").read_text())
    for i in stn["ideas"].values():
        assert i["prompt"]["b_tilde"] is None       # naive: uniform over the prompt pool, no posterior draws
    assert len({i["prompt"]["config"]["base"] for i in stn["ideas"].values()}) > 1
    n0 = len(st["ideas"])

    # status script runs on it
    import importlib.util  # noqa: PLC0415
    spec = importlib.util.spec_from_file_location("ar_status", Path(orch.__file__).parents[1] / "scripts" / "status.py")
    status_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(status_mod)
    s = status_mod.summarize(tmp_path / "runs" / "e2e_opt")
    assert s["n_ideas"] == n0 and s["tag_pool"] == len(st["tags"]) and s["llm"]["calls"] > 0
    assert "F>0" in status_mod.fmt(s)

    # resume: extend max_ideas, same journal continues
    d = yaml.safe_load(a.read_text())
    d["stop"]["max_ideas"] = n0 + 8
    a.write_text(yaml.safe_dump(d))
    with pytest.raises(SystemExit):
        _run(_argv(tmp_path, [a]))                      # existing journal needs --resume
    assert _run(_argv(tmp_path, [a], "--resume")) == 0
    st2 = _check_run(tmp_path / "runs" / "e2e_opt", n0 + 8)
    assert set(st["ideas"]) <= set(st2["ideas"])
    assert len(st2["logs"]["refits"]) > len(st["logs"]["refits"])
