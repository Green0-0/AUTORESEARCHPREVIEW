from pathlib import Path

import pytest
import yaml

from ar import config as C
from ar.config import ConfigError, load, task_prompt

REPO = Path(__file__).resolve().parents[1]


def _w(tmp_path, d, name="c.yaml"):
    p = tmp_path / name
    p.write_text(yaml.safe_dump(d) if not isinstance(d, str) else d)
    return p


def test_defaults_filled(tmp_path):
    cfg = load(_w(tmp_path, {"run_id": "r1", "task_id": "t19_data_ordering"}))
    assert cfg["arm"] == "optimized" and cfg["seed"] == 1
    assert cfg["llm"]["max_inflight"] == 64
    assert cfg["llm"]["purposes"]["judge"] == {"thinking": True, "max_tokens": 12000, "temperature": 0.7}
    assert cfg["sampler"] == {"mode": "thompson", "lambda0": 0.1, "lambda": 0.1, "sigma": 0.1, "gamma": 0.99}
    assert cfg["scoring"]["calibration"] == {"a": 0.0, "b": 1.0}
    assert cfg["tagging"] == {"trigger_count": 20, "trigger_minutes": 10, "batch_limit": 100,
                              "S": 10, "r": 2, "N": 4, "K": 4, "poll_seconds": 5}
    assert cfg["stop"] == {"max_hours": 24, "max_ideas": None}
    assert cfg["claims"]["timeout_minutes"] == 30 and cfg["judge"]["retries"] == 2
    assert cfg["references"] == [] and cfg["facts"] == []
    assert Path(cfg["task_prompt_file"]) == C.REPO_ROOT / "tasks/t19_data_ordering/TASK.md"


def test_deep_merge_partial(tmp_path):
    cfg = load(_w(tmp_path, {"run_id": "r", "task_id": "t",
                             "llm": {"purposes": {"judge": {"temperature": 0.6}}},
                             "scoring": {"calibration": {"b": 0.8}}}))
    assert cfg["llm"]["purposes"]["judge"] == {"thinking": True, "max_tokens": 12000, "temperature": 0.6}
    assert cfg["llm"]["purposes"]["ideation"]["max_tokens"] == 16000
    assert cfg["scoring"]["calibration"] == {"a": 0.0, "b": 0.8}
    assert cfg["llm"]["base_url"] == "http://127.0.0.1:8000/v1"
    assert C.DEFAULTS["scoring"]["calibration"]["b"] == 1.0  # defaults not mutated


@pytest.mark.parametrize("bad,msg", [
    ({"arm": "smart"}, "arm"),
    ({"sampler": {"mode": "greedy"}}, "sampler.mode"),
    ({"seed": "one"}, "seed"),
    ({"seed": True}, "seed"),
    ({"llm": {"max_inflight": 0}}, "max_inflight"),
    ({"llm": {"purposes": {"judge": {"thinking": "yes"}}}}, "thinking"),
    ({"tagging": {"S": 2.5}}, "tagging.S"),
    ({"references": "a"}, "references"),
    ({"facts": [1, 2]}, "facts"),
    ({"bogus": 1}, "unknown"),
    ({"sampler": {"sigmaa": 1}}, "unknown"),
    ({"arm": "naive", "sampler": {"mode": "thompson"}}, "requires"),
    ({"sampler": {"mode": "fixed_generic"}}, "requires"),
    ({"sampler": {"gamma": 1.5}}, "gamma"),
    ({"stop": {"max_ideas": -3}}, "max_ideas"),
])
def test_validation_errors(tmp_path, bad, msg):
    with pytest.raises(ConfigError, match=msg):
        load(_w(tmp_path, {"run_id": "r", "task_id": "t", **bad}))


def test_required_and_file_errors(tmp_path):
    with pytest.raises(ConfigError, match="run_id"):
        load(_w(tmp_path, {"task_id": "t"}))
    with pytest.raises(ConfigError, match="not found"):
        load(tmp_path / "nope.yaml")
    with pytest.raises(ConfigError, match="mapping"):
        load(_w(tmp_path, "- a\n- b\n"))
    with pytest.raises(ConfigError, match="invalid YAML"):
        load(_w(tmp_path, "a: [1,\n"))


def test_naive_defaults_to_uniform(tmp_path):
    cfg = load(_w(tmp_path, {"run_id": "r", "task_id": "t", "arm": "naive"}))
    assert cfg["sampler"]["mode"] == "uniform"


def test_facts_file_and_task_prompt(tmp_path):
    ff = tmp_path / "facts.yaml"
    ff.write_text(yaml.safe_dump(["fact two", "fact one"]))
    tp = tmp_path / "TASK.md"
    tp.write_text("Do the task.\n")
    cfg = load(_w(tmp_path, {"run_id": "r", "task_id": "t", "facts": ["fact one"],
                             "facts_file": str(ff), "task_prompt_file": str(tp)}))
    assert cfg["facts"] == ["fact one", "fact two"]
    assert task_prompt(cfg) == "Do the task."
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump({"a": 1}))
    with pytest.raises(ConfigError, match="list of strings"):
        load(_w(tmp_path, {"run_id": "r", "task_id": "t", "facts_file": str(bad)}))
    with pytest.raises(ConfigError, match="facts_file not found"):
        load(_w(tmp_path, {"run_id": "r", "task_id": "t", "facts_file": str(tmp_path / "x.yaml")}))
    cfg2 = load(_w(tmp_path, {"run_id": "r", "task_id": "t",
                              "task_prompt_file": str(tmp_path / "missing.md")}))
    with pytest.raises(ConfigError, match="task_prompt_file not found"):
        task_prompt(cfg2)


def test_relative_paths_resolve_to_repo_root(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "REPO_ROOT", tmp_path)
    (tmp_path / "tasks" / "tx").mkdir(parents=True)
    (tmp_path / "tasks" / "tx" / "TASK.md").write_text("hello")
    (tmp_path / "tasks" / "tx" / "facts.yaml").write_text("- f\n")
    sub = tmp_path / "configs"
    sub.mkdir()
    cfg = load(_w(sub, {"run_id": "r", "task_id": "tx", "facts_file": "tasks/tx/facts.yaml"}))
    assert task_prompt(cfg) == "hello" and cfg["facts"] == ["f"]


@pytest.mark.parametrize("name,arm,mode", [("example_optimized.yaml", "optimized", "thompson"),
                                           ("example_naive.yaml", "naive", "uniform")])
def test_example_configs(name, arm, mode):
    cfg = load(REPO / "configs" / name)
    assert cfg["task_id"] == "t07_activation_tuning"
    assert cfg["arm"] == arm and cfg["sampler"]["mode"] == mode
    assert cfg["task_prompt_file"].endswith("tasks/t07_activation_tuning/TASK.md")
