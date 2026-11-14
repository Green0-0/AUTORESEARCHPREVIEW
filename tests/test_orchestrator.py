"""Orchestrator unit tests with fake Journal / LLM / workers (no other ar modules needed)."""
from __future__ import annotations

import asyncio
import contextlib
import json
import os
import signal
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from ar import orchestrator as orch


# ----------------------------------------------------------------------------- fakes
class FakeJournal:
    _locks: dict[str, threading.Lock] = {}

    def __init__(self, run_dir):
        self.run_dir = Path(run_dir)
        self.path = self.run_dir / "journal.json"
        self.lock = FakeJournal._locks.setdefault(str(self.path), threading.Lock())

    @staticmethod
    def init(run_dir, config):
        run_dir = Path(run_dir)
        if (run_dir / "journal.json").exists():
            raise FileExistsError(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        state = {"config": config, "ideas": {}, "tags": {},
                 "logs": {"refits": [], "batches": [], "tagging": [], "realjudge": []},
                 "counters": {"next_idea": 0, "wave": 0, "scored_seq": 0}}
        (run_dir / "journal.json").write_text(json.dumps(state))
        return FakeJournal(run_dir)

    def read(self):
        return json.loads(self.path.read_text())

    @contextlib.contextmanager
    def transaction(self, worker, op):
        with self.lock:
            st = self.read()
            yield st
            self.path.write_text(json.dumps(st))
            with open(self.run_dir / "events.jsonl", "a") as f:
                f.write(json.dumps({"t": time.time(), "worker": worker, "op": op}) + "\n")


class FakeLLM:
    def __init__(self, cfg):
        self.cfg = cfg
        self.closed = False

    async def aclose(self):
        self.closed = True


def fake_load(path):
    cfg = yaml.safe_load(Path(path).read_text())
    cfg.setdefault("llm", {"base_url": "http://x", "model": "m"})
    cfg.setdefault("claims", {"timeout_minutes": 30})
    cfg.setdefault("stop", {"max_hours": 24, "max_ideas": None})
    return cfg


def _advance(journal, src, dst, worker):
    with journal.transaction(worker, "advance") as st:
        n = 0
        for idea in st["ideas"].values():
            if idea["status"] == src and not idea.get("claim"):
                idea["status"] = dst
                n += 1
    return n


async def fake_ideator(journal, cfg, llm, stop):
    while not stop.is_set():
        with journal.transaction("ideator", "gen") as st:
            for _ in range(2):
                i = f"i{st['counters']['next_idea']:06d}"
                st["counters"]["next_idea"] += 1
                st["ideas"][i] = {"id": i, "status": "summarized", "claim": None, "created_at": time.time()}
        await asyncio.sleep(0.01)


async def fake_tagger(journal, cfg, llm, stop):
    while not stop.is_set():
        _advance(journal, "summarized", "tagged", "tagger")
        await asyncio.sleep(0.02)


async def fake_judge(journal, cfg, llm, stop):
    while not stop.is_set():
        _advance(journal, "tagged", "scored", "judge")
        await asyncio.sleep(0.02)


def make_deps(**workers):
    w = {"ideator": fake_ideator, "tagger": fake_tagger, "judge": fake_judge}
    w.update(workers)
    return SimpleNamespace(load_config=fake_load, Journal=FakeJournal, LLM=FakeLLM, MockLLM=FakeLLM, workers=w)


def write_cfg(tmp_path, run_id, **stop):
    p = tmp_path / f"{run_id}.yaml"
    p.write_text(yaml.safe_dump({"run_id": run_id, "stop": {"max_hours": 1, "max_ideas": None, **stop}}))
    return str(p)


def args_for(tmp_path, cfgs, *extra):
    argv = []
    for c in cfgs:
        argv += ["--config", c]
    argv += ["--mock", "--runs-root", str(tmp_path / "runs"), "--poll-seconds", "0.05",
             "--reap-seconds", "0.05", "--drain-minutes", "0.2", "--grace", "5", "--restart-backoff", "0.05", *extra]
    return orch.parse_args(argv)


def run(args, deps):
    return asyncio.run(asyncio.wait_for(orch.amain(args, deps), timeout=20))


# ----------------------------------------------------------------------------- tests
def test_max_ideas_drains_and_heartbeat(tmp_path):
    cfg = write_cfg(tmp_path, "r1", max_ideas=10)
    assert run(args_for(tmp_path, [cfg]), make_deps()) == 0
    rd = tmp_path / "runs" / "r1"
    st = json.loads((rd / "journal.json").read_text())
    assert len(st["ideas"]) >= 10
    assert all(i["status"] == "scored" for i in st["ideas"].values()), "drain must finish in-flight ideas"
    hb = json.loads((rd / "heartbeat.json").read_text())
    assert hb["finished"] and hb["stopping"]["reason"] == "max_ideas"
    assert hb["status_counts"]["scored"] == len(st["ideas"])
    assert (rd / "config.yaml").exists()
    ops = [json.loads(l)["op"] for l in (rd / "events.jsonl").read_text().splitlines()]
    assert "start" in ops and "stop" in ops and "finished" in ops


def test_multiple_runs_separate_dirs_and_llm(tmp_path):
    a = write_cfg(tmp_path, "ra", max_ideas=6)
    b = write_cfg(tmp_path, "rb", max_ideas=12)
    assert run(args_for(tmp_path, [a, b]), make_deps()) == 0
    for rid, n in (("ra", 6), ("rb", 12)):
        st = json.loads((tmp_path / "runs" / rid / "journal.json").read_text())
        assert len(st["ideas"]) >= n


def test_existing_journal_requires_resume_and_resume_releases_claims(tmp_path):
    cfg = write_cfg(tmp_path, "r2", max_ideas=4)
    run(args_for(tmp_path, [cfg]), make_deps())
    with pytest.raises(SystemExit):
        run(args_for(tmp_path, [cfg]), make_deps())
    rd = tmp_path / "runs" / "r2"
    st = json.loads((rd / "journal.json").read_text())
    st["ideas"]["i999999"] = {"id": "i999999", "status": "summarized", "created_at": 0,
                              "claim": {"worker": "dead", "stage": "tagger", "at": time.time()}}
    (rd / "journal.json").write_text(json.dumps(st))
    n_before = len(st["ideas"])
    cfg2 = write_cfg(tmp_path, "r2", max_ideas=n_before + 4)    # extend the run on resume
    assert run(args_for(tmp_path, [cfg2], "--resume"), make_deps()) == 0
    st = json.loads((rd / "journal.json").read_text())
    assert st["ideas"]["i999999"]["claim"] is None
    assert st["ideas"]["i999999"]["status"] == "scored"
    assert len(st["ideas"]) >= n_before + 4
    assert st["config"]["stop"]["max_ideas"] == n_before + 4


def test_stale_claim_reaper(tmp_path):
    cfg = write_cfg(tmp_path, "r3", max_ideas=4)
    rd = tmp_path / "runs" / "r3"
    FakeJournal.init(rd, fake_load(cfg))
    reaped = {}

    async def slow_ideator(journal, cfg_, llm, stop):
        # first insert one idea with a stale claim, then behave normally
        with journal.transaction("ideator", "gen") as st:
            st["ideas"]["i000000"] = {"id": "i000000", "status": "summarized", "created_at": 0,
                                      "claim": {"worker": "x", "stage": "tagger", "at": time.time() - 3600}}
            st["counters"]["next_idea"] = 1
        await asyncio.sleep(0.3)
        reaped["claim"] = journal.read()["ideas"]["i000000"]["claim"]
        await fake_ideator(journal, cfg_, llm, stop)

    deps = make_deps(ideator=slow_ideator)
    # resume so the pre-made journal is used (and the resume path itself does not touch the fresh claim)
    assert run(args_for(tmp_path, [cfg], "--resume"), deps) == 0
    assert reaped["claim"] is None


def test_worker_crash_is_restarted(tmp_path):
    calls = {"n": 0}

    async def flaky_judge(journal, cfg, llm, stop):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("boom")
        await fake_judge(journal, cfg, llm, stop)

    cfg = write_cfg(tmp_path, "r4", max_ideas=6)
    assert run(args_for(tmp_path, [cfg]), make_deps(judge=flaky_judge)) == 0
    hb = json.loads((tmp_path / "runs" / "r4" / "heartbeat.json").read_text())
    assert hb["workers"]["judge"]["restarts"] >= 1
    assert "boom" in hb["workers"]["judge"]["last_error"]
    st = json.loads((tmp_path / "runs" / "r4" / "journal.json").read_text())
    assert all(i["status"] == "scored" for i in st["ideas"].values())


def test_sigterm_graceful_stop(tmp_path):
    finished_write = {"ok": False}

    async def careful_tagger(journal, cfg, llm, stop):
        while not stop.is_set():
            await asyncio.sleep(0.05)       # an "LLM call"
            _advance(journal, "summarized", "tagged", "tagger")
        finished_write["ok"] = True

    async def killer():
        await asyncio.sleep(0.5)
        os.kill(os.getpid(), signal.SIGTERM)

    cfg = write_cfg(tmp_path, "r5")         # no max_ideas: only the signal stops it

    async def go():
        k = asyncio.create_task(killer())
        rc = await orch.amain(args_for(tmp_path, [cfg]), make_deps(tagger=careful_tagger))
        await k
        return rc

    assert asyncio.run(asyncio.wait_for(go(), timeout=20)) == 0
    hb = json.loads((tmp_path / "runs" / "r5" / "heartbeat.json").read_text())
    assert hb["finished"] and hb["stopping"]["reason"] == "signal"
    assert finished_write["ok"]


def test_max_hours_stop(tmp_path):
    cfg = write_cfg(tmp_path, "r6")
    p = Path(cfg)
    d = yaml.safe_load(p.read_text())
    d["stop"]["max_hours"] = 0.3 / 3600    # 0.3 s
    p.write_text(yaml.safe_dump(d))
    assert run(args_for(tmp_path, [cfg]), make_deps()) == 0
    hb = json.loads((tmp_path / "runs" / "r6" / "heartbeat.json").read_text())
    assert hb["stopping"]["reason"] == "max_hours"
    assert hb["elapsed_s"] >= 0.3


def test_duplicate_run_ids_rejected(tmp_path):
    a = write_cfg(tmp_path, "dup")
    with pytest.raises(SystemExit):
        run(args_for(tmp_path, [a, a]), make_deps())


# ----------------------------------------------------------------------------- watchdog + pipeline.sbatch
def _load_watchdog():
    import importlib.util
    p = Path(orch.__file__).parents[1] / "slurm" / "watchdog.py"
    spec = importlib.util.spec_from_file_location("ar_watchdog", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


METRICS = """# HELP vllm:num_requests_running Number of requests in model execution batches.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{engine="0",model_name="qwen"} 3.0
vllm:num_requests_waiting{engine="0",model_name="qwen"} 1.0
vllm:num_requests_running{engine="1",model_name="qwen"} 2.0
vllm:generation_tokens_total{model_name="qwen"} 12345.0
"""


def test_watchdog_parse_metrics():
    wd = _load_watchdog()
    assert wd.parse_metrics(METRICS) == {"running": 5.0, "waiting": 1.0}
    assert wd.parse_metrics("") == {"running": 0.0, "waiting": 0.0}


def test_watchdog_idle_and_orchestrator_exit():
    wd = _load_watchdog()
    clock = {"t": 0.0}
    busy = {8000: True, 8001: True}
    orch_alive = {"v": True}

    def scrape(port):
        return {"running": 1.0 if busy[port] else 0.0, "waiting": 0.0} if port in busy else None

    args = wd.parse_args(["--ports", "8000", "8001", "--orch-pid", "123", "--idle-minutes", "10"])
    w = wd.Watchdog(args, now=lambda: clock["t"], scrape_fn=scrape,
                    alive_fn=lambda pid: orch_alive["v"] if pid == 123 else True)
    assert w.check() is None
    busy[8001] = False
    clock["t"] = 300
    assert w.check() is None
    clock["t"] = 599
    assert w.check() is None
    clock["t"] = 601                     # :8001 idle since t=0 -> 601 s > 10 min
    assert "8001" in w.check()
    busy[8001] = True
    w2 = wd.Watchdog(args, now=lambda: clock["t"], scrape_fn=scrape,
                     alive_fn=lambda pid: orch_alive["v"] if pid == 123 else True)
    assert w2.check() is None
    orch_alive["v"] = False
    assert w2.check() == "orchestrator exited"
    # unreachable server counts as idle
    del busy[8000]
    orch_alive["v"] = True
    w3 = wd.Watchdog(args, now=lambda: clock["t"], scrape_fn=scrape, alive_fn=lambda pid: True)
    clock["t"] += 700
    assert "8000" in w3.check()


def test_watchdog_vllm_pid_death():
    wd = _load_watchdog()
    args = wd.parse_args(["--ports", "8000", "--vllm-pids", "11", "22"])
    w = wd.Watchdog(args, scrape_fn=lambda p: {"running": 1, "waiting": 0}, alive_fn=lambda pid: pid != 22)
    assert "22" in w.check()


def test_pipeline_sbatch_flags():
    import subprocess
    root = Path(orch.__file__).parents[1]
    s = (root / "slurm" / "pipeline.sbatch").read_text()
    assert subprocess.run(["bash", "-n", str(root / "slurm" / "pipeline.sbatch")]).returncode == 0
    probe = (root / "probes" / "probe_gpu.sbatch").read_text()
    for flag in ("--pipeline-parallel-size 2", "--max-model-len 32768", "--gpu-memory-utilization 0.90",
                 "--max-num-seqs 64", "--enable-prefix-caching", "--kv-cache-dtype bfloat16",
                 "--attention-backend TRITON_ATTN", """--limit-mm-per-prompt '{"image":0,"video":0}'"""):
        assert flag in probe and flag in s, flag
    for must in ("--gres=gpu:a5000:4", "-p compsci-gpu", "nvidia-smi dmon", "watchdog.py", "serve 0,1 8000",
                 "serve 2,3 8001", "ar.orchestrator"):
        assert must in s, must
