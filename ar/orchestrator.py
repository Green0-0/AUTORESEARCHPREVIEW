"""Pipeline orchestrator: runs ideator / tagger / judge workers for one or more runs
in a single asyncio process (CONTRACTS.md "Orchestrator").

    python -m ar.orchestrator --config configs/X.yaml [--config configs/Y.yaml ...] [--resume] [--mock]

Per run:
  * its own Journal (``$AR_ROOT/runs/<run_id>/``) and its own LLM client (MockLLM with --mock);
  * the three workers, each under a supervisor that restarts it (with backoff) if it crashes;
  * a monitor loop that writes ``heartbeat.json``, reaps stale claims and checks the stop rules.

Stopping:
  * ``stop.max_hours`` (cumulative over --resume sessions) or ``stop.max_ideas`` (ideas created):
    the ideator is stopped, then tagger/judge drain the ideas already in flight
    (bounded by --drain-minutes), then everything stops.
  * SIGTERM / SIGINT: every worker is told to stop at once; they finish their in-flight journal
    writes (LLM calls never hold the lock), and are cancelled after --grace seconds.
    A second signal cancels immediately.
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import fcntl
import json
import logging
import os
import signal
import socket
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any

log = logging.getLogger("ar.orchestrator")

DOWNSTREAM_PENDING = ("summarized", "tagged", "judged")   # statuses tagger/judge still owe work on
DONE_STATUSES = ("scored", "promoted", "real_judged")
WORKER_NAMES = ("ideator", "tagger", "judge")


# --------------------------------------------------------------------------- dependencies
def _deps() -> SimpleNamespace:
    """Imported lazily so tests can substitute fakes (monkeypatch this function)."""
    from ar import config as C
    from ar import journal as J
    from ar import llm as L
    from ar.workers import ideator, judge, tagger

    return SimpleNamespace(
        load_config=C.load,
        Journal=J.Journal,
        LLM=L.LLM,
        MockLLM=L.MockLLM,
        workers={"ideator": ideator.run, "tagger": tagger.run, "judge": judge.run},
    )


def default_runs_root() -> Path:
    if os.environ.get("AR_RUNS"):
        return Path(os.environ["AR_RUNS"])
    return Path(os.environ.get("AR_ROOT", "/usr/xtmp/kg349/ar")) / "runs"


def _atomic_write_json(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=str)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def status_counts(state: dict) -> dict[str, int]:
    out: dict[str, int] = {}
    for idea in state.get("ideas", {}).values():
        s = idea.get("status", "?")
        out[s] = out.get(s, 0) + 1
    return out


# --------------------------------------------------------------------------- one run
class Run:
    def __init__(self, cfg_path: str, args, deps: SimpleNamespace):
        self.args = args
        self.deps = deps
        self.cfg_path = cfg_path
        new_cfg = deps.load_config(cfg_path)
        self.run_id = new_cfg["run_id"]
        self.run_dir = Path(args.runs_root) / self.run_id
        self.new_cfg = new_cfg
        self.cfg: dict = new_cfg
        self.journal = None
        self.llm = None
        self.stop_ideate = asyncio.Event()
        self.stop_all = asyncio.Event()
        self.tasks: dict[str, asyncio.Task] = {}
        self.wstate = {n: {"restarts": 0, "last_error": None, "alive": False} for n in WORKER_NAMES}
        self.reason: str | None = None
        self.prev_elapsed = 0.0
        self.t0 = time.time()
        self.started_at = time.time()
        self.drain_deadline: float | None = None
        self._lockfile = None
        self.log = logging.getLogger(f"ar.orchestrator.{self.run_id}")

    # ---- setup -----------------------------------------------------------------------
    def setup(self) -> None:
        d = self.deps
        jpath = self.run_dir / "journal.json"
        if jpath.exists():
            if not self.args.resume:
                raise SystemExit(f"[{self.run_id}] {jpath} exists; pass --resume to continue it")
            self.journal = d.Journal(self.run_dir)
            self._take_run_lock()
            state = self.journal.read()
            frozen = state.get("config") or self.new_cfg
            self.cfg = dict(frozen)
            # Only the stop rules may be changed on resume (e.g. to extend a run).
            if self.new_cfg.get("stop") != frozen.get("stop"):
                self.log.warning("resume: stop rules changed %s -> %s", frozen.get("stop"), self.new_cfg.get("stop"))
                self.cfg["stop"] = self.new_cfg.get("stop")
            if any(self.new_cfg.get(k) != frozen.get(k) for k in self.new_cfg if k != "stop"):
                self.log.warning("resume: config file differs from the frozen journal config; using the frozen one")
            hb = self._read_heartbeat()
            self.prev_elapsed = float(hb.get("elapsed_s", 0.0)) if hb else 0.0
            self.started_at = float(hb.get("started_at", time.time())) if hb else time.time()
            with self.journal.transaction("orchestrator", "resume") as st:
                n = 0
                for idea in st["ideas"].values():
                    if idea.get("claim"):           # single orchestrator per run: all old claims are dead
                        idea["claim"] = None
                        n += 1
                if self.cfg.get("stop") != st["config"].get("stop"):
                    st["config"]["stop"] = self.cfg.get("stop")
            self.log.info("resumed: released %d claims, prior elapsed %.2f h", n, self.prev_elapsed / 3600)
        else:
            if self.args.resume:
                self.log.warning("--resume given but no journal at %s; starting a new run", jpath)
            self.journal = d.Journal.init(self.run_dir, self.new_cfg)
            self._take_run_lock()
        cfg_yaml = self.run_dir / "config.yaml"
        if not cfg_yaml.exists():
            import yaml
            cfg_yaml.write_text(yaml.safe_dump(self.cfg, sort_keys=False))
        llm_cfg = dict(self.cfg["llm"])
        llm_cfg.setdefault("usage_path", str(self.run_dir / "llm_usage.jsonl"))
        self.llm = (d.MockLLM if self.args.mock else d.LLM)(llm_cfg)
        # file log per run
        fh = logging.FileHandler(self.run_dir / "orchestrator.log")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        self.log.addHandler(fh)

    def _take_run_lock(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        f = open(self.run_dir / "orchestrator.pid", "a+")
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise SystemExit(f"[{self.run_id}] another orchestrator holds {self.run_dir}/orchestrator.pid")
        f.seek(0)
        f.truncate()
        f.write(f"{socket.gethostname()} {os.getpid()}\n")
        f.flush()
        self._lockfile = f

    def _read_heartbeat(self) -> dict | None:
        p = self.run_dir / "heartbeat.json"
        try:
            return json.loads(p.read_text())
        except (OSError, ValueError):
            return None

    # ---- workers ---------------------------------------------------------------------
    async def _supervise(self, name: str, fn, stop: asyncio.Event) -> None:
        backoff = self.args.restart_backoff
        while not stop.is_set():
            self.wstate[name]["alive"] = True
            try:
                await fn(self.journal, self.cfg, self.llm, stop)
                if stop.is_set():
                    break
                self.log.warning("worker %s returned without stop; restarting", name)
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001 - supervisor must survive anything
                tb = traceback.format_exc()
                self.wstate[name]["last_error"] = f"{type(e).__name__}: {e}"
                self.log.error("worker %s crashed:\n%s", name, tb)
                with contextlib.suppress(Exception):
                    with self.journal.transaction("orchestrator", f"worker_crash:{name}") as st:
                        st.setdefault("logs", {}).setdefault("orchestrator", []).append(
                            {"t": time.time(), "event": "worker_crash", "worker": name, "error": tb[-4000:]})
            finally:
                self.wstate[name]["alive"] = False
            self.wstate[name]["restarts"] += 1
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(stop.wait(), timeout=backoff)
            backoff = min(backoff * 2, 300.0)

    def start_workers(self) -> None:
        w = self.deps.workers
        self.tasks["ideator"] = asyncio.create_task(self._supervise("ideator", w["ideator"], self.stop_ideate))
        self.tasks["tagger"] = asyncio.create_task(self._supervise("tagger", w["tagger"], self.stop_all))
        self.tasks["judge"] = asyncio.create_task(self._supervise("judge", w["judge"], self.stop_all))

    # ---- monitor ---------------------------------------------------------------------
    def elapsed(self) -> float:
        return self.prev_elapsed + (time.time() - self.t0)

    def _orch_event(self, op: str, **kw) -> None:
        with contextlib.suppress(Exception):
            with self.journal.transaction("orchestrator", op) as st:
                st.setdefault("logs", {}).setdefault("orchestrator", []).append({"t": time.time(), "event": op, **kw})

    def reap(self, state: dict, now: float) -> int:
        timeout_s = float(self.cfg.get("claims", {}).get("timeout_minutes", 30)) * 60
        stale = [i for i, idea in state["ideas"].items()
                 if idea.get("claim") and now - float(idea["claim"].get("at", now)) > timeout_s]
        if not stale:
            return 0
        n = 0
        with self.journal.transaction("orchestrator", "reap") as st:
            for i in stale:
                c = st["ideas"].get(i, {}).get("claim")
                if c and now - float(c.get("at", now)) > timeout_s:
                    st["ideas"][i]["claim"] = None
                    n += 1
        if n:
            self.log.warning("reaped %d stale claims", n)
        return n

    def write_heartbeat(self, state: dict | None, finished: bool = False) -> None:
        counts = status_counts(state) if state else {}
        hb = {
            "t": time.time(), "run_id": self.run_id, "host": socket.gethostname(), "pid": os.getpid(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "started_at": self.started_at,
            "session_started_at": self.t0, "elapsed_s": self.elapsed(), "mock": bool(self.args.mock),
            "n_ideas": sum(counts.values()), "status_counts": counts,
            "n_tags": len(state.get("tags", {})) if state else None,
            "stopping": {"ideation": self.stop_ideate.is_set(), "all": self.stop_all.is_set(),
                         "reason": self.reason, "drain_deadline": self.drain_deadline},
            "workers": {n: {**self.wstate[n], "done": n in self.tasks and self.tasks[n].done()} for n in WORKER_NAMES},
            "finished": finished,
        }
        with contextlib.suppress(OSError):
            _atomic_write_json(self.run_dir / "heartbeat.json", hb)

    def begin_stop(self, reason: str, drain: bool) -> None:
        if self.reason is None:
            self.reason = reason
            self.log.info("stopping: %s (drain=%s)", reason, drain)
            self._orch_event("stop", reason=reason, drain=drain, elapsed_s=self.elapsed())
        self.stop_ideate.set()
        if drain:
            if self.drain_deadline is None:
                self.drain_deadline = time.time() + self.args.drain_minutes * 60
        else:
            self.stop_all.set()

    async def monitor(self, global_stop: asyncio.Event) -> None:
        last_reap = 0.0
        stop_cfg = self.cfg.get("stop") or {}
        max_hours = stop_cfg.get("max_hours")
        max_ideas = stop_cfg.get("max_ideas")
        while True:
            now = time.time()
            try:
                state = await asyncio.to_thread(self.journal.read)   # lock-free snapshot; keep the loop responsive
            except Exception as e:  # noqa: BLE001
                self.log.error("journal read failed: %s", e)
                state = None
            if global_stop.is_set():
                self.begin_stop("signal", drain=False)
            elif state is not None and not self.stop_ideate.is_set():
                if max_hours is not None and self.elapsed() >= float(max_hours) * 3600:
                    self.begin_stop("max_hours", drain=True)
                elif max_ideas is not None and len(state["ideas"]) >= int(max_ideas):
                    self.begin_stop("max_ideas", drain=True)
            if state is not None and now - last_reap >= self.args.reap_seconds:
                last_reap = now
                with contextlib.suppress(Exception):
                    self.reap(state, now)
            # drain finished?
            if self.stop_ideate.is_set() and not self.stop_all.is_set() and self.tasks["ideator"].done():
                pending = sum(status_counts(state).get(s, 0) for s in DOWNSTREAM_PENDING) if state else 0
                if pending == 0:
                    self.log.info("drain complete")
                    self.stop_all.set()
                elif self.drain_deadline is not None and now > self.drain_deadline:
                    self.log.warning("drain timeout with %d ideas still pending", pending)
                    self.stop_all.set()
            # all workers gone for some other reason -> nothing left to do
            self.write_heartbeat(state)
            if self.stop_all.is_set():
                return
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(global_stop.wait(), timeout=self.args.poll_seconds)

    async def finish(self) -> None:
        tasks = list(self.tasks.values())
        if tasks:
            done, pending = await asyncio.wait(tasks, timeout=self.args.grace)
            for t in pending:
                self.log.warning("cancelling worker task after %.0fs grace", self.args.grace)
                t.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
        state = None
        with contextlib.suppress(Exception):
            state = self.journal.read()
        self._orch_event("finished", reason=self.reason, elapsed_s=self.elapsed())
        self.write_heartbeat(state, finished=True)
        for meth in ("aclose", "close"):
            fn = getattr(self.llm, meth, None)
            if fn is not None:
                with contextlib.suppress(Exception):
                    r = fn()
                    if asyncio.iscoroutine(r):
                        await r
                break
        self.log.info("run finished: reason=%s counts=%s", self.reason, status_counts(state) if state else {})

    async def run(self, global_stop: asyncio.Event) -> None:
        self._orch_event("start", pid=os.getpid(), host=socket.gethostname(), mock=bool(self.args.mock),
                         resume=bool(self.args.resume), slurm_job_id=os.environ.get("SLURM_JOB_ID"))
        self.start_workers()
        try:
            await self.monitor(global_stop)
        finally:
            await self.finish()


# --------------------------------------------------------------------------- main
def parse_args(argv=None):
    p = argparse.ArgumentParser(prog="python -m ar.orchestrator")
    p.add_argument("--config", action="append", required=True, help="run config YAML (repeatable)")
    p.add_argument("--resume", action="store_true", help="continue from existing journals")
    p.add_argument("--mock", action="store_true", help="use MockLLM (offline)")
    p.add_argument("--runs-root", default=str(default_runs_root()))
    p.add_argument("--poll-seconds", type=float, default=30.0, help="monitor / heartbeat interval")
    p.add_argument("--reap-seconds", type=float, default=60.0, help="stale-claim reaper interval")
    p.add_argument("--drain-minutes", type=float, default=30.0,
                   help="after max_hours/max_ideas: max time for tagger/judge to finish in-flight ideas")
    p.add_argument("--grace", type=float, default=120.0, help="seconds to let workers finish before cancelling")
    p.add_argument("--restart-backoff", type=float, default=5.0)
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


async def amain(args, deps: SimpleNamespace | None = None) -> int:
    deps = deps or _deps()
    runs = [Run(c, args, deps) for c in args.config]
    ids = [r.run_id for r in runs]
    if len(set(ids)) != len(ids):
        raise SystemExit(f"duplicate run_id among configs: {ids}")
    for r in runs:
        r.setup()

    global_stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    n_signals = {"n": 0}
    main_task = asyncio.current_task()

    def on_signal(signame: str) -> None:
        n_signals["n"] += 1
        log.warning("received %s (%d)", signame, n_signals["n"])
        global_stop.set()
        if n_signals["n"] >= 2 and main_task is not None:
            for r in runs:
                for t in r.tasks.values():
                    t.cancel()

    for s in (signal.SIGTERM, signal.SIGINT):
        with contextlib.suppress(NotImplementedError, RuntimeError, ValueError):
            loop.add_signal_handler(s, on_signal, s.name)
    try:
        results = await asyncio.gather(*(r.run(global_stop) for r in runs), return_exceptions=True)
    finally:
        for s in (signal.SIGTERM, signal.SIGINT):
            with contextlib.suppress(Exception):
                loop.remove_signal_handler(s)
    rc = 0
    for r, res in zip(runs, results):
        if isinstance(res, BaseException):
            log.error("run %s failed: %r", r.run_id, res)
            rc = 1
    return rc


def main(argv=None, deps: SimpleNamespace | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s", stream=sys.stderr)
    return asyncio.run(amain(args, deps))


if __name__ == "__main__":
    sys.exit(main())
