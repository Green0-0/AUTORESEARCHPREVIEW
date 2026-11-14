#!/usr/bin/env python3
"""GPU-idle watchdog for slurm/pipeline.sbatch (plan §2.1: no idle GPUs).

Every --interval seconds it scrapes each vLLM replica's /metrics
(vllm:num_requests_running + vllm:num_requests_waiting). It ends the job when
  * any replica has had no running/waiting requests for --idle-minutes (unreachable counts as idle), or
  * a vLLM server process (--vllm-pids) has died, or
  * the orchestrator process (--orch-pid) has exited.
Ending = SIGTERM the orchestrator (graceful: it finishes in-flight journal writes), wait up to
--term-grace s for it to exit (the batch script then tears down vLLM and exits), and as a backstop
`scancel $SLURM_JOB_ID` if the job is still alive after --cancel-grace s.
Stdlib only.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.request

METRIC_RE = re.compile(r"^(vllm:num_requests_(running|waiting))(\{[^}]*\})?\s+([0-9.eE+-]+)\s*$", re.M)


def parse_metrics(text: str) -> dict[str, float]:
    out = {"running": 0.0, "waiting": 0.0}
    for m in METRIC_RE.finditer(text):
        out[m.group(2)] += float(m.group(4))
    return out


def scrape(port: int, host: str = "127.0.0.1", timeout: float = 5.0) -> dict[str, float] | None:
    try:
        with urllib.request.urlopen(f"http://{host}:{port}/metrics", timeout=timeout) as r:
            return parse_metrics(r.read().decode("utf-8", "replace"))
    except Exception:  # noqa: BLE001
        return None


def alive(pid: int | None) -> bool:
    if not pid:
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    # zombie (exited child of our parent shell) counts as dead
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().split(") ", 1)[1][0] != "Z"
    except OSError:
        return False


class Watchdog:
    def __init__(self, args, now=time.time, scrape_fn=scrape, alive_fn=alive):
        self.a = args
        self.now = now
        self.scrape = scrape_fn
        self.alive = alive_fn
        t = now()
        self.last_active = {p: t for p in args.ports}

    def check(self) -> str | None:
        """One poll. Returns a reason to stop, or None."""
        t = self.now()
        if self.a.orch_pid and not self.alive(self.a.orch_pid):
            return "orchestrator exited"
        for pid in self.a.vllm_pids or []:
            if not self.alive(pid):
                return f"vLLM server pid {pid} died"
        stats = {}
        for p in self.a.ports:
            m = self.scrape(p)
            stats[p] = m
            if m is not None and (m["running"] + m["waiting"]) > 0:
                self.last_active[p] = t
        self.log({"t": t, "metrics": {str(k): v for k, v in stats.items()},
                  "idle_s": {str(p): round(t - self.last_active[p]) for p in self.a.ports}})
        for p in self.a.ports:
            if t - self.last_active[p] >= self.a.idle_minutes * 60:
                return f"replica :{p} had no requests for {self.a.idle_minutes:g} min"
        return None

    def log(self, rec: dict) -> None:
        if self.a.log:
            with open(self.a.log, "a") as f:
                f.write(json.dumps(rec) + "\n")

    def end_job(self, reason: str) -> None:
        print(f"[watchdog] {time.strftime('%F %T')} stopping: {reason}", flush=True)
        self.log({"t": self.now(), "stop": reason})
        if self.a.orch_pid and self.alive(self.a.orch_pid):
            try:
                os.kill(self.a.orch_pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            t0 = time.time()
            while self.alive(self.a.orch_pid) and time.time() - t0 < self.a.term_grace:
                time.sleep(2)
        # The batch script tears everything down once the orchestrator is gone; scancel is the backstop.
        time.sleep(self.a.cancel_grace)
        job = os.environ.get("SLURM_JOB_ID")
        if job and not self.a.dry_run:
            print(f"[watchdog] job still alive after grace -> scancel {job}", flush=True)
            subprocess.run(["scancel", job], check=False)

    def loop(self) -> str:
        while True:
            reason = self.check()
            if reason:
                self.end_job(reason)
                return reason
            time.sleep(self.a.interval)


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--ports", type=int, nargs="+", default=[8000, 8001])
    p.add_argument("--orch-pid", type=int, default=None)
    p.add_argument("--vllm-pids", type=int, nargs="*", default=[])
    p.add_argument("--idle-minutes", type=float, default=10.0)
    p.add_argument("--interval", type=float, default=30.0)
    p.add_argument("--term-grace", type=float, default=180.0)
    p.add_argument("--cancel-grace", type=float, default=120.0)
    p.add_argument("--log", default=None, help="jsonl of every poll")
    p.add_argument("--dry-run", action="store_true", help="never scancel")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    print(f"[watchdog] ports={args.ports} orch={args.orch_pid} vllm={args.vllm_pids} "
          f"idle={args.idle_minutes}min job={os.environ.get('SLURM_JOB_ID')}", flush=True)
    Watchdog(args).loop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
