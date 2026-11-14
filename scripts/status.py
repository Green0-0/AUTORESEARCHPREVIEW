#!/usr/bin/env python3
"""Summary of one or more runs: counts by status, tag pool, refits, F>0 rate, LLM tokens,
length-truncation rate, heartbeat, real-judging.

    python scripts/status.py <run_id|run_dir> [...] [--json]
Reads journal.json (read-only, no lock), llm_usage.jsonl and heartbeat.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ORDER = ["generated", "summarized", "tagged", "judged", "scored", "promoted", "real_judged", "failed"]


def runs_root() -> Path:
    if os.environ.get("AR_RUNS"):
        return Path(os.environ["AR_RUNS"])
    return Path(os.environ.get("AR_ROOT", "/usr/xtmp/kg349/ar")) / "runs"


def resolve(run: str) -> Path:
    p = Path(run)
    return p if (p / "journal.json").exists() else runs_root() / run


def _tok(rec: dict, key: str) -> int:
    for k in (key, f"{key}_tokens"):
        if isinstance(rec.get(k), (int, float)):
            return int(rec[k])
    for sub in ("tokens", "usage"):
        if isinstance(rec.get(sub), dict):
            v = rec[sub].get(key) or rec[sub].get(f"{key}_tokens")
            if isinstance(v, (int, float)):
                return int(v)
    return 0


def usage_summary(path: Path) -> dict:
    by = defaultdict(lambda: {"calls": 0, "prompt": 0, "completion": 0, "length": 0, "latency_s": 0.0, "errors": 0})
    if path.exists():
        with open(path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                b = by[r.get("purpose", "?")]
                if r.get("ok") is False or r.get("error"):
                    b["errors"] += 1
                    continue
                b["calls"] += 1
                b["prompt"] += _tok(r, "prompt")
                b["completion"] += _tok(r, "completion")
                b["length"] += int(r.get("finish_reason") == "length" or bool(r.get("truncated")))
                lat = r.get("latency_s", r.get("latency"))
                if isinstance(lat, (int, float)):
                    b["latency_s"] += float(lat)
    return dict(by)


def summarize(run_dir: Path) -> dict:
    st = json.loads((run_dir / "journal.json").read_text())
    ideas = list(st.get("ideas", {}).values())
    counts = Counter(i.get("status") for i in ideas)
    scored = [i for i in ideas if i.get("F") is not None and i.get("status") != "failed"]
    fpos = [i for i in scored if i["F"] > 0]
    s_cal = [i["s_cal"] for i in ideas if i.get("s_cal") is not None]
    logs = st.get("logs", {})
    refits = logs.get("refits", [])
    tag_logs = logs.get("tagging", [])
    errors = Counter((i.get("error") or "")[:60] for i in ideas if i.get("status") == "failed")
    real = [i for i in ideas if i.get("status") == "real_judged"]
    rj = [e for e in logs.get("realjudge", []) if e.get("op") == "ingest"]
    hb = None
    try:
        hb = json.loads((run_dir / "heartbeat.json").read_text())
    except (OSError, ValueError):
        pass
    usage = usage_summary(run_dir / "llm_usage.jsonl")
    tot_calls = sum(b["calls"] for b in usage.values())
    tot_len = sum(b["length"] for b in usage.values())
    first = min((i.get("created_at", 0) for i in ideas), default=None)
    last = max((i.get("created_at", 0) for i in ideas), default=None)
    return {
        "run_id": st.get("config", {}).get("run_id", run_dir.name), "dir": str(run_dir),
        "arm": st.get("config", {}).get("arm"), "task_id": st.get("config", {}).get("task_id"),
        "n_ideas": len(ideas), "status_counts": {k: counts.get(k, 0) for k in ORDER if counts.get(k)},
        "tag_pool": len(st.get("tags", {})), "tagging_cycles": len(tag_logs),
        "tag_pool_per_cycle": [e.get("pool_size_after", e.get("pool_size")) for e in tag_logs
                               if isinstance(e, dict) and ("pool_size_after" in e or "pool_size" in e)][-10:],
        "refits": len(refits), "waves": st.get("counters", {}).get("wave"),
        "n_scored": len(scored), "F_pos": len(fpos), "F_pos_rate": len(fpos) / len(scored) if scored else None,
        "s_cal_mean": sum(s_cal) / len(s_cal) if s_cal else None, "s_cal_max": max(s_cal) if s_cal else None,
        "ideas_per_hour": (len(ideas) - 1) / ((last - first) / 3600) if ideas and last and first and last > first else None,
        "failed_top_errors": errors.most_common(5),
        "llm": {"calls": tot_calls, "prompt_tokens": sum(b["prompt"] for b in usage.values()),
                "completion_tokens": sum(b["completion"] for b in usage.values()),
                "length_truncation_rate": tot_len / tot_calls if tot_calls else None,
                "by_purpose": {p: {**b, "length_rate": b["length"] / b["calls"] if b["calls"] else None}
                               for p, b in sorted(usage.items())}},
        "real_judged": len(real), "spearman": rj[-1].get("spearman") if rj else None,
        "heartbeat": None if hb is None else {
            "age_s": round(time.time() - hb.get("t", 0)), "host": hb.get("host"), "finished": hb.get("finished"),
            "elapsed_h": round(hb.get("elapsed_s", 0) / 3600, 2), "stopping": hb.get("stopping"),
            "workers": hb.get("workers")},
    }


def fmt(s: dict) -> str:
    L = [f"== {s['run_id']}  ({s['task_id']}, arm {s['arm']})  {s['dir']}"]
    L.append(f"ideas {s['n_ideas']}: " + "  ".join(f"{k}={v}" for k, v in s["status_counts"].items()))
    rate = f"{s['F_pos_rate']:.3f}" if s["F_pos_rate"] is not None else "-"
    L.append(f"tags {s['tag_pool']} (cycles {s['tagging_cycles']}; last pool sizes {s['tag_pool_per_cycle']})  "
             f"refits {s['refits']}  waves {s['waves']}")
    sm = f"{s['s_cal_mean']:.3f}" if s["s_cal_mean"] is not None else "-"
    iph = f"{s['ideas_per_hour']:.0f}" if s["ideas_per_hour"] else "-"
    L.append(f"F>0 {s['F_pos']}/{s['n_scored']} = {rate}   s_cal mean {sm} max {s['s_cal_max']}   ideas/h {iph}")
    u = s["llm"]
    lr = f"{u['length_truncation_rate']:.3f}" if u["length_truncation_rate"] is not None else "-"
    L.append(f"LLM calls {u['calls']}  prompt tok {u['prompt_tokens']:,}  completion tok {u['completion_tokens']:,}  "
             f"length-truncated {lr}")
    for p, b in u["by_purpose"].items():
        r = f"{b['length_rate']:.3f}" if b["length_rate"] is not None else "-"
        L.append(f"   {p:<10} calls {b['calls']:>6}  completion {b['completion']:>11,}  "
                 f"avg {b['completion'] / max(1, b['calls']):>7.0f}  length {r}  errors {b['errors']}")
    if s["failed_top_errors"]:
        L.append("failed: " + "; ".join(f"{n}x {e!r}" for e, n in s["failed_top_errors"]))
    L.append(f"real-judged {s['real_judged']}  spearman {s['spearman']}")
    hb = s["heartbeat"]
    if hb:
        L.append(f"heartbeat {hb['age_s']}s ago on {hb['host']}  elapsed {hb['elapsed_h']} h  finished {hb['finished']}  "
                 f"stopping {hb['stopping']}")
        if hb.get("workers"):
            L.append("workers " + "  ".join(f"{n}:{'up' if w.get('alive') else 'down'}/r{w.get('restarts')}"
                                           for n, w in hb["workers"].items()))
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    out = [summarize(resolve(r)) for r in a.runs]
    if a.json:
        print(json.dumps(out, indent=1, default=str))
    else:
        print("\n\n".join(fmt(s) for s in out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
