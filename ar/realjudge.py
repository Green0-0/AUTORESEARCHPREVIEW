"""Real judging with Claude Code (CONTRACTS.md "Real judging"; plan "Real eval procedure", §2.1).

    python -m ar.realjudge select  --run <run_id> --top K --random R [--stratified N] [--seed S]
    python -m ar.realjudge submit  --run <run_id> --ids i000001 i000002 [--dry-run] [--force]
    python -m ar.realjudge ingest  --run <run_id> [--rescore]

Per idea, ``submit`` builds ``runs/<run>/real/<id>/``:

    harness/               read-only copy of tasks/<task>/harness   (sha256 of every file -> harness_hashes.json)
    solution/              seeded from tasks/<task>/starter         (the only place Claude may write)
    task.md idea.md task.yaml implement_prompt.md repair_template.md review_template.md meta.json
    logs/                  slurm logs, claude JSON outputs, smoke logs, dmon
    claude_transcripts/    copies of the headless sessions' transcripts (removed from ~/.claude afterwards)

and submits the chain (no GPU is held while Claude works):

    1. realjudge/cpu_implement.sbatch    CPU: claude implements, run.sh --cpu-smoke   (exit 0 iff smoke valid)
    2. realjudge/gpu_run.sbatch          GPU, afterok:1 (--kill-on-invalid-dep): run.sh solution out   only
       (realjudge/cpu_run.sbatch on compsci with -c <cpus>, no GPU, for tasks whose task.yaml says device: cpu)
    3. realjudge/cpu_review.sbatch 1     CPU, afterany:2: if the run failed -> one `claude --resume` repair +
                                         CPU smoke -> resubmit 2 (out2) once + `cpu_review.sbatch 2` afterany;
                                         else review -> result.json
The two CPU job bodies are the hidden subcommands ``job-implement`` and ``job-review``.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import re
import shutil
import socket
import stat
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
RJ_DIR = REPO / "realjudge"
CLAUDE_TIMEOUT_IMPL = "40m"
CLAUDE_TIMEOUT_REVIEW = "15m"
SMOKE_TIMEOUT = "6m"
GRADES = (0.0, 0.5, 1.0)
_COPY_IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc")


def warn(msg: str) -> None:
    print(f"[realjudge] WARNING: {msg}", file=sys.stderr)


def info(msg: str) -> None:
    print(f"[realjudge] {msg}", file=sys.stderr)


# --------------------------------------------------------------------------- environment hooks
def tasks_root() -> Path:
    return Path(os.environ.get("AR_TASKS", REPO / "tasks"))


def runs_root() -> Path:
    if os.environ.get("AR_RUNS"):
        return Path(os.environ["AR_RUNS"])
    return Path(os.environ.get("AR_ROOT", "/usr/xtmp/kg349/ar")) / "runs"


def claude_bin() -> str:
    return os.environ.get("AR_CLAUDE_BIN", "claude")


def cpu_partition() -> str:
    """Partition for CPU-only steps (implement, review, CPU-task full runs). The compsci CPU partition was ~75%
    offline on 2026-11-09, so the default is CPU-only jobs (no --gres) on compsci-gpu, approved by the user.
    Set AR_CPU_PARTITION=compsci to switch back."""
    return os.environ.get("AR_CPU_PARTITION", "compsci-gpu")


def sbatch_bin() -> str:
    return os.environ.get("AR_SBATCH", "sbatch")


def claude_home() -> Path:
    return Path(os.environ.get("AR_CLAUDE_HOME", Path.home() / ".claude"))


def _journal_mod():
    from ar import journal
    return journal


def resolve_run_dir(run: str, root: str | None = None) -> Path:
    p = Path(run)
    if p.is_dir() and (p / "journal.json").exists():
        return p
    return Path(root or runs_root()) / run


def _warn_cross_host(run_dir: Path) -> None:
    """Journal writers should share one host (plan §2): warn if a live orchestrator runs elsewhere."""
    with contextlib.suppress(Exception):
        hb = json.loads((run_dir / "heartbeat.json").read_text())
        if not hb.get("finished") and time.time() - hb["t"] < 300 and hb.get("host") != socket.gethostname():
            warn(f"orchestrator for this run is live on {hb.get('host')}; this journal write relies on "
                 f"filelock working across NFS hosts. Prefer running this on {hb.get('host')}.")


# --------------------------------------------------------------------------- small utils
def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _ignored(rel: str) -> bool:
    return "__pycache__" in rel.split("/") or rel.endswith(".pyc")


def hash_tree(root: Path) -> dict[str, str]:
    out = {}
    for p in sorted(root.rglob("*")):
        if p.is_file() and not p.is_symlink():
            rel = p.relative_to(root).as_posix()
            if not _ignored(rel):
                out[rel] = sha256_file(p)
    return out


def make_read_only(root: Path) -> None:
    for p in [root, *root.rglob("*")]:
        if p.is_symlink():
            continue
        m = p.stat().st_mode
        p.chmod(m & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))


def make_writable(root: Path) -> None:
    for p in [root, *root.rglob("*")]:
        if p.is_symlink():
            continue
        with contextlib.suppress(OSError):
            p.chmod(p.stat().st_mode | stat.S_IWUSR)


def rmtree_force(p: Path) -> None:
    if p.exists():
        make_writable(p)
        shutil.rmtree(p)


def atomic_write_json(path: Path, obj: Any) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=str)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def read_json(path: Path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def render(template: str, values: dict[str, str]) -> str:
    """Single-pass {{KEY}} substitution; substituted text is never re-scanned."""
    return re.sub(r"\{\{([A-Z_]+)\}\}", lambda m: str(values.get(m.group(1), m.group(0))), template)


def tail(path: Path, n_chars: int = 6000) -> str:
    try:
        t = Path(path).read_text(errors="replace")
    except OSError:
        return "(missing)"
    return t if len(t) <= n_chars else "…(truncated)…\n" + t[-n_chars:]


def load_task(task_id: str) -> dict:
    import yaml
    p = tasks_root() / task_id / "task.yaml"
    return yaml.safe_load(p.read_text()) or {}


# --------------------------------------------------------------------------- scoring
def normalize(value, floor, ref, *, valid=True, adherence=None) -> tuple[float | None, str | None]:
    """real_score = clip((value-floor)/(ref-floor), 0, 1); 0 on crash / invalid / adherence 0.
    The ratio form is direction-agnostic: for a lower-is-better metric ref < floor and the sign flips."""
    if value is None or not isinstance(value, (int, float)) or not math.isfinite(value):
        return 0.0, "no valid metric value (crash) -> 0"
    if valid is False:
        return 0.0, "run invalid (constraint violated) -> 0"
    if adherence is not None and float(adherence) == 0.0:
        return 0.0, "adherence 0 -> 0"
    if floor is None or ref is None:
        return None, "task.yaml floor/ref not set: value recorded, real_score None"
    if ref == floor:
        return None, "task.yaml floor == ref: cannot normalize"
    return float(min(1.0, max(0.0, (float(value) - float(floor)) / (float(ref) - float(floor))))), None


def snap_grade(x) -> float | None:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(x):
        return None
    return min(GRADES, key=lambda g: abs(g - x))


def parse_review(text: str) -> dict:
    """Last ```json block (or last {...} containing "adherence") -> grades snapped to {0, 0.5, 1}."""
    cands = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text or "", flags=re.S)
    if not cands:
        cands = re.findall(r"\{[^{}]*\"adherence\"[^{}]*\}", text or "", flags=re.S)
    for c in reversed(cands):
        try:
            d = json.loads(c)
        except ValueError:
            continue
        if "adherence" in d or "fidelity" in d:
            return {"adherence": snap_grade(d.get("adherence")), "fidelity": snap_grade(d.get("fidelity")),
                    "adherence_reason": d.get("adherence_reason"), "fidelity_reason": d.get("fidelity_reason")}
    return {"adherence": None, "fidelity": None, "parse_error": True}


def spearman(pairs: list[tuple[float, float]]) -> tuple[float | None, float | None]:
    if len(pairs) < 3:
        return None, None
    from scipy.stats import spearmanr
    x, y = zip(*pairs)
    if len(set(x)) < 2 or len(set(y)) < 2:
        return None, None
    r = spearmanr(x, y)
    return float(r.statistic if hasattr(r, "statistic") else r[0]), float(r.pvalue if hasattr(r, "pvalue") else r[1])


# --------------------------------------------------------------------------- select
def choose(state: dict, top: int, random_n: int, stratified: int, rng: np.random.Generator) -> list[tuple[str, str]]:
    """Pure selection. Eligible: status 'scored' with s_cal, no real score yet.
    top-K by s_cal, then N stratified picks (one per s_cal quantile bin of the rest), then R random of the rest."""
    elig = [i for i in state["ideas"].values()
            if i.get("status") == "scored" and i.get("s_cal") is not None and i.get("real_score") is None]
    elig.sort(key=lambda i: (-float(i["s_cal"]), i.get("created_at", 0), i["id"]))
    picked: list[tuple[str, str]] = [(i["id"], "top") for i in elig[:max(0, top)]]
    rest = [i for i in elig if i["id"] not in {p for p, _ in picked}]
    if stratified and rest:
        rest_sorted = sorted(rest, key=lambda i: float(i["s_cal"]))
        bins = np.array_split(np.arange(len(rest_sorted)), min(stratified, len(rest_sorted)))
        for b in bins:
            if len(b):
                picked.append((rest_sorted[int(rng.choice(b))]["id"], "stratified"))
        rest = [i for i in rest if i["id"] not in {p for p, _ in picked}]
    if random_n and rest:
        idx = rng.choice(len(rest), size=min(random_n, len(rest)), replace=False)
        picked += [(rest[int(k)]["id"], "random") for k in sorted(idx)]
    return picked


def cmd_select(args) -> int:
    J = _journal_mod()
    run_dir = resolve_run_dir(args.run, args.runs_root)
    _warn_cross_host(run_dir)
    journal = J.Journal(run_dir)
    rng = np.random.default_rng(args.seed)
    picked = choose(journal.read(), args.top, args.random, args.stratified, rng)
    now = time.time()
    done = []
    with journal.transaction("realjudge", "select") as st:
        for iid, why in picked:
            idea = st["ideas"][iid]
            if idea.get("status") != "scored" or idea.get("claim"):
                continue
            J.set_status(st, iid, "promoted", now)
            idea["real"] = {"selection": why, "selected_at": now}
            done.append((iid, why, idea.get("s_cal")))
        st["logs"].setdefault("realjudge", []).append(
            {"t": now, "op": "select", "seed": args.seed, "top": args.top, "random": args.random,
             "stratified": args.stratified, "ideas": [{"id": i, "selection": w, "s_cal": s} for i, w, s in done]})
    for iid, why, s in done:
        info(f"promoted {iid} ({why}, s_cal={s})")
    print(" ".join(i for i, _, _ in done))
    return 0


# --------------------------------------------------------------------------- submit
def _task_prompt_text(cfg: dict, task_id: str) -> str:
    rel = cfg.get("task_prompt_file")
    for p in ([REPO / rel] if rel else []) + [tasks_root() / task_id / "TASK.md"]:
        if Path(p).exists():
            return Path(p).read_text()
    raise FileNotFoundError(f"no TASK.md for {task_id}")


def compute_values(task: dict) -> dict[str, str]:
    """Prompt placeholders describing where the full run executes (GPU default; task.yaml `device: cpu`)."""
    device, cpus = run_device({"task": task})
    if device == "cpu":
        vals = {"COMPUTE": f"{cpus} CPU cores (no GPU)", "COMPUTE_SHORT": f"{cpus} CPU cores, no GPU",
                "COMPUTE_NOTE": f"Keep all {cpus} cores busy (e.g. parallel worker processes); the CPU smoke "
                                "mode gets fewer cores, so read the core count from the harness, don't hard-code it.",
                "PACKAGES": "torch, torchvision, numpy"}
    else:
        vals = {"COMPUTE": "one A5000 GPU", "COMPUTE_SHORT": "one A5000",
                "COMPUTE_NOTE": "Keep the GPU busy (data on GPU, GPU augmentation, AMP, large batches; "
                                "no CPU DataLoader bottleneck).",
                "PACKAGES": "torch, torchvision, numpy"}
    if task.get("packages"):
        vals["PACKAGES"] = str(task["packages"])
    return vals


def build_idea_dir(d: Path, idea: dict, cfg: dict, run_dir: Path) -> dict:
    task_id = cfg["task_id"]
    tdir = tasks_root() / task_id
    task = load_task(task_id)
    d.mkdir(parents=True)
    (d / "logs").mkdir()
    shutil.copytree(tdir / "harness", d / "harness", symlinks=False, ignore=_COPY_IGNORE)
    hashes = hash_tree(d / "harness")
    make_read_only(d / "harness")
    shutil.copytree(tdir / "starter", d / "solution", symlinks=False, ignore=_COPY_IGNORE)
    make_writable(d / "solution")
    task_md = _task_prompt_text(cfg, task_id)
    (d / "task.md").write_text(task_md)
    (d / "idea.md").write_text(idea.get("description") or "")
    shutil.copy(tdir / "task.yaml", d / "task.yaml")
    budget = task.get("budget_minutes") or 15
    vals = {"TASK_ID": task_id, "TASK_PROMPT": task_md, "IDEA": idea.get("description") or "",
            "BUDGET_MINUTES": str(budget), "METRIC": str(task.get("metric", "?")), **compute_values(task)}
    (d / "implement_prompt.md").write_text(render((RJ_DIR / "claude_implement_prompt.md").read_text(), vals))
    (d / "repair_template.md").write_text(render((RJ_DIR / "claude_repair_prompt.md").read_text(), vals))
    (d / "review_template.md").write_text(render((RJ_DIR / "claude_review_prompt.md").read_text(), vals))
    atomic_write_json(d / "harness_hashes.json", hashes)
    meta = {
        "idea_id": idea["id"], "run_id": cfg.get("run_id"), "run_dir": str(run_dir), "task_id": task_id,
        "task": task, "budget_minutes": budget, "metric": task.get("metric"),
        "higher_is_better": task.get("higher_is_better", True), "floor": task.get("floor"), "ref": task.get("ref"),
        "s_cal": idea.get("s_cal"), "F": idea.get("F"), "created": time.time(),
        "harness_src": str(tdir / "harness"), "starter_hashes": hash_tree(d / "solution"), "jobs": [],
    }
    atomic_write_json(d / "meta.json", meta)
    return meta


def gpu_minutes(meta: dict) -> int:
    t = meta.get("task") or {}
    if t.get("slurm_minutes"):
        return int(t["slurm_minutes"])
    b = float(meta.get("budget_minutes") or 15)
    return int(math.ceil(max(b * 1.5, b + 15)))   # budget + eval + startup; run.sh enforces the budget itself


def _hhmm(minutes: int) -> str:
    return f"{minutes // 60:02d}:{minutes % 60:02d}:00"


def sbatch(args: list[str]) -> str:
    out = subprocess.run([sbatch_bin(), "--parsable", *args], capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"sbatch failed ({out.returncode}): {out.stderr.strip()} :: {args}")
    return out.stdout.strip().split(";")[0]


def run_device(meta: dict) -> tuple[str, int | None]:
    """Where the full harness run executes: task.yaml `device: cpu` (+ `cpus: N`) -> ("cpu", N); else ("gpu", None)."""
    t = meta.get("task") or {}
    if str(t.get("device") or "gpu").strip().lower() == "cpu":
        return "cpu", int(t.get("cpus") or 16)
    return "gpu", None


def submit_gpu_and_review(d: Path, meta: dict, out_name: str, review_phase: int, after: str | None = None) -> list[dict]:
    """Full run (GPU job, or for `device: cpu` tasks a CPU job on compsci) + the review job after it.
    Both scripts write <out>/gpu_job.json and log to logs/gpu_<out>_<job>.out, so check_run/review are shared."""
    iid = meta["idea_id"]
    device, cpus = run_device(meta)
    tag = "gpu" if device == "gpu" else "cpu"
    gpu_args = ["-J", f"rj-{tag}-{iid}", "-t", _hhmm(gpu_minutes(meta)), "-o", str(d / "logs" / f"gpu_{out_name}_%j.out")]
    if device == "cpu":
        gpu_args += ["-p", cpu_partition(), "-c", str(cpus)]
    if after:
        gpu_args += [f"--dependency=afterok:{after}", "--kill-on-invalid-dep=yes"]
    script = "gpu_run.sbatch" if device == "gpu" else "cpu_run.sbatch"
    j2 = sbatch(gpu_args + [str(RJ_DIR / script), str(d), out_name])
    j3 = sbatch(["-J", f"rj-review-{iid}", "-p", cpu_partition(), f"--dependency=afterany:{j2}",
                 "-o", str(d / "logs" / f"review{review_phase}_%j.out"),
                 str(RJ_DIR / "cpu_review.sbatch"), str(d), str(review_phase)])
    return [{"step": f"{tag}:{out_name}", "job": j2, "t": time.time()},
            {"step": f"review{review_phase}", "job": j3, "t": time.time()}]


def submit_chain(d: Path, meta: dict) -> list[dict]:
    iid = meta["idea_id"]
    j1 = sbatch(["-J", f"rj-impl-{iid}", "-p", cpu_partition(), "-o", str(d / "logs" / "implement_%j.out"),
                 str(RJ_DIR / "cpu_implement.sbatch"), str(d)])
    return [{"step": "implement", "job": j1, "t": time.time()}] + submit_gpu_and_review(d, meta, "out", 1, after=j1)


def cmd_submit(args) -> int:
    J = _journal_mod()
    run_dir = resolve_run_dir(args.run, args.runs_root)
    journal = J.Journal(run_dir)
    state = journal.read()
    cfg = state["config"]
    submitted = []
    for iid in args.ids:
        idea = state["ideas"].get(iid)
        if idea is None:
            warn(f"{iid}: no such idea")
            continue
        if idea.get("status") != "promoted" and not args.force:
            warn(f"{iid}: status {idea.get('status')} (not promoted); use select first or --force")
            continue
        d = run_dir / "real" / iid
        if d.exists():
            if not args.force:
                warn(f"{iid}: {d} exists; --force to rebuild (old dir is kept as a backup)")
                continue
            d.rename(d.with_name(f"{iid}.old.{int(time.time())}"))
        meta = build_idea_dir(d, idea, cfg, run_dir)
        if args.dry_run:
            info(f"{iid}: built {d} (dry run, nothing submitted)")
            continue
        jobs = submit_chain(d, meta)
        meta["jobs"] = jobs
        atomic_write_json(d / "meta.json", meta)
        submitted.append({"id": iid, "dir": str(d), "jobs": jobs})
        info(f"{iid}: submitted " + " -> ".join(f"{j['step']}={j['job']}" for j in jobs))
    if submitted:
        _warn_cross_host(run_dir)
        with journal.transaction("realjudge", "submit") as st:
            st["logs"].setdefault("realjudge", []).append({"t": time.time(), "op": "submit", "ideas": submitted})
    return 0


# --------------------------------------------------------------------------- job helpers (run on CPU nodes)
def _enc_project(path: Path) -> str:
    return re.sub(r"[^A-Za-z0-9]", "-", str(path))


def _jail(d: Path) -> list[str]:
    """Command prefix that runs inside realjudge/jail.sh. AR_NO_JAIL=1 disables it (unit tests with fake executables only)."""
    return [] if os.environ.get("AR_NO_JAIL") == "1" else ["bash", str(RJ_DIR / "jail.sh"), str(d.resolve())]


def _claude_settings(d: Path, phase: str) -> Path:
    """Per-session permission rules, written outside the idea dir (read-only inside the jail, so tamper-proof)."""
    home = Path.home().resolve()
    root = Path(os.environ.get("AR_ROOT", "/usr/xtmp/kg349/ar")).resolve()
    repo = REPO.resolve()
    # (runs/ other than this idea dir is already invisible inside the jail; the idea dir lives under runs/ and
    #  deny beats allow, so runs/ is not listed here.)
    deny_dirs = [root / "private", repo, home / ".claude", home / ".ssh", root / "anchors", root / "probes",
                 root / "data" / "cifar-10-batches-py", root / "data" / "cifar-100-python"]
    allow = [f"Read(/{d}/**)", f"Edit(/{d}/**)", "Glob", "Grep", "Bash"]
    deny = [f"{t}(/{p}/**)" for p in deny_dirs for t in ("Read", "Edit")] + ["WebFetch", "WebSearch", "Agent"]
    out = root / "sandbox" / f"settings_{d.parent.parent.name}_{d.name}_{phase}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with contextlib.suppress(FileNotFoundError):
        out.chmod(0o644)
    out.write_text(json.dumps({"permissions": {"allow": allow, "deny": deny}}, indent=1))
    out.chmod(0o444)
    return out


def run_claude(d: Path, prompt: str, phase: str, *, timeout: str, max_turns: int,
               resume: str | None = None, read_only: bool = False) -> dict:
    # No --dangerously-skip-permissions: the session runs inside realjudge/jail.sh (read-only filesystem except
    # the idea dir; test data, repo and other runs hidden), in dontAsk mode with explicit allow/deny rules, and
    # anything that would prompt is denied. The reviewer additionally runs --restricted with read-only tools.
    d = d.resolve()
    tmp = d / ".tmp"
    tmp.mkdir(exist_ok=True)
    env = {**os.environ, "TMPDIR": str(tmp), "XDG_CACHE_HOME": str(tmp / "cache"),
           "TRITON_CACHE_DIR": str(tmp / "triton"), "TORCH_HOME": str(tmp / "torch")}
    env.pop("ANTHROPIC_API_KEY", None)
    settings = _claude_settings(d, phase)
    cmd = [*_jail(d), "timeout", timeout, claude_bin(), "-p"]
    if resume:
        cmd += ["--resume", resume]
    if read_only:
        cmd += ["--restricted", "--no-session-persistence", "--tools", "Read", "Glob", "Grep"]
    cmd += ["--model", "sonnet", "--max-turns", str(max_turns), "--output-format", "json",
            "--permission-mode", "dontAsk", "--permission-prompts", "none", "--settings", str(settings)]
    out_p, err_p = d / "logs" / f"claude_{phase}.json", d / "logs" / f"claude_{phase}.err"
    t0 = time.time()
    with open(out_p, "w") as fo, open(err_p, "w") as fe:
        rc = subprocess.run(cmd, input=prompt, text=True, cwd=d, stdout=fo, stderr=fe, env=env).returncode
    wall = time.time() - t0
    raw = out_p.read_text(errors="replace")
    parsed = None
    with contextlib.suppress(ValueError):
        parsed = json.loads(raw)
    if isinstance(parsed, list):     # stream-ish array form: keep the final result record
        parsed = next((x for x in reversed(parsed) if isinstance(x, dict) and x.get("type") == "result"), None)
    if not isinstance(parsed, dict):
        parsed = None
    p = parsed or {}
    rec = {"phase": phase, "rc": rc, "wall_s": round(wall, 1), "parsed": parsed is not None,
           "session_id": p.get("session_id"), "is_error": p.get("is_error"), "subtype": p.get("subtype"),
           "num_turns": p.get("num_turns"), "duration_ms": p.get("duration_ms"),
           "total_cost_usd": p.get("total_cost_usd"), "usage": p.get("usage"), "timed_out": rc == 124}
    rec["_result_text"] = p.get("result") if isinstance(p.get("result"), str) else ""
    return rec


def collect_transcripts(d: Path, session_ids: list[str], cleanup: bool) -> list[str]:
    """Copy the headless sessions' transcripts into d/claude_transcripts/; with cleanup, delete them
    (and the per-cwd project dir) from ~/.claude, which lives on the near-full home quota."""
    home = claude_home()
    proj = home / "projects"
    dst = d / "claude_transcripts"
    copied: list[str] = []
    own_dirs = {proj / _enc_project(d), proj / _enc_project(d.resolve())}
    files: set[Path] = set()
    for pd in own_dirs:
        if pd.is_dir():
            files |= {p for p in pd.rglob("*") if p.is_file()}
    for sid in filter(None, session_ids):
        files |= set(proj.glob(f"*/{sid}.jsonl"))
        for sd in proj.glob(f"*/{sid}"):
            if sd.is_dir():
                files |= {p for p in sd.rglob("*") if p.is_file()}
    for f in sorted(files):
        rel = f.relative_to(proj)
        out = dst / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        with contextlib.suppress(OSError):
            shutil.copy2(f, out)
            copied.append(str(rel))
    if cleanup:
        for pd in own_dirs:
            if pd.is_dir():
                shutil.rmtree(pd, ignore_errors=True)
        for sid in filter(None, session_ids):
            for p in list(proj.glob(f"*/{sid}.jsonl")) + [home / "file-history" / sid, home / "session-env" / sid] \
                    + list(proj.glob(f"*/{sid}")) + list((home / "todos").glob(f"{sid}*")):
                if p.is_dir():
                    shutil.rmtree(p, ignore_errors=True)
                elif p.exists():
                    with contextlib.suppress(OSError):
                        p.unlink()
    return copied


def check_harness(d: Path) -> list[str]:
    """Files of harness/ that changed, vanished or appeared since submit (pycache ignored)."""
    want = read_json(d / "harness_hashes.json", {})
    have = hash_tree(d / "harness") if (d / "harness").exists() else {}
    changed = [k for k in want if have.get(k) != want[k]]
    changed += [f"+{k}" for k in have if k not in want]
    return changed


def restore_harness(d: Path, meta: dict) -> bool:
    rmtree_force(d / "harness")
    shutil.copytree(meta["harness_src"], d / "harness", symlinks=False, ignore=_COPY_IGNORE)
    ok = hash_tree(d / "harness") == read_json(d / "harness_hashes.json", {})
    make_read_only(d / "harness")
    if not ok:
        warn(f"restored harness differs from the submit-time hashes ({meta['harness_src']} changed?)")
    return ok


def check_run(d: Path, out_name: str) -> dict:
    p = d / out_name / "metric.json"
    m = read_json(p)
    gj = read_json(d / out_name / "gpu_job.json")
    if not (d / out_name).exists():
        return {"out": out_name, "ok": False, "kind": "not_run", "metric_json": None, "gpu_job": gj}
    if not isinstance(m, dict):
        return {"out": out_name, "ok": False, "kind": "no_metric", "metric_json": None, "gpu_job": gj}
    v = m.get("value")
    finite = isinstance(v, (int, float)) and math.isfinite(v)
    if not finite:
        kind = "nonfinite"
    elif m.get("valid") is False:
        kind = "invalid"
    else:
        kind = "ok"
    return {"out": out_name, "ok": kind == "ok", "kind": kind, "metric_json": m, "gpu_job": gj}


def run_smoke(d: Path, name: str) -> dict:
    rmtree_force(d / name)
    log_p = d / "logs" / f"{name}.log"
    tmp = d.resolve() / ".tmp"
    tmp.mkdir(exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", AR_SEED=os.environ.get("AR_SEED", "0"), JAIL_MODE="harness",
               TMPDIR=str(tmp), XDG_CACHE_HOME=str(tmp / "cache"), TRITON_CACHE_DIR=str(tmp / "triton"))
    t0 = time.time()
    with open(log_p, "w") as f:   # solution code runs jailed: read-only fs except the idea dir
        rc = subprocess.run([*_jail(d), "timeout", SMOKE_TIMEOUT,
                             "bash", "harness/run.sh", "solution", name, "--cpu-smoke"],
                            cwd=d, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
    r = check_run(d, name)
    r.update({"rc": rc, "wall_s": round(time.time() - t0, 1), "log": str(log_p)})
    if rc != 0 and r["ok"]:
        r.update(ok=False, kind="smoke_rc_nonzero")
    return r


def _state(d: Path) -> dict:
    return read_json(d / "state.json", None) or {"claude": [], "session_id": None, "repaired": False,
                                                    "harness_tampered": [], "wall": {}}


def _save_state(d: Path, st: dict) -> None:
    atomic_write_json(d / "state.json", st)


def _harness_guard(d: Path, meta: dict, st: dict, when: str) -> None:
    changed = check_harness(d)
    if changed:
        warn(f"harness modified ({when}): {changed[:10]} -> restoring pristine copy")
        st["harness_tampered"].append({"when": when, "files": changed})
        restore_harness(d, meta)


def _record_claude(st: dict, rec: dict) -> None:
    st["claude"].append({k: v for k, v in rec.items() if not k.startswith("_")})


def job_implement(d: Path) -> int:
    d = Path(d)
    meta = read_json(d / "meta.json")
    st = _state(d)
    t0 = time.time()
    rec = run_claude(d, (d / "implement_prompt.md").read_text(), "implement",
                     timeout=CLAUDE_TIMEOUT_IMPL, max_turns=60)
    _record_claude(st, rec)
    st["session_id"] = rec["session_id"]
    st["wall"]["implement_s"] = round(time.time() - t0, 1)
    collect_transcripts(d, [rec["session_id"]], cleanup=False)   # keep: a repair may --resume it
    _harness_guard(d, meta, st, "after implement")
    unchanged = hash_tree(d / "solution") == meta.get("starter_hashes")
    if unchanged:
        # Never score the starter as if it were the idea: skip the GPU run (afterok fails).
        st["implement_status"] = "claude_failed" if not rec["parsed"] else "no_change"
        smoke = {"ok": False, "kind": st["implement_status"], "wall_s": 0.0, "log": None}
    else:
        st["implement_status"] = "changed"
        smoke = run_smoke(d, "out_cpu_smoke")
    st["smoke"] = [smoke]
    st["wall"]["smoke_s"] = smoke["wall_s"]
    _save_state(d, st)
    info(f"implement: claude rc={rec['rc']} session={rec['session_id']} smoke={smoke['kind']}")
    return 0 if smoke["ok"] else 1


def _failure_details(d: Path, st: dict, chk: dict) -> tuple[str, str]:
    if chk["kind"] == "not_run" and st.get("implement_status") == "no_change":
        return ("solution/ is still identical to the starter baseline -- the idea was not implemented",
                "Implement the idea in solution/ now (see implement_prompt.md for the full task, idea and rules).\n")
    if chk["kind"] == "not_run":
        sm = (st.get("smoke") or [{}])[-1]
        kind = f"the CPU smoke run failed ({sm.get('kind', 'unknown')}), so the full GPU run was not started"
        body = "### CPU smoke log (tail)\n```\n" + tail(Path(sm.get("log") or d / "logs" / "out_cpu_smoke.log")) + "\n```\n"
        if sm.get("metric_json"):
            body += "### out_cpu_smoke/metric.json\n```json\n" + json.dumps(sm["metric_json"], indent=1)[:4000] + "\n```\n"
        return kind, body
    logs = sorted((d / "logs").glob(f"gpu_{chk['out']}_*.out"), key=lambda p: p.stat().st_mtime)
    rc = (chk.get("gpu_job") or {}).get("rc")
    kind = {"no_metric": f"the full GPU run crashed (exit code {rc}) without writing metric.json",
            "nonfinite": "the full GPU run produced a non-finite metric value",
            "invalid": "the full GPU run finished but violated a constraint (valid: false)"}.get(chk["kind"], chk["kind"])
    body = ""
    if chk.get("metric_json"):
        body += "### metric.json\n```json\n" + json.dumps(chk["metric_json"], indent=1)[:4000] + "\n```\n"
    body += "### full-run log (tail)\n```\n" + (tail(logs[-1]) if logs else "(no log)") + "\n```\n"
    return kind, body


def do_repair(d: Path, meta: dict, st: dict, chk: dict) -> bool:
    kind, body = _failure_details(d, st, chk)
    prompt = render((d / "repair_template.md").read_text(), {"FAILURE_KIND": kind, "ERROR_LOG": body})
    if not st.get("session_id"):           # first session died before starting: fresh session with full context
        prompt = (d / "implement_prompt.md").read_text() + "\n\n---\n\n" + prompt
    (d / "repair_prompt.md").write_text(prompt)
    t0 = time.time()
    rec = run_claude(d, prompt, "repair", timeout=CLAUDE_TIMEOUT_IMPL, max_turns=60, resume=st.get("session_id"))
    _record_claude(st, rec)
    st["repair_session_id"] = rec["session_id"]
    _harness_guard(d, meta, st, "after repair")
    smoke = run_smoke(d, "out_cpu_smoke2")
    st.setdefault("smoke", []).append(smoke)
    st["wall"]["repair_s"] = round(time.time() - t0, 1)
    st["repair"] = {"failure": kind, "smoke": smoke["kind"]}
    info(f"repair: claude rc={rec['rc']} smoke={smoke['kind']}")
    return bool(smoke["ok"])


def finalize(d: Path, meta: dict, st: dict) -> dict:
    out_name = "out2" if (d / "out2").exists() else "out"
    chk = check_run(d, out_name)
    m = chk.get("metric_json") or {}
    value = m.get("value") if isinstance(m.get("value"), (int, float)) else None
    review = {"adherence": None, "fidelity": None, "skipped": "no metric (run failed)"}
    if value is not None:
        prompt = render((d / "review_template.md").read_text(),
                        {"OUT_DIR": out_name, "METRIC_JSON": "```json\n" + json.dumps(m, indent=1)[:6000] + "\n```"})
        t0 = time.time()
        sol_before = hash_tree(d / "solution")
        rec = run_claude(d, prompt, "review", timeout=CLAUDE_TIMEOUT_REVIEW, max_turns=30, read_only=True)
        _record_claude(st, rec)
        review = parse_review(rec["_result_text"])
        review["solution_modified_by_review"] = hash_tree(d / "solution") != sol_before
        st["wall"]["review_s"] = round(time.time() - t0, 1)
    final_changes = check_harness(d)
    if final_changes:
        st["harness_tampered"].append({"when": "final", "files": final_changes})
    adherence = review.get("adherence")
    notes = []
    if st["harness_tampered"]:
        adherence = 0.0
        notes.append("harness was modified by Claude -> adherence forced to 0")
    valid = m.get("valid") if m else None
    unchanged = hash_tree(d / "solution") == meta.get("starter_hashes")
    if chk["ok"]:
        status = "ok"
    elif st.get("implement_status") == "claude_failed" and unchanged:
        status = "infra_error"          # Claude never produced a session (auth/limit/network), nothing was tried
    elif chk["kind"] == "invalid":
        status = "invalid"
    else:
        status = "crash"
    real_score, why = normalize(value, meta.get("floor"), meta.get("ref"), valid=valid, adherence=adherence)
    if why:
        notes.append(why)
    gj = chk.get("gpu_job") or {}
    wall = dict(st.get("wall", {}))
    wall["gpu_run_s"] = m.get("wall_s") if m else None
    if gj:
        wall["gpu_job_s"] = gj.get("end", 0) - gj.get("start", 0)
    result = {
        "idea_id": meta["idea_id"], "run_id": meta.get("run_id"), "task_id": meta.get("task_id"),
        "metric": m.get("metric") or meta.get("metric"), "value": value, "real_score": real_score,
        "adherence": adherence, "fidelity": review.get("fidelity"), "valid": valid,
        "repaired": bool(st.get("repaired")), "status": status, "out": out_name, "metric_json": m or None,
        "review": review, "harness_tampered": st["harness_tampered"], "solution_unchanged_from_starter": unchanged,
        "claude_usage": st["claude"], "wall": wall, "notes": notes, "dir": str(d),
        "floor": meta.get("floor"), "ref": meta.get("ref"), "finished_at": time.time(),
    }
    atomic_write_json(d / "result.json", result)
    sids = [c.get("session_id") for c in st["claude"]]
    st["transcripts"] = collect_transcripts(d, sids, cleanup=True)
    _save_state(d, st)
    info(f"result: status={status} value={value} real_score={real_score} adherence={adherence} "
         f"fidelity={review.get('fidelity')}")
    return result


def job_review(d: Path, phase: int) -> int:
    d = Path(d)
    meta = read_json(d / "meta.json")
    st = _state(d)
    if phase == 1:
        chk = check_run(d, "out")
        st["run1"] = {k: v for k, v in chk.items() if k != "metric_json"}
        if not chk["ok"] and st.get("implement_status") == "claude_failed":
            info("implement session never ran (auth/limit/network?) -> infra_error, no repair")
        elif not chk["ok"] and not st.get("repaired"):
            st["repaired"] = True
            _save_state(d, st)
            if do_repair(d, meta, st, chk):
                try:
                    jobs = submit_gpu_and_review(d, meta, "out2", 2)
                except Exception as e:  # noqa: BLE001 - still write a result for the first attempt
                    warn(f"resubmission failed: {e}")
                    st["repair"]["resubmit_error"] = str(e)
                else:
                    meta.setdefault("jobs", []).extend(jobs)
                    atomic_write_json(d / "meta.json", meta)
                    _save_state(d, st)
                    info("repair smoke ok -> resubmitted GPU run: " + ", ".join(j["job"] for j in jobs))
                    return 0
            _save_state(d, st)
    finalize(d, meta, st)
    return 0


# --------------------------------------------------------------------------- ingest
def cmd_ingest(args) -> int:
    J = _journal_mod()
    run_dir = resolve_run_dir(args.run, args.runs_root)
    _warn_cross_host(run_dir)
    journal = J.Journal(run_dir)
    state = journal.read()
    task_cache: dict[str, dict] = {}
    todo, pending = [], []
    for iid, idea in state["ideas"].items():
        st = idea.get("status")
        if st == "promoted" or (args.rescore and st == "real_judged"):
            res = read_json(run_dir / "real" / iid / "result.json")
            if res is None:
                if st == "promoted":
                    pending.append(iid)
                continue
            if res.get("status") == "infra_error":
                warn(f"{iid}: infra_error (Claude never ran); left promoted -- resubmit with submit --force")
                continue
            tid = res.get("task_id") or state["config"]["task_id"]
            if tid not in task_cache:
                try:
                    task_cache[tid] = load_task(tid)
                except OSError:
                    task_cache[tid] = {"floor": res.get("floor"), "ref": res.get("ref")}
            t = task_cache[tid]
            rs, why = normalize(res.get("value"), t.get("floor"), t.get("ref"),
                                valid=res.get("valid"), adherence=res.get("adherence"))
            if rs is None:
                warn(f"{iid}: {why}")
            todo.append((iid, res, rs, why))
    now = time.time()
    with journal.transaction("realjudge", "ingest") as st:
        for iid, res, rs, why in todo:
            idea = st["ideas"][iid]
            prev = idea.get("real") or {}
            idea["real"] = {**{k: prev[k] for k in ("selection", "selected_at") if k in prev},
                            **{k: v for k, v in res.items() if k != "claude_usage"},
                            "claude_usage": res.get("claude_usage"), "real_score_note": why, "ingested_at": now}
            idea["real_score"] = rs
            if idea.get("status") != "real_judged":
                J.set_status(st, iid, "real_judged", now)
        pairs = [(float(i["s_cal"]), float(i["real_score"])) for i in st["ideas"].values()
                 if i.get("status") == "real_judged" and i.get("real_score") is not None and i.get("s_cal") is not None]
        rho, p = spearman(pairs)
        entry = {"t": now, "op": "ingest", "ingested": [i for i, *_ in todo], "pending": pending,
                 "n": len(pairs), "spearman": rho, "p": p,
                 "ideas": [{"id": i["id"], "s_cal": i.get("s_cal"), "real_score": i.get("real_score")}
                           for i in st["ideas"].values() if i.get("status") == "real_judged"]}
        st["logs"].setdefault("realjudge", []).append(entry)
    for iid, res, rs, _ in todo:
        info(f"ingested {iid}: value={res.get('value')} real_score={rs} adherence={res.get('adherence')} "
             f"fidelity={res.get('fidelity')} status={res.get('status')}")
    if pending:
        info(f"still pending (no result.json yet): {' '.join(pending)}")
    info(f"surrogate-real Spearman over {len(pairs)} real-judged ideas: rho={rho} p={p}")
    print(json.dumps({"ingested": len(todo), "pending": len(pending), "n": len(pairs), "spearman": rho, "p": p}))
    return 0


# --------------------------------------------------------------------------- CLI
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m ar.realjudge")
    sub = p.add_subparsers(dest="cmd", required=True)

    def run_args(sp):
        sp.add_argument("--run", required=True, help="run_id (under $AR_ROOT/runs) or a run dir")
        sp.add_argument("--runs-root", default=None)

    s = sub.add_parser("select", help="mark ideas promoted, print their ids")
    run_args(s)
    s.add_argument("--top", type=int, default=0)
    s.add_argument("--random", type=int, default=0)
    s.add_argument("--stratified", type=int, default=0, help="one pick per s_cal quantile bin (calibration)")
    s.add_argument("--seed", type=int, default=0)
    s = sub.add_parser("submit", help="build real/<id>/ and submit the Slurm chain")
    run_args(s)
    s.add_argument("--ids", nargs="+", required=True)
    s.add_argument("--dry-run", action="store_true", help="build the idea dirs only")
    s.add_argument("--force", action="store_true", help="rebuild existing dirs / allow non-promoted ideas")
    s = sub.add_parser("ingest", help="result.json -> journal; log Spearman")
    run_args(s)
    s.add_argument("--rescore", action="store_true", help="recompute real_score of already ingested ideas")
    s = sub.add_parser("job-implement", help=argparse.SUPPRESS)
    s.add_argument("--dir", required=True)
    s = sub.add_parser("job-review", help=argparse.SUPPRESS)
    s.add_argument("--dir", required=True)
    s.add_argument("--phase", type=int, default=1)
    return p


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    if a.cmd == "select":
        return cmd_select(a)
    if a.cmd == "submit":
        return cmd_submit(a)
    if a.cmd == "ingest":
        return cmd_ingest(a)
    if a.cmd == "job-implement":
        return job_implement(Path(a.dir))
    if a.cmd == "job-review":
        return job_review(Path(a.dir), a.phase)
    return 2


if __name__ == "__main__":
    sys.exit(main())
