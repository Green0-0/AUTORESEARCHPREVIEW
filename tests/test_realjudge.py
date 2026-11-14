"""Real-judge tests: fake journal, fake `claude` and fake `sbatch` executables; no submissions, no network."""
from __future__ import annotations

import contextlib
import json
import os
import subprocess
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ar import realjudge as rj


# ----------------------------------------------------------------------------- fake journal module
class FakeJournal:
    _lock = threading.Lock()

    def __init__(self, run_dir):
        self.run_dir = Path(run_dir)
        self.path = self.run_dir / "journal.json"

    def read(self):
        return json.loads(self.path.read_text())

    @contextlib.contextmanager
    def transaction(self, worker, op):
        with self._lock:
            st = self.read()
            yield st
            self.path.write_text(json.dumps(st))


def fake_set_status(state, idea_id, status, now):
    idea = state["ideas"][idea_id]
    idea["status"] = status
    idea.setdefault("times", {})[status] = now
    idea["claim"] = None


FAKE_J = SimpleNamespace(Journal=FakeJournal, set_status=fake_set_status)

# ----------------------------------------------------------------------------- fake executables
FAKE_CLAUDE = r'''#!/usr/bin/env python3
import json, os, re, sys, uuid, pathlib
args = sys.argv[1:]
prompt = sys.stdin.read()
cwd = pathlib.Path.cwd()
log = pathlib.Path(os.environ["FAKE_LOG"])
with open(log, "a") as f:
    f.write(json.dumps({"argv": args, "cwd": str(cwd), "api_key": os.environ.get("ANTHROPIC_API_KEY"),
                        "prompt_head": prompt[:200]}) + "\n")
mode = os.environ.get("FAKE_CLAUDE_MODE", "ok")
if mode == "dead":
    sys.exit(1)
resume = args[args.index("--resume") + 1] if "--resume" in args else None
sid = resume or str(uuid.uuid4())
if "strict, read-only reviewer" in prompt:
    text = 'Looks fine.\n```json\n{"adherence": 1, "fidelity": 0.6, "adherence_reason": "ok", "fidelity_reason": "mostly"}\n```'
elif resume:
    (cwd / "solution" / "model.py").write_text("print('repaired')\n")
    text = "repaired"
else:
    (cwd / "solution" / "model.py").write_text(os.environ.get("FAKE_IMPL", "ok") + "\n")
    if mode == "tamper":
        p = cwd / "harness" / "eval.py"
        os.chmod(p, 0o644); p.write_text("cheat\n")
    if mode == "nochange":
        (cwd / "solution" / "model.py").write_text("STARTER\n")
    text = "implemented"
if "--no-session-persistence" not in args:
    home = pathlib.Path(os.environ["AR_CLAUDE_HOME"])
    pd = home / "projects" / re.sub(r"[^A-Za-z0-9]", "-", str(cwd))
    pd.mkdir(parents=True, exist_ok=True)
    with open(pd / f"{sid}.jsonl", "a") as f:
        f.write(json.dumps({"prompt": prompt[:50]}) + "\n")
    (home / "file-history" / sid).mkdir(parents=True, exist_ok=True)
print(json.dumps({"type": "result", "subtype": "success", "is_error": False, "result": text,
                  "session_id": sid, "num_turns": 3, "duration_ms": 1000, "total_cost_usd": 0.01,
                  "usage": {"input_tokens": 10, "output_tokens": 20}}))
'''

FAKE_SBATCH = r'''#!/usr/bin/env python3
import json, os, sys, pathlib
p = pathlib.Path(os.environ["FAKE_SBATCH_LOG"])
n = len(p.read_text().splitlines()) if p.exists() else 0
with open(p, "a") as f:
    f.write(json.dumps(sys.argv[1:]) + "\n")
print(f"{1000 + n};cluster")
'''

RUN_SH = r'''#!/bin/bash
# fake harness: run.sh <solution> <out> [--cpu-smoke|--smoke]
SOL=$1; OUT=$2; MODE=${3:-full}
mkdir -p "$OUT"
if grep -q CRASH "$SOL/model.py"; then echo "Traceback: boom" >&2; exit 1; fi
VAL=${FAKE_VALUE:-0.35}
python3 -c "import json,sys; json.dump({'metric':'top1','value':float(sys.argv[1]),'valid':True,'constraint_report':{},'wall_s':1.0,'mode':sys.argv[2]}, open(sys.argv[3],'w'))" "$VAL" "${MODE#--}" "$OUT/metric.json"
'''


def _exe(p: Path, text: str):
    p.write_text(text)
    p.chmod(0o755)


@pytest.fixture
def env(tmp_path, monkeypatch):
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    _exe(bin_ / "claude", FAKE_CLAUDE)
    _exe(bin_ / "sbatch", FAKE_SBATCH)
    tasks = tmp_path / "tasks"
    t = tasks / "tt"
    (t / "harness").mkdir(parents=True)
    (t / "starter").mkdir()
    (t / "TASK.md").write_text("Train the best CIFAR model in 15 minutes.\n")
    (t / "task.yaml").write_text("id: tt\ntype: I\nsplit: train\nbudget_minutes: 15\nmetric: top1\n"
                                 "higher_is_better: true\nfloor: 0.1\nref: 0.6\n")
    (t / "harness" / "CONTRACT.md").write_text("provide solution/model.py\n")
    (t / "harness" / "eval.py").write_text("print('eval')\n")
    _exe(t / "harness" / "run.sh", RUN_SH)
    (t / "starter" / "model.py").write_text("STARTER\n")
    claude_home = tmp_path / "claude_home"
    claude_home.mkdir()
    monkeypatch.setenv("AR_TASKS", str(tasks))
    monkeypatch.setenv("AR_CLAUDE_BIN", str(bin_ / "claude"))
    monkeypatch.setenv("AR_NO_JAIL", "1")   # fake executables write to tmp dirs; jail tested separately
    monkeypatch.setenv("AR_SBATCH", str(bin_ / "sbatch"))
    monkeypatch.setenv("AR_CLAUDE_HOME", str(claude_home))
    monkeypatch.setenv("FAKE_LOG", str(tmp_path / "claude_calls.jsonl"))
    monkeypatch.setenv("FAKE_SBATCH_LOG", str(tmp_path / "sbatch_calls.jsonl"))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "must-not-leak")
    monkeypatch.setattr(rj, "_journal_mod", lambda: FAKE_J)
    run_dir = tmp_path / "runs" / "run1"
    run_dir.mkdir(parents=True)
    ideas = {}
    for k in range(20):
        iid = f"i{k:06d}"
        ideas[iid] = {"id": iid, "created_at": float(k), "status": "scored", "s_cal": round(0.05 * k, 3),
                      "F": 0.0, "description": f"Idea {k}: use {{{{prompt}}}} cosine schedule", "real_score": None,
                      "real": None, "claim": None, "times": {}}
    state = {"config": {"run_id": "run1", "task_id": "tt", "task_prompt_file": str(t / "TASK.md")},
             "ideas": ideas, "tags": {}, "logs": {"refits": [], "batches": [], "tagging": [], "realjudge": []},
             "counters": {}}
    (run_dir / "journal.json").write_text(json.dumps(state))
    return SimpleNamespace(tmp=tmp_path, run_dir=run_dir, tasks=tasks, claude_home=claude_home,
                           calls=tmp_path / "claude_calls.jsonl", sb=tmp_path / "sbatch_calls.jsonl")


def jstate(env):
    return json.loads((env.run_dir / "journal.json").read_text())


def sbatch_calls(env):
    return [json.loads(l) for l in env.sb.read_text().splitlines()] if env.sb.exists() else []


def claude_calls(env):
    return [json.loads(l) for l in env.calls.read_text().splitlines()] if env.calls.exists() else []


def fake_gpu_job(d: Path, out="out", value=None):
    """What gpu_run.sbatch does, minus slurm/nvidia-smi."""
    e = dict(os.environ)
    if value is not None:
        e["FAKE_VALUE"] = str(value)
    t0 = time.time()
    rc = subprocess.run(["bash", "harness/run.sh", "solution", out], cwd=d, env=e).returncode
    (d / out).mkdir(exist_ok=True)
    (d / out / "gpu_job.json").write_text(json.dumps({"rc": rc, "start": t0, "end": time.time()}))
    (d / "logs" / f"gpu_{out}_1.out").write_text("Traceback: boom\n" if rc else "ok\n")


def promote_and_submit(env, ids):
    st = jstate(env)
    for i in ids:
        st["ideas"][i]["status"] = "promoted"
    (env.run_dir / "journal.json").write_text(json.dumps(st))
    assert rj.main(["submit", "--run", str(env.run_dir), "--ids", *ids]) == 0
    return [env.run_dir / "real" / i for i in ids]


# ----------------------------------------------------------------------------- unit tests
def test_normalize():
    assert rj.normalize(0.35, 0.1, 0.6) == (pytest.approx(0.5), None)
    assert rj.normalize(0.9, 0.1, 0.6)[0] == 1.0
    assert rj.normalize(0.0, 0.1, 0.6)[0] == 0.0
    assert rj.normalize(2.0, 4.0, 1.0)[0] == pytest.approx(2 / 3)     # lower-is-better: ref < floor
    assert rj.normalize(0.35, None, 0.6)[0] is None
    assert rj.normalize(None, 0.1, 0.6)[0] == 0.0
    assert rj.normalize(float("nan"), 0.1, 0.6)[0] == 0.0
    assert rj.normalize(0.5, 0.1, 0.6, valid=False)[0] == 0.0
    assert rj.normalize(0.5, 0.1, 0.6, adherence=0)[0] == 0.0
    assert rj.normalize(0.35, 0.1, 0.6, adherence=0.5)[0] == pytest.approx(0.5)


def test_parse_review_and_render():
    r = rj.parse_review('blah {"adherence": 0} ```json\n{"adherence": 0.4, "fidelity": 1}\n```')
    assert r["adherence"] == 0.5 and r["fidelity"] == 1.0
    assert rj.parse_review("no json")["adherence"] is None
    out = rj.render("A {{IDEA}} B {{TASK_ID}} {{UNKNOWN}}", {"IDEA": "x {{TASK_ID}}", "TASK_ID": "t"})
    assert out == "A x {{TASK_ID}} B t {{UNKNOWN}}"      # substituted text is not re-rendered


def test_choose_top_stratified_random(env):
    st = jstate(env)
    st["ideas"]["i000019"]["status"] = "real_judged"     # not eligible
    picks = rj.choose(st, top=3, random_n=2, stratified=4, rng=np.random.default_rng(0))
    ids = [i for i, _ in picks]
    assert len(ids) == len(set(ids)) == 9
    assert [i for i, w in picks if w == "top"] == ["i000018", "i000017", "i000016"]
    strat = [int(i[1:]) for i, w in picks if w == "stratified"]
    assert len(strat) == 4 and min(strat) <= 3 and max(strat) >= 12     # spread over the s_cal range
    assert "i000019" not in ids


def test_select_cli_marks_promoted(env, capsys):
    assert rj.main(["select", "--run", str(env.run_dir), "--top", "2", "--random", "1", "--seed", "1"]) == 0
    printed = capsys.readouterr().out.split()
    st = jstate(env)
    assert len(printed) == 3
    for i in printed:
        assert st["ideas"][i]["status"] == "promoted"
        assert st["ideas"][i]["real"]["selection"] in ("top", "random")
    assert st["logs"]["realjudge"][-1]["op"] == "select"


def test_submit_builds_dir_and_chain(env):
    (d,) = promote_and_submit(env, ["i000005"])
    for f in ("task.md", "idea.md", "task.yaml", "implement_prompt.md", "repair_template.md",
              "review_template.md", "harness_hashes.json", "meta.json", "harness/run.sh", "solution/model.py"):
        assert (d / f).exists(), f
    hashes = json.loads((d / "harness_hashes.json").read_text())
    assert set(hashes) == {"CONTRACT.md", "eval.py", "run.sh"}
    assert hashes["eval.py"] == rj.sha256_file(env.tasks / "tt" / "harness" / "eval.py")
    assert not os.access(d / "harness" / "eval.py", os.W_OK)
    assert not os.access(d / "harness", os.W_OK)
    assert os.access(d / "harness" / "run.sh", os.X_OK)
    assert os.access(d / "solution" / "model.py", os.W_OK)
    p = (d / "implement_prompt.md").read_text()
    assert "Idea 5: use {{prompt}} cosine schedule" in p          # idea verbatim, not re-rendered
    assert "Train the best CIFAR model" in p
    for must in ("faithfully", "harness/CONTRACT.md", "Only write inside `solution/`", "--cpu-smoke",
                 "No access to test data", "15 minutes"):
        assert must in p, must
    calls = sbatch_calls(env)
    assert len(calls) == 3
    assert calls[0][-2].endswith("cpu_implement.sbatch") and calls[0][-1] == str(d)
    assert calls[1][-3].endswith("gpu_run.sbatch") and calls[1][-1] == "out"
    assert "--dependency=afterok:1000" in calls[1] and "--kill-on-invalid-dep=yes" in calls[1]
    assert calls[1][calls[1].index("-t") + 1] == "00:30:00"
    assert calls[2][-3].endswith("cpu_review.sbatch") and calls[2][-1] == "1"
    assert "--dependency=afterany:1001" in calls[2]
    meta = json.loads((d / "meta.json").read_text())
    assert [j["job"] for j in meta["jobs"]] == ["1000", "1001", "1002"]
    # second submit without --force is refused
    rj.main(["submit", "--run", str(env.run_dir), "--ids", "i000005"])
    assert len(sbatch_calls(env)) == 3


def test_submit_dry_run_and_not_promoted(env):
    assert rj.main(["submit", "--run", str(env.run_dir), "--ids", "i000001"]) == 0   # status scored: refused
    assert not (env.run_dir / "real" / "i000001").exists()
    st = jstate(env)
    st["ideas"]["i000001"]["status"] = "promoted"
    (env.run_dir / "journal.json").write_text(json.dumps(st))
    assert rj.main(["submit", "--run", str(env.run_dir), "--ids", "i000001", "--dry-run"]) == 0
    assert (env.run_dir / "real" / "i000001" / "harness").exists()
    assert sbatch_calls(env) == []


def test_full_chain_happy_path_and_ingest(env, monkeypatch):
    ids = ["i000004", "i000010", "i000016"]
    dirs = promote_and_submit(env, ids)
    for d, v in zip(dirs, (0.2, 0.3, 0.5)):
        assert rj.job_implement(d) == 0
        smoke = json.loads((d / "out_cpu_smoke" / "metric.json").read_text())
        assert smoke["mode"] == "cpu-smoke"
        fake_gpu_job(d, value=v)
        assert rj.job_review(d, 1) == 0
        res = json.loads((d / "result.json").read_text())
        assert res["status"] == "ok" and res["valid"] is True and res["repaired"] is False
        assert res["value"] == v
        assert res["real_score"] == pytest.approx((v - 0.1) / 0.5)
        assert res["adherence"] == 1.0 and res["fidelity"] == 0.5
        assert [c["phase"] for c in res["claude_usage"]] == ["implement", "review"]
        # transcripts copied into the idea dir, removed from ~/.claude
        assert list((d / "claude_transcripts").rglob("*.jsonl"))
        enc = rj._enc_project(d)
        assert not (env.claude_home / "projects" / enc).exists()
        assert not any((env.claude_home / "file-history").iterdir())
    calls = claude_calls(env)
    assert all(c["api_key"] is None for c in calls), "ANTHROPIC_API_KEY must be unset for claude"
    impl = [c for c in calls if "--no-session-persistence" not in c["argv"]]
    for c in impl:
        a = c["argv"]
        assert a[0] == "-p" and a[a.index("--model") + 1] == "sonnet" and a[a.index("--max-turns") + 1] == "60"
        assert "--dangerously-skip-permissions" not in a and a[a.index("--permission-mode") + 1] == "dontAsk" and a[a.index("--output-format") + 1] == "json"
    review = [c for c in calls if "--no-session-persistence" in c["argv"]]
    assert len(review) == 3 and all("Write" not in c["argv"] for c in review)
    # no extra sbatch from the happy path
    assert len(sbatch_calls(env)) == 9
    assert rj.main(["ingest", "--run", str(env.run_dir)]) == 0
    st = jstate(env)
    for i, v in zip(ids, (0.2, 0.3, 0.5)):
        idea = st["ideas"][i]
        assert idea["status"] == "real_judged"
        assert idea["real_score"] == pytest.approx((v - 0.1) / 0.5)
        assert idea["real"]["value"] == v
    log = st["logs"]["realjudge"][-1]
    assert log["op"] == "ingest" and log["n"] == 3 and log["spearman"] == pytest.approx(1.0)


def test_repair_path(env, monkeypatch):
    monkeypatch.setenv("FAKE_IMPL", "CRASH")
    (d,) = promote_and_submit(env, ["i000007"])
    assert rj.job_implement(d) == 1                      # smoke failed -> GPU job (afterok) never runs
    # slurm would cancel the GPU job; the review job runs afterany
    assert rj.job_review(d, 1) == 0
    assert not (d / "result.json").exists()
    calls = claude_calls(env)
    repair = calls[-1]["argv"]
    sid = json.loads((d / "state.json").read_text())["session_id"]
    assert repair[repair.index("--resume") + 1] == sid
    assert "CPU smoke run failed" in (d / "repair_prompt.md").read_text()
    sb = sbatch_calls(env)
    assert len(sb) == 5
    assert sb[3][-3].endswith("gpu_run.sbatch") and sb[3][-1] == "out2"
    assert not any(a.startswith("--dependency") for a in sb[3])
    assert sb[4][-1] == "2" and "--dependency=afterany:1003" in sb[4]
    # transcripts kept in ~/.claude until the final review (resume needs them)
    assert (env.claude_home / "projects" / rj._enc_project(d)).exists()
    fake_gpu_job(d, out="out2", value=0.6)
    assert rj.job_review(d, 2) == 0
    res = json.loads((d / "result.json").read_text())
    assert res["repaired"] is True and res["status"] == "ok" and res["out"] == "out2"
    assert res["real_score"] == 1.0
    assert not (env.claude_home / "projects" / rj._enc_project(d)).exists()


def test_gpu_crash_repair_once_then_fail(env, monkeypatch):
    (d,) = promote_and_submit(env, ["i000008"])
    assert rj.job_implement(d) == 0
    (d / "solution" / "model.py").write_text("CRASH\n")   # e.g. a GPU-only bug
    fake_gpu_job(d)
    monkeypatch.setenv("FAKE_CLAUDE_MODE", "ok")
    assert rj.job_review(d, 1) == 0                      # repair ok -> resubmitted
    assert "crashed" in (d / "repair_prompt.md").read_text()
    (d / "solution" / "model.py").write_text("CRASH\n")
    fake_gpu_job(d, out="out2")
    assert rj.job_review(d, 2) == 0
    res = json.loads((d / "result.json").read_text())
    assert res["status"] == "crash" and res["real_score"] == 0.0 and res["repaired"] is True
    assert len(sbatch_calls(env)) == 5                   # only one resubmission ever


def test_harness_tamper_detected(env, monkeypatch):
    monkeypatch.setenv("FAKE_CLAUDE_MODE", "tamper")
    (d,) = promote_and_submit(env, ["i000009"])
    assert rj.job_implement(d) == 0
    assert (d / "harness" / "eval.py").read_text() == "print('eval')\n"     # restored
    assert not os.access(d / "harness" / "eval.py", os.W_OK)
    fake_gpu_job(d)
    monkeypatch.setenv("FAKE_CLAUDE_MODE", "ok")
    rj.job_review(d, 1)
    res = json.loads((d / "result.json").read_text())
    assert res["harness_tampered"] and res["adherence"] == 0.0 and res["real_score"] == 0.0


def test_unchanged_solution_and_dead_claude(env, monkeypatch):
    monkeypatch.setenv("FAKE_CLAUDE_MODE", "dead")
    (d,) = promote_and_submit(env, ["i000011"])
    assert rj.job_implement(d) == 1
    assert rj.job_review(d, 1) == 0
    res = json.loads((d / "result.json").read_text())
    assert res["status"] == "infra_error"
    assert len(sbatch_calls(env)) == 3                   # no repair resubmission
    assert rj.main(["ingest", "--run", str(env.run_dir)]) == 0
    assert jstate(env)["ideas"]["i000011"]["status"] == "promoted"    # left for resubmission


def test_ingest_null_floor_ref(env, monkeypatch):
    (env.tasks / "tt" / "task.yaml").write_text("id: tt\nbudget_minutes: 15\nmetric: top1\nfloor: null\nref: null\n")
    (d,) = promote_and_submit(env, ["i000012"])
    rj.job_implement(d)
    fake_gpu_job(d)
    rj.job_review(d, 1)
    res = json.loads((d / "result.json").read_text())
    assert res["value"] == 0.35 and res["real_score"] is None
    rj.main(["ingest", "--run", str(env.run_dir)])
    idea = jstate(env)["ideas"]["i000012"]
    assert idea["status"] == "real_judged" and idea["real_score"] is None and idea["real"]["value"] == 0.35
    # later the anchors are filled in -> --rescore
    (env.tasks / "tt" / "task.yaml").write_text("id: tt\nbudget_minutes: 15\nmetric: top1\nfloor: 0.1\nref: 0.6\n")
    rj.main(["ingest", "--run", str(env.run_dir), "--rescore"])
    assert jstate(env)["ideas"]["i000012"]["real_score"] == pytest.approx(0.5)


def test_sbatch_scripts_shape():
    gpu = (rj.RJ_DIR / "gpu_run.sbatch").read_text()
    assert "--gres=gpu:a5000:1" in gpu and "nvidia-smi dmon -s um -d 5" in gpu
    assert "claude" not in gpu.replace("Claude", "")     # the GPU job never runs Claude
    for f in ("cpu_implement.sbatch", "cpu_review.sbatch"):
        t = (rj.RJ_DIR / f).read_text()
        assert "#SBATCH -p compsci\n" in t and "gres" not in t
    for f in ("gpu_run.sbatch", "cpu_implement.sbatch", "cpu_review.sbatch"):
        r = subprocess.run(["bash", "-n", str(rj.RJ_DIR / f)], capture_output=True)
        assert r.returncode == 0, r.stderr


# ----------------------------------------------------------------------------- CPU-only tasks (task.yaml device: cpu)
def _make_cpu_task(env, cpus="12"):
    t = env.tasks / "tt"
    y = (t / "task.yaml").read_text() + "budget_minutes: 30\ndevice: cpu\n" + (f"cpus: {cpus}\n" if cpus else "")
    (t / "task.yaml").write_text(y.replace("budget_minutes: 15\n", "") + "packages: numpy, gymnasium[box2d]\n")


def test_run_device_and_gpu_default():
    assert rj.run_device({"task": {}}) == ("gpu", None)
    assert rj.run_device({"task": {"device": "GPU"}}) == ("gpu", None)
    assert rj.run_device({"task": {"device": "cpu", "cpus": 8}}) == ("cpu", 8)
    assert rj.run_device({"task": {"device": "cpu"}}) == ("cpu", 16)
    assert rj.run_device({}) == ("gpu", None)


def test_gpu_prompt_rendering_unchanged():
    # GPU tasks must render exactly the pre-placeholder wording
    v = {"TASK_ID": "x", "TASK_PROMPT": "P", "IDEA": "I", "BUDGET_MINUTES": "15", "METRIC": "m", **rj.compute_values({})}
    p = rj.render((rj.RJ_DIR / "claude_implement_prompt.md").read_text(), v)
    assert ("the full run gets **15 minutes on one A5000 GPU**, enforced by a timeout; overrunning it makes the "
            "result invalid. Keep the GPU busy (data on GPU, GPU augmentation, AMP, large batches; no CPU "
            "DataLoader bottleneck).\n") in p
    assert "6. Only use packages already installed (torch, torchvision, numpy). Do not install" in p
    r = rj.render((rj.RJ_DIR / "claude_repair_prompt.md").read_text(), v)
    assert "the full run has 15 minutes on one A5000 (timeout enforced" in r
    assert "{{" not in p and "{{COMPUTE" not in r


def test_submit_cpu_task_uses_cpu_run(env):
    _make_cpu_task(env)
    (d,) = promote_and_submit(env, ["i000005"])
    calls = sbatch_calls(env)
    assert len(calls) == 3
    c = calls[1]
    assert c[-3].endswith("cpu_run.sbatch") and c[-2] == str(d) and c[-1] == "out"
    assert c[c.index("-p") + 1] == rj.cpu_partition() and c[c.index("-c") + 1] == "12"
    assert not any("gres" in a for a in c) and not any(a.endswith("gpu_run.sbatch") for a in c)
    assert c[c.index("-J") + 1] == "rj-cpu-i000005"
    assert c[c.index("-t") + 1] == "00:45:00"                         # 30 min budget * 1.5
    assert "--dependency=afterok:1000" in c and "--kill-on-invalid-dep=yes" in c
    assert calls[2][-3].endswith("cpu_review.sbatch") and "--dependency=afterany:1001" in calls[2]
    meta = json.loads((d / "meta.json").read_text())
    assert [j["step"] for j in meta["jobs"]] == ["implement", "cpu:out", "review1"]
    p = (d / "implement_prompt.md").read_text()
    assert "**30 minutes on 12 CPU cores (no GPU)**" in p and "A5000" not in p and "Keep all 12 cores busy" in p
    assert "packages already installed (numpy, gymnasium[box2d])" in p
    assert "30 minutes on 12 CPU cores, no GPU (timeout" in (d / "repair_template.md").read_text()


def test_cpu_task_repair_resubmits_cpu_run(env, monkeypatch):
    _make_cpu_task(env, cpus=None)                                     # default 16 cores
    monkeypatch.setenv("FAKE_IMPL", "CRASH")
    (d,) = promote_and_submit(env, ["i000007"])
    assert rj.job_implement(d) == 1
    assert rj.job_review(d, 1) == 0
    sb = sbatch_calls(env)
    assert len(sb) == 5
    assert sb[3][-3].endswith("cpu_run.sbatch") and sb[3][-1] == "out2"
    assert sb[3][sb[3].index("-c") + 1] == "16" and sb[3][sb[3].index("-p") + 1] == rj.cpu_partition()
    fake_gpu_job(d, out="out2", value=0.6)                             # cpu_run.sbatch writes the same gpu_job.json
    assert rj.job_review(d, 2) == 0
    res = json.loads((d / "result.json").read_text())
    assert res["repaired"] is True and res["status"] == "ok" and res["out"] == "out2"


def test_cpu_run_sbatch_shape():
    cpu = (rj.RJ_DIR / "cpu_run.sbatch").read_text()
    gpu = (rj.RJ_DIR / "gpu_run.sbatch").read_text()
    assert "#SBATCH -p compsci\n" in cpu and "gres" not in cpu and "nvidia-smi" not in cpu
    assert "claude" not in cpu.replace("Claude", "")
    # same jail invocation and output file as the GPU job
    jail = 'bash "$AR_REPO/realjudge/jail.sh" "$PWD" bash harness/run.sh solution "$OUT_NAME"'
    assert jail in cpu and jail in gpu and "JAIL_MODE=harness" in cpu
    assert '"$OUT_NAME/gpu_job.json"' in cpu and "exit $rc" in cpu
    r = subprocess.run(["bash", "-n", str(rj.RJ_DIR / "cpu_run.sbatch")], capture_output=True)
    assert r.returncode == 0, r.stderr
