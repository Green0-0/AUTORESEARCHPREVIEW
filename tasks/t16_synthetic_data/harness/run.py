"""Driver: synthesis (solution, subprocess, timed) -> validation -> fixed classifier eval -> metric.json."""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

TASK_ID = "t16_synthetic_data"
METRIC = "mean_top1_3arch"
BUDGET_S = {"full": 15 * 60, "smoke": 60, "cpu-smoke": 90}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("solution_dir")
    ap.add_argument("out_dir")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--smoke", action="store_true")
    g.add_argument("--cpu-smoke", action="store_true")
    ap.add_argument("--eval-only", action="store_true", help="skip synthesis; evaluate existing artifact")
    a = ap.parse_args()
    mode = "cpu-smoke" if a.cpu_smoke else "smoke" if a.smoke else "full"
    t_start = time.time()
    sol = Path(a.solution_dir).resolve()
    out = Path(a.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    art = out / "artifact"
    seed = int(os.environ.get("AR_SEED", "0"))
    ar_data = Path(os.environ.get("AR_DATA", "/usr/xtmp/kg349/ar/data"))
    budget = BUDGET_S[mode]
    report = {"mode": mode, "seed": seed, "synthesis_budget_s": budget, "errors": []}
    result = {"metric": METRIC, "value": 0.0, "valid": False, "constraint_report": report,
              "wall_s": 0.0, "mode": mode}

    def finish(code=0):
        result["wall_s"] = round(time.time() - t_start, 2)
        (out / "metric.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
        sys.exit(code)

    # ---------------- step 1: synthesis (solution code; sees only the 1000 real images)
    if not a.eval_only:
        real = ar_data / TASK_ID / "real10.npz"
        if not real.is_file():
            report["errors"].append(f"missing {real}; run tasks/{TASK_ID}/prepare_data.py")
            finish(2)
        inp = out / "input"
        inp.mkdir(exist_ok=True)
        shutil.copyfile(real, inp / "real10.npz")
        if art.exists():
            shutil.rmtree(art)
        art.mkdir()
        entry = sol / "synthesize.py"
        if not entry.is_file():
            report["errors"].append(f"solution entry point missing: {entry}")
            finish(2)
        env = dict(os.environ, AR_DATA=str(inp), AR_SEED=str(seed), T16_MODE=mode,
                   T16_BUDGET_S=str(budget))
        for k in ("AR_PRIVATE_DIR",):
            env.pop(k, None)
        if mode == "cpu-smoke":
            env["CUDA_VISIBLE_DEVICES"] = ""
        cmd = [sys.executable, str(entry), "--real", str(inp / "real10.npz"), "--out", str(art),
               "--seed", str(seed), "--budget-s", str(budget), "--mode", mode]
        t0 = time.time()
        timed_out = False
        with open(out / "synthesis.log", "w") as logf:
            p = subprocess.Popen(cmd, cwd=str(sol), env=env, stdout=logf, stderr=subprocess.STDOUT,
                                 start_new_session=True)
            try:
                rc = p.wait(timeout=budget)
            except subprocess.TimeoutExpired:
                timed_out = True
                os.killpg(p.pid, 9)
                rc = p.wait()
        report["synthesis_wall_s"] = round(time.time() - t0, 2)
        report["synthesis_returncode"] = rc
        report["synthesis_timed_out"] = timed_out
        if timed_out:
            report["errors"].append(f"synthesis exceeded budget {budget}s")
        elif rc != 0:
            report["errors"].append(f"synthesis exited {rc}; see synthesis.log")
        if timed_out or rc != 0:
            print((out / "synthesis.log").read_text()[-3000:])
            finish(1)

    # ---------------- step 2: validation
    import evaluate as E
    import torch
    ok, vrep, x, y = E.validate(art / "synthetic.npz")
    report.update({k: v for k, v in vrep.items() if k != "errors"})
    report["errors"] += vrep["errors"]
    others = [q.name for q in art.iterdir() if q.name != "synthetic.npz"] if art.exists() else []
    report["storage_bytes_artifact_dir"] = sum(q.stat().st_size for q in art.rglob("*") if q.is_file()) \
        if art.exists() else 0
    if others:
        report["artifact_extra_files_ignored"] = others
    if not ok:
        finish(1)

    # ---------------- step 3: fixed classifiers (harness cost, not counted in the synthesis budget)
    device = torch.device("cuda" if (mode != "cpu-smoke" and torch.cuda.is_available()) else "cpu")
    if mode != "cpu-smoke" and device.type != "cuda":
        report["errors"].append("no CUDA device for GPU mode")
        finish(2)
    t0 = time.time()
    try:
        mean, per_arch, walls = E.evaluate(x, y, ar_data, mode, seed, device)
    except E.EvalError as e:
        report["errors"].append(str(e))
        print(f"ERROR: {e}", file=sys.stderr)
        finish(3)
    report["classifier_wall_s"] = round(time.time() - t0, 2)
    report["per_arch_top1"] = per_arch
    report["per_arch_wall_s"] = walls
    result["value"] = mean
    result["valid"] = True
    finish(0)


if __name__ == "__main__":
    main()
