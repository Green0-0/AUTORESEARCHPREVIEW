"""t10_label_noise harness driver: static check -> train (own process, offline sandbox, timeout)
-> eval (own process, timeout) -> <out_dir>/metric.json.
usage: run.py <solution_dir> <out_dir> [--smoke | --cpu-smoke]"""
import json, os, shutil, subprocess, sys, time
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import MODES, ARTIFACT_MAX_BYTES  # noqa: E402
from check import static_check, sandbox_check  # noqa: E402


def sandbox_env(sb: Path, mode: str):
    env = dict(os.environ)
    for k in ("TORCH_HOME", "HF_HOME", "XDG_CACHE_HOME", "TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR"):
        p = sb / k.lower(); p.mkdir(parents=True, exist_ok=True); env[k] = str(p)
    env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1",
               http_proxy="http://127.0.0.1:9", https_proxy="http://127.0.0.1:9",
               HTTP_PROXY="http://127.0.0.1:9", HTTPS_PROXY="http://127.0.0.1:9", ALL_PROXY="http://127.0.0.1:9",
               PYTHONDONTWRITEBYTECODE="1", PYTHONHASHSEED=env.get("AR_SEED", "0"))
    env.pop("no_proxy", None); env.pop("NO_PROXY", None)
    if mode == "cpu-smoke":
        env["CUDA_VISIBLE_DEVICES"] = ""
    return env


def step(script, args, env, kill_s, log):
    t0 = time.time()
    with open(log, "w") as lf:
        try:
            rc = subprocess.run([sys.executable, str(HERE / script), *args], env=env, stdout=lf,
                                stderr=subprocess.STDOUT, timeout=kill_s).returncode
            timed_out = False
        except subprocess.TimeoutExpired:
            rc, timed_out = -9, True
    return rc, timed_out, time.time() - t0


def main():
    sol, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
    mode = "full"
    if "--smoke" in sys.argv[3:]: mode = "smoke"
    if "--cpu-smoke" in sys.argv[3:]: mode = "cpu-smoke"
    cfg = MODES[mode]
    t_all = time.time()
    out.mkdir(parents=True, exist_ok=True)
    art, sb = out / "artifact", out / "sandbox"
    shutil.rmtree(art, ignore_errors=True); shutil.rmtree(sb, ignore_errors=True)
    env = sandbox_env(sb, mode)
    rep = {"static": static_check(sol)}

    rc, to, wall = step("train_entry.py", [str(sol), str(art), mode, str(out / "train_report.json")], env, cfg["train_kill_s"], out / "train.log")
    tr = json.loads((out / "train_report.json").read_text()) if (out / "train_report.json").exists() else {}
    train_s = tr.get("train_s", wall)
    rep["train"] = {"ok": rc == 0 and not to, "returncode": rc, "killed_by_timeout": to, "train_s": round(train_s, 2),
                    "process_wall_s": round(wall, 2), "budget_s": cfg["train_budget_s"],
                    "within_budget": (not to) and train_s <= cfg["train_budget_s"], "error": tr.get("error")}
    asize = sum(p.stat().st_size for p in art.rglob("*") if p.is_file()) if art.exists() else 0
    rep["artifact"] = {"bytes": asize, "ok": 0 < asize <= ARTIFACT_MAX_BYTES}

    ev = {}
    if rep["train"]["ok"]:
        rc, to, ewall = step("eval_entry.py", [str(sol), str(art), mode, str(out / "eval_report.json")], env, cfg["eval_limit_s"], out / "eval.log")
        ev = json.loads((out / "eval_report.json").read_text()) if (out / "eval_report.json").exists() else {}
        rep["eval"] = {"ok": rc == 0 and not to and ev.get("ok", False), "killed_by_timeout": to, "process_wall_s": round(ewall, 2),
                       "limit_s": cfg["eval_limit_s"], "error": ev.get("error"), "n": ev.get("n")}
    else:
        rep["eval"] = {"ok": False, "error": "skipped: training failed"}
    rep["sandbox"] = sandbox_check(sb)

    valid = all(rep[k]["ok"] for k in ("static", "train", "artifact", "eval", "sandbox")) and rep["train"]["within_budget"]
    metric = {"metric": ev.get("metric", "clean_test_top1" if mode == "full" else "noisy_holdout_top1"),
              "value": float(ev.get("value", 0.0)), "valid": bool(valid), "constraint_report": rep,
              "wall_s": round(time.time() - t_all, 2), "mode": mode}
    (out / "metric.json").write_text(json.dumps(metric, indent=1))
    shutil.rmtree(sb, ignore_errors=True)
    print(json.dumps({k: metric[k] for k in ("metric", "value", "valid", "wall_s", "mode")}))
    if not valid:
        print("INVALID:", json.dumps({k: v for k, v in rep.items() if not v.get("ok", True) or k == "train"}, indent=1)[:3000])


if __name__ == "__main__":
    main()
