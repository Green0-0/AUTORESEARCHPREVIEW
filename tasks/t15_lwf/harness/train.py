"""t15_lwf training stream (harness-owned).

python train.py --solution DIR --out DIR --mode full|smoke|cpu-smoke

Builds the model with solution.build_model(), constructs solution.Learner(model, info), then for
t = 0..9 hands the learner ONLY task t's training tensors (GPU-resident uint8 images + labels),
zeroes those tensors in place afterwards and drops them, so earlier tasks cannot be revisited.
After every task the persistent state is audited (aux_state() size + reachability scan).
Finally it saves artifact/{model.pt, aux.pt} and train_report.json.
"""
import argparse, gc, json, os, sys, time, traceback
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C


def snapshot_files(*dirs):
    out = {}
    for d in dirs:
        d = Path(d)
        if d.exists():
            for p in d.rglob("*"):
                if p.is_file() and "__pycache__" not in p.parts:
                    try:
                        out[str(p)] = p.stat().st_mtime_ns
                    except OSError:
                        pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solution", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", default="full", choices=list(C.MODES))
    a = ap.parse_args()
    cfg = C.MODES[a.mode]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    art = out / "artifact"; art.mkdir(exist_ok=True)
    sol_dir = Path(a.solution).resolve()
    seed = int(os.environ.get("AR_SEED", "0"))
    rep = dict(mode=a.mode, seed=seed, budget_s=cfg["budget_s"], tasks=[], errors=[], ok=False)

    def dump():
        (out / "train_report.json").write_text(json.dumps(rep, indent=1, default=str))

    try:
        dev = torch.device(cfg["device"])
        if dev.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA not available (use --cpu-smoke on CPU nodes)")
        d = np.load(C.data_dir() / C.TASK_ID / "train.npz")
        x_all, y_all = d["x"], d["y"]
        keep = C.subset_per_class(y_all, cfg["train_per_class"], seed=0)
        x_all, y_all = x_all[keep], y_all[keep]
        # Pinned CPU copies only; each task is moved to the device on its turn.
        task_data = []
        for t in range(C.NUM_TASKS):
            m = (y_all >= t * C.CPT) & (y_all < (t + 1) * C.CPT)
            task_data.append((torch.from_numpy(x_all[m].copy()), torch.from_numpy(y_all[m].copy())))
        del d, x_all, y_all
        # The solution gets no path to the data files.
        for k in ("AR_DATA", "AR_ROOT"):
            os.environ.pop(k, None)

        torch.manual_seed(seed); np.random.seed(seed)
        files_before = snapshot_files(sol_dir)
        sol = C.load_solution(sol_dir)
        info = dict(num_tasks=C.NUM_TASKS, classes_per_task=C.CPT, num_classes=C.NUM_CLASSES,
                    device=str(dev), budget_s=cfg["budget_s"], mode=a.mode, seed=seed,
                    image_shape=(3, 32, 32), image_dtype="uint8 0..255 NCHW")
        clock = 0.0
        t0 = time.perf_counter()
        model = sol.build_model().to(dev)
        learner = sol.Learner(model, dict(info))
        if dev.type == "cuda":
            torch.cuda.synchronize()
        clock += time.perf_counter() - t0
        max_aux = max_scan = max_unacc = 0
        exempt_viol = False
        for t in range(C.NUM_TASKS):
            xc, yc = task_data[t]
            x = xc.to(dev, non_blocking=True).contiguous()
            y = yc.to(dev).contiguous()
            task_data[t] = None
            del xc, yc
            classes = list(range(t * C.CPT, (t + 1) * C.CPT))
            t0 = time.perf_counter()
            learner.learn_task(t, x, y, dict(classes=classes, time_left_s=cfg["budget_s"] - clock))
            if dev.type == "cuda":
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            clock += dt
            # Destroy this task's data: any retained alias now sees zeros.
            x.zero_(); y.zero_()
            # --- audit (not on the solution clock)
            aux = learner.aux_state()
            aux_bytes = C.serialized_size(aux)
            del aux
            acc = C.account(learner, sol_dir)
            del x, y
            gc.collect()
            unacc = 0
            if dev.type == "cuda":
                torch.cuda.synchronize()
                try:  # cuBLAS workspaces live in the caching allocator; not solution state
                    torch._C._cuda_clearCublasWorkspaces()
                except Exception:
                    pass
                unacc = max(0, torch.cuda.memory_allocated() - acc["cuda_reached_bytes"])
            ex_ok = acc["param_shaped_bytes"] <= C.EXEMPT_FACTOR * acc["model_param_bytes"]
            exempt_viol |= not ex_ok
            max_aux, max_scan, max_unacc = max(max_aux, aux_bytes), max(max_scan, acc["scanned_aux_bytes"]), max(max_unacc, unacc)
            rec = dict(task=t, learn_s=round(dt, 2), clock_s=round(clock, 2), aux_state_bytes=aux_bytes,
                       scanned_aux_bytes=acc["scanned_aux_bytes"], param_shaped_bytes=acc["param_shaped_bytes"],
                       model_param_bytes=acc["model_param_bytes"], unaccounted_cuda_bytes=unacc,
                       top_aux=acc["top_aux"][:4])
            rep["tasks"].append(rec)
            print(json.dumps({k: v for k, v in rec.items() if k != "top_aux"}), flush=True)
            dump()
            if clock > cfg["budget_s"]:
                rep["errors"].append(f"budget exceeded after task {t}: {clock:.1f}s > {cfg['budget_s']}s")
                break
        # Final artifact: harness serializes model weights + aux state itself.
        aux = learner.aux_state()
        torch.save(learner.model.state_dict(), art / "model.pt")
        torch.save(aux, art / "aux.pt")
        new_files = sorted(p for p, m in snapshot_files(sol_dir).items() if files_before.get(p) != m)
        rep.update(
            stream_s=round(clock, 2), tasks_completed=len(rep["tasks"]),
            max_aux_state_bytes=max_aux, max_scanned_aux_bytes=max_scan, max_unaccounted_cuda_bytes=max_unacc,
            final_aux_file_bytes=(art / "aux.pt").stat().st_size, files_written_in_solution=new_files,
            aux_limit_bytes=C.AUX_LIMIT_BYTES)
        checks = {
            "all_tasks_seen": len(rep["tasks"]) == C.NUM_TASKS,
            "within_budget": clock <= cfg["budget_s"],
            "aux_state_le_1MiB": max_aux <= C.AUX_LIMIT_BYTES and rep["final_aux_file_bytes"] <= C.AUX_LIMIT_BYTES,
            "scanned_aux_le_1MiB": max_scan <= C.AUX_LIMIT_BYTES,
            "param_shaped_copies_le_8x_model": not exempt_viol,
            "no_hidden_cuda_memory": max_unacc <= C.UNACCOUNTED_GPU_LIMIT,
            "no_files_written": not new_files,
        }
        rep["checks"] = checks
        rep["ok"] = True
    except Exception as e:
        rep["errors"].append("".join(traceback.format_exception(e))[-4000:])
        traceback.print_exc()
    dump()
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
