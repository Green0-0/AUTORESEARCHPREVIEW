"""Fixed training loop for t19_data_ordering. The solution only supplies the example order.

usage: python train.py --solution DIR --out DIR --mode full|smoke|cpu-smoke [--t0 EPOCH_SECONDS]
Writes <out>/artifact/model.pt and <out>/train_report.json.
"""
import argparse
import importlib.util
import json
import math
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402


class ScheduleError(Exception):
    pass


class BudgetExceeded(Exception):
    pass


class Context:
    """Everything the scheduler is allowed to see. Read-only by convention (tensors are copies)."""

    def __init__(self, h):
        self._h = h
        self.mode = h.mode
        self.device = h.device
        self.seed = common.seed_value()
        self.n_train = h.n
        self.num_classes = common.NUM_CLASSES
        self.batch_size = h.bs
        self.total_steps = h.total_steps
        self.total_presentations = h.bs * h.total_steps
        self.budget_s = h.budget_s
        self.images = h.x.clone()          # uint8 (N,3,32,32) on device; a copy, so edits do nothing
        self.labels = h.y.clone()          # int64 (N,) on device
        self.recipe = dict(common.OPT, batch_size=h.bs, total_steps=h.total_steps)

    def time_left(self) -> float:
        """Seconds left in the whole training-process budget."""
        return self._h.budget_s - (time.time() - self._h.t0)

    def normalize(self, x_uint8):
        """The harness's fixed input normalization (no augmentation)."""
        return common.normalize(x_uint8)

    def probe(self, indices, augment=False, batch_size=2048):
        """Forward pass of the CURRENT model (eval mode, no grad, BN stats untouched) on training examples.
        Returns float32 logits (len(indices), 100) on device. Time is charged to the budget and logged."""
        return self._h.probe(indices, augment, batch_size)


class Harness:
    def __init__(self, args):
        self.t0 = args.t0 if args.t0 else time.time()
        cfg = common.MODES[args.mode]
        self.mode = args.mode
        self.device = torch.device(cfg["device"])
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("mode needs CUDA")
        self.bs, self.total_steps, self.budget_s = cfg["batch_size"], cfg["total_steps"], cfg["budget_s"]
        self.use_amp = self.device.type == "cuda"
        torch.backends.cudnn.benchmark = True
        xs, ys = common.load_train(cfg["n_train"])
        self.x = torch.from_numpy(xs).to(self.device)
        self.y = torch.from_numpy(ys).to(self.device)
        self.n = self.x.shape[0]
        self.model = common.build_model(self.device)
        self.aug_gen = torch.Generator(device=self.device).manual_seed(common.seed_value() + 1)
        self.probe_gen = torch.Generator(device=self.device).manual_seed(common.seed_value() + 2)
        decay, no_decay = [], []
        for p in self.model.parameters():
            (decay if p.ndim > 1 else no_decay).append(p)
        o = common.OPT
        self.opt = torch.optim.SGD([{"params": decay, "weight_decay": o["weight_decay"]},
                                    {"params": no_decay, "weight_decay": 0.0}],
                                   lr=o["lr"], momentum=o["momentum"], nesterov=o["nesterov"])
        warm = max(1, int(o["warmup_frac"] * self.total_steps))
        T = self.total_steps

        def lr_lambda(s):  # linear warmup then linear decay to 0, as a function of the step only
            return (s + 1) / warm if s < warm else max(0.0, (T - s) / (T - warm))
        self.lr_sched = torch.optim.lr_scheduler.LambdaLR(self.opt, lr_lambda)
        self.counts = torch.zeros(self.n, dtype=torch.int64, device=self.device)
        self.bad = torch.zeros((), dtype=torch.int64, device=self.device)
        self.t = dict(sched_init=0.0, next_batch=0.0, observe=0.0, probe=0.0)
        self.probe_examples = 0
        self.probe_calls = 0

    def sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize()

    def check_time(self):
        if time.time() - self.t0 > self.budget_s:
            raise BudgetExceeded(f"wall budget {self.budget_s:.0f}s exceeded")

    # ---- probe ---------------------------------------------------------------------
    def probe(self, indices, augment, batch_size):
        t = time.time()
        self.sync()
        idx = self._as_index(indices, exact_len=None)
        out = []
        was = self.model.training
        self.model.eval()
        with torch.no_grad():
            for i in range(0, idx.shape[0], batch_size):
                xb = self.x[idx[i:i + batch_size]]
                xb = common.augment(xb, self.probe_gen) if augment else common.normalize(xb)
                xb = xb.contiguous(memory_format=torch.channels_last)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.use_amp):
                    out.append(self.model(xb).float())
        if was:
            self.model.train()
        res = torch.cat(out) if out else torch.empty(0, common.NUM_CLASSES, device=self.device)
        self.sync()
        self.t["probe"] += time.time() - t
        self.probe_examples += int(idx.shape[0])
        self.probe_calls += 1
        self.check_time()
        return res

    # ---- index validation ------------------------------------------------------------
    def _as_index(self, idx, exact_len):
        if isinstance(idx, np.ndarray):
            if idx.dtype.kind not in "iu":
                raise ScheduleError(f"indices must be integers, got numpy {idx.dtype}")
            idx = torch.from_numpy(np.ascontiguousarray(idx).astype(np.int64))
        elif isinstance(idx, (list, tuple)):
            if not all(isinstance(v, (int, np.integer)) and not isinstance(v, bool) for v in idx):
                raise ScheduleError("indices list must contain ints")
            idx = torch.tensor(idx, dtype=torch.int64)
        elif not isinstance(idx, torch.Tensor):
            raise ScheduleError(f"indices must be a tensor/ndarray/list, got {type(idx)}")
        if idx.dtype not in (torch.int64, torch.int32, torch.int16, torch.uint8, torch.int8):
            raise ScheduleError(f"indices must be an integer tensor, got {idx.dtype}")
        if idx.ndim != 1:
            raise ScheduleError(f"indices must be 1-D, got shape {tuple(idx.shape)}")
        if exact_len is not None and idx.shape[0] != exact_len:
            raise ScheduleError(f"batch must have exactly {exact_len} indices, got {idx.shape[0]}")
        if idx.device.type == "cpu":
            if idx.numel() and (int(idx.min()) < 0 or int(idx.max()) >= self.n):
                raise ScheduleError(f"index out of range [0,{self.n})")
            return idx.to(self.device, torch.int64, non_blocking=False)
        idx = idx.to(self.device, torch.int64).clone()
        # GPU indices: accumulate the range check asynchronously (checked every 50 steps); clamp so the
        # gather itself can never fault. Any out-of-range index invalidates the run.
        self.bad += ((idx < 0) | (idx >= self.n)).sum()
        return idx.clamp_(0, self.n - 1)

    def check_bad(self):
        if int(self.bad) > 0:
            raise ScheduleError(f"{int(self.bad)} out-of-range indices returned (range [0,{self.n}))")

    # ---- main loop -------------------------------------------------------------------
    def run(self, solution_dir, out_dir):
        spec = importlib.util.spec_from_file_location("solution_scheduler", Path(solution_dir) / "scheduler.py")
        mod = importlib.util.module_from_spec(spec)
        sys.path.insert(0, str(Path(solution_dir).resolve()))
        spec.loader.exec_module(mod)
        ctx = Context(self)
        t = time.time()
        sched = mod.Scheduler(ctx)
        self.sync()
        self.t["sched_init"] = time.time() - t
        self.check_time()
        observe = getattr(sched, "observe", None)
        self.model.train()
        o = common.OPT
        loss_acc = torch.zeros((), device=self.device)
        log = []
        t_loop = time.time()
        for step in range(self.total_steps):
            t = time.time()
            raw = sched.next_batch(step)
            self.t["next_batch"] += time.time() - t
            idx = self._as_index(raw, exact_len=self.bs)
            xb = common.augment(self.x[idx], self.aug_gen).contiguous(memory_format=torch.channels_last)
            yb = self.y[idx]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.use_amp):
                logits = self.model(xb)
            losses = F.cross_entropy(logits.float(), yb, label_smoothing=o["label_smoothing"], reduction="none")
            loss = losses.mean()
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            self.opt.step()
            self.lr_sched.step()
            self.counts.index_add_(0, idx, torch.ones_like(idx))
            loss_acc += loss.detach()
            if observe is not None:
                t = time.time()
                observe(step, idx, losses.detach(), logits.detach().float())
                self.t["observe"] += time.time() - t
            if (step + 1) % 50 == 0 or step + 1 == self.total_steps:
                self.check_bad()
                self.check_time()
            if (step + 1) % 500 == 0 or step + 1 == self.total_steps:
                k = 500 if (step + 1) % 500 == 0 else (step + 1) % 500
                log.append(dict(step=step + 1, train_loss=float(loss_acc) / k, elapsed=round(time.time() - self.t0, 1)))
                loss_acc.zero_()
                print(f"step {step + 1}/{self.total_steps} loss {log[-1]['train_loss']:.4f} t={log[-1]['elapsed']}s", flush=True)
        self.sync()
        loop_s = time.time() - t_loop
        art = Path(out_dir) / "artifact"
        art.mkdir(parents=True, exist_ok=True)
        torch.save({k: v.detach().cpu() for k, v in self.model.state_dict().items()}, art / "model.pt")
        counts = self.counts.cpu().numpy()
        np.save(art / "presentation_counts.npy", counts.astype(np.int32))
        wall = time.time() - self.t0
        return dict(loop_s=loop_s, wall_s=wall, log=log, counts=counts)


def histogram_report(counts, bs, steps):
    c = np.sort(counts.astype(np.int64))
    n = len(c)
    total = int(c.sum())
    lorenz = np.cumsum(c) / max(total, 1)
    gini = float(1 - 2 * lorenz.sum() / n + 1 / n)
    hist = np.bincount(np.minimum(c, 200))
    return dict(total_presentations=total, expected_presentations=bs * steps,
                n_unique_seen=int((c > 0).sum()), n_never_seen=int((c == 0).sum()),
                min=int(c[0]), max=int(c[-1]), mean=float(c.mean()), std=float(c.std()),
                median=float(np.median(c)), gini=gini,
                count_histogram={int(k): int(v) for k, v in enumerate(hist) if v})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solution", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", default="full", choices=list(common.MODES))
    ap.add_argument("--t0", type=float, default=None)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = common.MODES[args.mode]
    rep = dict(mode=args.mode, seed=common.seed_value(), batch_size=cfg["batch_size"],
               total_steps=cfg["total_steps"], budget_s=cfg["budget_s"], ok=False, error=None)
    h = None
    try:
        h = Harness(args)
        res = h.run(args.solution, out)
        hist = histogram_report(res["counts"], h.bs, h.total_steps)
        rep.update(ok=True, wall_s=res["wall_s"], loop_s=res["loop_s"], log=res["log"], histogram=hist,
                   presentations_ok=hist["total_presentations"] == hist["expected_presentations"],
                   within_budget=res["wall_s"] <= h.budget_s)
    except (ScheduleError, BudgetExceeded) as e:
        rep.update(error=f"{type(e).__name__}: {e}")
    except Exception as e:  # solution crash or harness failure
        rep.update(error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc()[-4000:])
    if h is not None:
        rep["timing"] = {k: round(v, 3) for k, v in h.t.items()}
        rep["probe_examples"] = h.probe_examples
        rep["probe_calls"] = h.probe_calls
    rep.setdefault("wall_s", time.time() - (args.t0 or time.time()))
    (out / "train_report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: v for k, v in rep.items() if k not in ("log", "traceback")}, indent=1))
    if rep.get("traceback"):
        print(rep["traceback"], file=sys.stderr)
    sys.exit(0 if rep["ok"] else 1)


if __name__ == "__main__":
    main()
