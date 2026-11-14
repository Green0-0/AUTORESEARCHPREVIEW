"""Starter training: joint training of all exits (mean CE), GPU-resident data and augmentation,
bf16 autocast, SGD+nesterov with a time-based one-cycle LR schedule that fills the budget."""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from model import AnytimeNet

T_START = time.time()
ap = argparse.ArgumentParser()
ap.add_argument("--train", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--mode", default="full")
ap.add_argument("--budget-s", type=float, required=True)
a = ap.parse_args()

seed = int(os.environ.get("AR_SEED", "0"))
torch.manual_seed(seed)
np.random.seed(seed)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
# leave room for startup, saving and process exit inside the hard budget
train_s = a.budget_s - 75 if a.mode == "full" else a.budget_s * 0.4
bs = 512 if dev.type == "cuda" else 128
widths = (32, 64, 128, 256)

with np.load(a.train) as z:
    x = torch.from_numpy(z["x"]).to(dev).permute(0, 3, 1, 2).float().div_(255)
    y = torch.from_numpy(z["y"]).to(dev)
model = AnytimeNet(widths).to(dev).to(memory_format=torch.channels_last)
x = ((x - model.mean) / model.std).to(memory_format=torch.channels_last)
n = len(y)
decay, no_decay = [], []
for p in model.parameters():
    (decay if p.dim() > 1 else no_decay).append(p)
peak_lr = 0.4 if dev.type == "cuda" else 0.1
opt = torch.optim.SGD([{"params": decay, "weight_decay": 5e-4}, {"params": no_decay, "weight_decay": 0}],
                      lr=0.0, momentum=0.9, nesterov=True)
ar32 = torch.arange(32, device=dev)


def augment(xb):
    """random crop (reflect pad 4) + horizontal flip, on device."""
    B = xb.shape[0]
    xp = F.pad(xb, (4, 4, 4, 4), mode="reflect")
    oy = torch.randint(0, 9, (B,), device=dev)
    ox = torch.randint(0, 9, (B,), device=dev)
    flip = torch.rand(B, device=dev) < 0.5
    cols = ox[:, None] + ar32
    cols = torch.where(flip[:, None], cols.flip(1), cols)
    i = (oy[:, None] + ar32)[:, None, :, None]
    j = cols[:, None, None, :]
    b = torch.arange(B, device=dev)[:, None, None, None]
    c = torch.arange(3, device=dev)[None, :, None, None]
    return xp[b, c, i, j].contiguous(memory_format=torch.channels_last)


def lr_at(frac):
    w = 0.15
    return peak_lr * (frac / w if frac < w else max(0.0, (1 - frac) / (1 - w)))


t0 = time.time()
step, epoch = 0, 0
done = False
model.train()
while not done:
    perm = torch.randperm(n, device=dev)
    for k in range(0, n - bs + 1, bs):
        frac = (time.time() - t0) / train_s
        if frac >= 1:
            done = True
            break
        for g in opt.param_groups:
            g["lr"] = lr_at(frac)
        idx = perm[k:k + bs]
        xb, yb = augment(x[idx]), y[idx]
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            outs = model.forward_all(xb)
            loss = sum(F.cross_entropy(o.float(), yb, label_smoothing=0.1) for o in outs) / len(outs)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        step += 1
    epoch += 1
    print(f"epoch {epoch} step {step} t={time.time() - t0:.0f}s loss {loss.item():.3f}", flush=True)

out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
torch.save(model.state_dict(), out / "model.pt")
(out / "config.json").write_text(json.dumps({"widths": list(widths)}))
print(f"done: {step} steps, {epoch} epochs, train loop {time.time() - t0:.0f}s, total {time.time() - T_START:.0f}s")
