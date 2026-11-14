"""Starter: plain cross-entropy ResNet-9 training. No robustness method at all.
GPU-resident data, GPU flip/crop augmentation, bf16 autocast, large batches."""
import argparse, json, os, shutil, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

from model import build_model

p = argparse.ArgumentParser()
p.add_argument("--train", required=True)
p.add_argument("--out", required=True)
p.add_argument("--mode", default="full", choices=["full", "smoke", "cpu-smoke"])
p.add_argument("--budget-s", type=float, default=1200)
p.add_argument("--device", default="cuda")
p.add_argument("--seed", type=int, default=int(os.environ.get("AR_SEED", 0)))
a = p.parse_args()

t0 = time.time()
torch.manual_seed(a.seed); np.random.seed(a.seed)
dev = torch.device(a.device)
torch.backends.cudnn.benchmark = True

d = np.load(a.train)
X = torch.from_numpy(d["x"]).to(dev).permute(0, 3, 1, 2).contiguous()  # uint8 NCHW on device
Y = torch.from_numpy(d["y"]).to(dev)
N = len(Y)

EPOCHS = {"full": 30, "smoke": 2, "cpu-smoke": 1}[a.mode]
BS = 512 if a.mode != "cpu-smoke" else 128
steps_per_epoch = N // BS
total = max(EPOCHS * steps_per_epoch, 10)

model = build_model().to(dev).to(memory_format=torch.channels_last)
opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, nesterov=True, weight_decay=5e-4)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=0.2, total_steps=total, pct_start=0.25,
                                            anneal_strategy="linear", div_factor=10, final_div_factor=100)
use_amp = dev.type == "cuda"


def augment(xb):
    # xb uint8 (B,3,32,32) -> float [0,1], random crop (pad 4, reflect) + horizontal flip, all on device
    B = xb.shape[0]
    x = xb.float().div_(255)
    x = F.pad(x, (4, 4, 4, 4), mode="reflect")
    dx = torch.randint(0, 9, (B,), device=x.device); dy = torch.randint(0, 9, (B,), device=x.device)
    ar = torch.arange(32, device=x.device)
    rows = (dy[:, None] + ar)[:, None, :, None]
    cols = (dx[:, None] + ar)[:, None, None, :]
    x = x[torch.arange(B, device=x.device)[:, None, None, None], torch.arange(3, device=x.device)[None, :, None, None], rows, cols]
    flip = torch.rand(B, device=x.device) < 0.5
    x = torch.where(flip[:, None, None, None], x.flip(3), x)
    return x.contiguous(memory_format=torch.channels_last)


log = []
model.train()
for ep in range(EPOCHS):
    perm = torch.randperm(N, device=dev)
    tl = torch.zeros((), device=dev)
    for i in range(steps_per_epoch):
        idx = perm[i * BS:(i + 1) * BS]
        xb, yb = augment(X[idx]), Y[idx]
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=use_amp):
            loss = F.cross_entropy(model(xb), yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step(); sched.step()
        tl += loss.detach()
    log.append({"epoch": ep, "loss": float(tl) / steps_per_epoch, "t": time.time() - t0})
    print(json.dumps(log[-1]), flush=True)

out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
torch.save({k: v.detach().float().cpu() if v.is_floating_point() else v.cpu() for k, v in model.state_dict().items()},
           out / "weights.pt")
shutil.copy(Path(__file__).with_name("model.py"), out / "model.py")
(out / "train_log.json").write_text(json.dumps(log))
print(f"done in {time.time() - t0:.1f}s")
