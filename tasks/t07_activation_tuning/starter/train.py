"""Starter training procedure: GPU-resident data, GPU flip+crop augmentation, AdamW + one-cycle LR,
label smoothing, bf16 autocast, large batches. Fixed epoch count, with a deadline guard."""
import os
import time

import torch
import torch.nn.functional as F

EPOCHS = {"full": 1500, "smoke": 3, "cpu-smoke": 1}
BATCH = 2048
LR = 3e-3
WD = 0.05


def augment(x_u8):
    """x_u8: (B,3,32,32) uint8 on device -> float in [0,1], random flip + random 4px translate."""
    B = x_u8.shape[0]
    dev = x_u8.device
    x = x_u8.float().div_(255.0)
    flip = torch.rand(B, device=dev) < 0.5
    x = torch.where(flip[:, None, None, None], x.flip(3), x)
    xp = F.pad(x, (4, 4, 4, 4), mode="reflect")
    i0 = torch.randint(0, 9, (B,), device=dev)
    j0 = torch.randint(0, 9, (B,), device=dev)
    ar = torch.arange(32, device=dev)
    rows = (i0[:, None] + ar)[:, None, :, None].expand(B, 3, 32, 40)
    cols = (j0[:, None] + ar)[:, None, None, :].expand(B, 3, 32, 32)
    xp = xp.gather(2, rows)
    return xp.gather(3, cols)


def train(model, x, y, ctx):
    dev = ctx["device"]
    mode = ctx["mode"]
    epochs = int(os.environ.get("T07_STARTER_EPOCHS", EPOCHS[mode]))
    n = x.shape[0]
    steps_per_epoch = n // BATCH if n >= BATCH else 1
    bs = min(BATCH, n)
    total = epochs * steps_per_epoch
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD, betas=(0.9, 0.99),
                            fused=(dev.type == "cuda"))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=total, pct_start=0.15)
    use_amp = dev.type == "cuda"
    t0 = time.time()
    for ep in range(epochs):
        perm = torch.randperm(n, device=dev)
        for s in range(steps_per_epoch):
            idx = perm[s * bs:(s + 1) * bs]
            xb = augment(x[idx])
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=use_amp):
                loss = F.cross_entropy(model(xb), y[idx], label_smoothing=0.1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
        if ep % 100 == 0 or ep == epochs - 1:
            print(f"epoch {ep} loss {loss.item():.3f} t {time.time() - t0:.1f}s", flush=True)
        if time.time() > ctx["deadline"] - 20:   # deadline guard (leave margin)
            print("deadline guard hit, stopping", flush=True)
            break
