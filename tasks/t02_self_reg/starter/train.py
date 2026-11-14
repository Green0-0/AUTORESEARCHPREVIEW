"""Starter training: plain cross-entropy on raw images, SGD (nesterov) with weight decay, linear
warmup + linear decay, bf16 autocast, data already on the GPU. No augmentation, no dropout, no mixing:
every training forward is model(x_raw[idx])."""
import time

import torch
import torch.nn.functional as F

HP = {
    "full":      dict(steps=3000, bs=250, lr=0.1, wd=5e-4, warmup=0.1),
    "smoke":     dict(steps=200,  bs=250, lr=0.1, wd=5e-4, warmup=0.1),
    "cpu-smoke": dict(steps=6,    bs=50,  lr=0.1, wd=5e-4, warmup=0.1),
}


def train(model, x, y, ctx):
    """x: float32 (N,3,32,32) in [0,1] on ctx['device']; y: int64 (N,). Trains `model` in place."""
    hp = HP[ctx["mode"]]
    dev = ctx["device"]
    use_amp = dev.type == "cuda"
    if use_amp:
        torch.backends.cudnn.benchmark = True
    model.to(memory_format=torch.channels_last)
    decay = [p for n, p in model.named_parameters() if p.ndim > 1]
    no_decay = [p for n, p in model.named_parameters() if p.ndim <= 1]
    opt = torch.optim.SGD([{"params": decay, "weight_decay": hp["wd"]}, {"params": no_decay, "weight_decay": 0.0}],
                          lr=hp["lr"], momentum=0.9, nesterov=True)
    total, warm = hp["steps"], max(1, int(hp["warmup"] * hp["steps"]))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else max(0.0, (total - s) / (total - warm)))
    N, B = len(y), min(hp["bs"], len(y))
    g = torch.Generator(device=dev); g.manual_seed(ctx["seed"])
    model.train()
    perm, pos, step = torch.randperm(N, device=dev, generator=g), 0, 0
    tl, nl = torch.zeros((), device=dev), 0
    t0 = time.time()
    while step < total:
        if pos + B > N:
            perm, pos = torch.randperm(N, device=dev, generator=g), 0
        idx = perm[pos:pos + B]; pos += B
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=use_amp):
            loss = F.cross_entropy(model(x[idx]), y[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step(); sched.step(); step += 1
        tl += loss.detach(); nl += 1
        if step % 500 == 0 or step == total:
            print(f"step {step}/{total} loss {float(tl) / nl:.4f} t {time.time() - t0:.1f}s", flush=True)
            tl.zero_(); nl = 0
        if time.time() > ctx["deadline"] - 15:      # safety margin; normally never hit
            print("deadline reached, stopping early", flush=True)
            break
