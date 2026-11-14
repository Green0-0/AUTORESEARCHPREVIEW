"""Starter training: frozen ImageNet backbone, only the 1x1 head is trained with pixel-wise cross-entropy on the
100 few-shot images (logits upsampled to 320x320, label 255 ignored). No augmentation. AdamW, constant LR,
bf16 autocast, data already on the GPU. Trains until shortly before the deadline."""
import time

import torch
import torch.nn.functional as F


def train(model, ctx):
    dev, mode = ctx["device"], ctx["mode"]
    g = torch.Generator(device="cpu").manual_seed(ctx["seed"])
    X = ctx["train_x"].permute(0, 3, 1, 2).contiguous()     # uint8 (N,3,320,320) on device
    Y = ctx["train_y"].long()                                # (N,320,320), 255 = ignore
    N = X.shape[0]
    bs = 16 if dev.type == "cuda" else 2
    margin = 15.0 if mode == "full" else 5.0
    max_steps = 10 ** 9 if mode != "cpu-smoke" else 3

    for p in model.backbone.parameters():
        p.requires_grad_(False)
    model.backbone.eval()                                     # frozen BN statistics
    model.to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.head.parameters(), lr=1e-3, weight_decay=1e-4)
    use_amp = dev.type == "cuda"
    step, t_log = 0, time.time()
    while time.time() < ctx["deadline"] - margin and step < max_steps:
        idx = torch.randint(0, N, (bs,), generator=g).to(dev)
        xb = X[idx].float().div_(255).contiguous(memory_format=torch.channels_last)
        yb = Y[idx]
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=use_amp):
            with torch.no_grad():
                f = model.features(xb)
            logits = model.head(f)
        logits = F.interpolate(logits.float(), size=yb.shape[-2:], mode="bilinear", align_corners=False)
        loss = F.cross_entropy(logits, yb, ignore_index=ctx["ignore_index"])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        step += 1
        if step % 200 == 0 or time.time() - t_log > 60:
            print(f"step {step} loss {loss.item():.4f} t {time.time() - ctx['deadline'] + ctx['time_budget_s']:.0f}s",
                  flush=True)
            t_log = time.time()
    print(f"done: {step} steps", flush=True)
