"""Starter for t09_tokenizer_transfer: naive tokenizer transfer.

New 32k embedding matrix drawn at random (N(0, 0.02)), LM head tied to it (as in GPT-2), then the whole model is
trained on random 1024-token windows of the training stream with AdamW until the time budget runs out.
LR schedule is time-based (linear warmup, cosine decay to 10%), so it adapts to whatever throughput the GPU gives.
"""
import math
import time

import torch
import torch.nn.functional as F

SEQ, BATCH = 1024, 16
LR, WD, BETAS = 6e-4, 0.1, (0.9, 0.95)
WARMUP_FRAC, FINAL_FRAC = 0.03, 0.1
MARGIN_S = 15.0


def adapt(model, ctx):
    dev, seed = ctx["device"], ctx["seed"]
    cpu = ctx["mode"] == "cpu-smoke"
    seq, batch = (64, 2) if cpu else (SEQ, BATCH)
    g = torch.Generator(device=dev).manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True

    with torch.no_grad():                         # random init of the new embeddings, tied head
        model.wte.weight.normal_(0.0, 0.02, generator=torch.Generator(device=dev).manual_seed(seed + 1))
    model.tie_weights()

    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": WD}, {"params": no_decay, "weight_decay": 0.0}],
                            lr=LR, betas=BETAS, fused=dev.type == "cuda")
    ids = ctx["train_ids"]
    n = ids.numel() - seq - 1
    t0, end = time.time(), ctx["deadline"] - (5.0 if cpu else MARGIN_S)
    total = end - t0
    step = 0
    model.train()
    while True:
        now = time.time()
        if now >= end:
            break
        frac = (now - t0) / total
        lr = LR * (frac / WARMUP_FRAC if frac < WARMUP_FRAC else
                   FINAL_FRAC + (1 - FINAL_FRAC) * 0.5 * (1 + math.cos(math.pi * (frac - WARMUP_FRAC) / (1 - WARMUP_FRAC))))
        for gr in opt.param_groups:
            gr["lr"] = lr
        starts = torch.randint(0, n, (batch,), device=dev, generator=g)
        idx = starts[:, None] + torch.arange(seq + 1, device=dev)[None]
        w = ids[idx].long()
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=not cpu):
            logits = model(w[:, :-1])
        loss = F.cross_entropy(logits.float().view(-1, logits.shape[-1]), w[:, 1:].reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
        if step % 200 == 0 or (cpu and step % 2 == 0):
            print(f"step {step} t={time.time() - t0:.0f}s lr={lr:.2e} loss={loss.item():.3f}", flush=True)
    print(f"done: {step} steps, {step * batch * seq / 1e6:.1f}M tokens in {time.time() - t0:.0f}s", flush=True)
