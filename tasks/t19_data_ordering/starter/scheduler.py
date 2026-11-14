"""Starter: uniform random reshuffle every epoch (each example once per pass)."""
import torch


class Scheduler:
    def __init__(self, ctx):
        self.n, self.bs, self.dev = ctx.n_train, ctx.batch_size, ctx.device
        self.gen = torch.Generator(device=self.dev).manual_seed(ctx.seed + 100)
        self.buf = torch.empty(0, dtype=torch.int64, device=self.dev)

    def next_batch(self, step):
        while self.buf.numel() < self.bs:  # epochs are concatenated; the last batch of an epoch spills over
            self.buf = torch.cat([self.buf, torch.randperm(self.n, generator=self.gen, device=self.dev)])
        idx, self.buf = self.buf[:self.bs], self.buf[self.bs:]
        return idx

    def observe(self, step, indices, losses, logits):
        pass  # starter ignores feedback
