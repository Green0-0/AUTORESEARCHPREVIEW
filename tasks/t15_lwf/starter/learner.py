"""Starter for t15_lwf: plain sequential fine-tuning (the catastrophic-forgetting baseline).

ResNet-9-style CNN, 100-way head, cross-entropy over all 100 logits on each task's data only.
GPU-resident data, GPU augmentation (random crop + flip), AMP bf16, channels_last. No aux state.
"""
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

MEAN = torch.tensor([0.507, 0.487, 0.441]).view(1, 3, 1, 1) * 255
STD = torch.tensor([0.267, 0.256, 0.276]).view(1, 3, 1, 1) * 255


def _cbr(ci, co):
    return nn.Sequential(nn.Conv2d(ci, co, 3, padding=1, bias=False), nn.BatchNorm2d(co), nn.ReLU(inplace=True))


class Res(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.a, self.b = _cbr(c, c), _cbr(c, c)

    def forward(self, x):
        return x + self.b(self.a(x))


class Net(nn.Module):
    def __init__(self, num_classes=100, w=64):
        super().__init__()
        self.body = nn.Sequential(
            _cbr(3, w), _cbr(w, 2 * w), nn.MaxPool2d(2), Res(2 * w),
            _cbr(2 * w, 4 * w), nn.MaxPool2d(2),
            _cbr(4 * w, 8 * w), nn.MaxPool2d(2), Res(8 * w),
            nn.AdaptiveMaxPool2d(1), nn.Flatten())
        self.head = nn.Linear(8 * w, num_classes)

    def forward(self, x):
        return self.head(self.body(x))


def build_model():
    return Net(100)


def _normalize(x_uint8):
    x = x_uint8.float()
    return ((x - MEAN.to(x.device)) / STD.to(x.device)).contiguous(memory_format=torch.channels_last)


def _augment(x):  # x float NCHW on device; random crop (pad 4) + horizontal flip, on GPU
    n = x.shape[0]
    xp = F.pad(x, (4, 4, 4, 4), mode="reflect")
    i = torch.randint(0, 9, (n,), device=x.device)
    j = torch.randint(0, 9, (n,), device=x.device)
    ar = torch.arange(32, device=x.device)
    rows = (i[:, None] + ar[None]).view(n, 1, 32, 1)
    cols = (j[:, None] + ar[None]).view(n, 1, 1, 32)
    out = xp[torch.arange(n, device=x.device).view(n, 1, 1, 1),
             torch.arange(3, device=x.device).view(1, 3, 1, 1), rows, cols]
    flip = torch.rand(n, device=x.device) < 0.5
    out = torch.where(flip[:, None, None, None], out.flip(3), out)
    return out.contiguous(memory_format=torch.channels_last)


class Learner:
    EPOCHS = {"full": 30, "smoke": 4, "cpu-smoke": 1}

    def __init__(self, model, info):
        self.model = model.to(memory_format=torch.channels_last)
        self.info = info
        self.dev = torch.device(info["device"])
        self.epochs = self.EPOCHS[info["mode"]]
        self.bs = 256 if self.dev.type == "cuda" else 64

    def learn_task(self, task_id, x, y, info):
        t_start, left = time.time(), info["time_left_s"]
        tasks_left = self.info["num_tasks"] - task_id
        slice_s = 0.95 * left / tasks_left  # never overrun the total budget
        model, n = self.model, x.shape[0]
        model.train()
        steps = self.epochs * max(1, n // self.bs)
        opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9, nesterov=True, weight_decay=5e-4)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=0.05, total_steps=steps, pct_start=0.25)
        xn = _normalize(x)
        amp = self.dev.type == "cuda"
        step = 0
        for ep in range(self.epochs):
            perm = torch.randperm(n, device=self.dev)
            for k in range(0, n - self.bs + 1 if n >= self.bs else 1, self.bs):
                idx = perm[k:k + self.bs]
                with torch.autocast(self.dev.type, dtype=torch.bfloat16, enabled=amp):
                    loss = F.cross_entropy(model(_augment(xn[idx])), y[idx])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                if step < steps - 1:
                    sched.step()
                step += 1
            if time.time() - t_start > slice_s:
                break
        # optimizer is local: nothing persists except the model

    def aux_state(self):
        return {}


def predict(model, aux, x):
    with torch.autocast(x.device.type, dtype=torch.bfloat16, enabled=x.device.type == "cuda"):
        return model(_normalize(x)).float()
