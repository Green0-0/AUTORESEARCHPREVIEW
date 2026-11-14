"""Starter for t10_label_noise: plain cross-entropy training of a small fast ResNet-style CNN
(airbench-like) on the noisy labels. GPU-resident data, GPU flip/crop, bf16 autocast, batch 1024.
No noise handling at all -- this is the baseline to beat."""
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

MEAN = torch.tensor([125.3, 123.0, 113.9]).view(1, 3, 1, 1)
STD = torch.tensor([63.0, 62.1, 66.7]).view(1, 3, 1, 1)


def conv_bn(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.GELU())


class Block(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1, bias=False)
        self.pool = nn.MaxPool2d(2)
        self.bn1 = nn.BatchNorm2d(cout)
        self.c2 = conv_bn(cout, cout)

    def forward(self, x):
        x = F.gelu(self.bn1(self.pool(self.c1(x))))
        return x + self.c2(x)


class Net(nn.Module):
    def __init__(self, widths=(64, 128, 256, 512), n_classes=10):
        super().__init__()
        w0, w1, w2, w3 = widths
        self.stem = conv_bn(3, w0)
        self.blocks = nn.Sequential(Block(w0, w1), Block(w1, w2), Block(w2, w3))
        self.head = nn.Linear(w3, n_classes, bias=False)

    def forward(self, x):
        x = self.blocks(self.stem(x))
        return 0.125 * self.head(F.adaptive_max_pool2d(x, 1).flatten(1))


def normalize(images_u8):
    x = images_u8.float()
    return ((x - MEAN.to(x.device)) / STD.to(x.device)).contiguous(memory_format=torch.channels_last)


def augment(xp, idx, pad=4):
    """random crop from reflect-padded images + random horizontal flip, all on the GPU."""
    B = idx.numel(); dev = xp.device
    dy = torch.randint(0, 2 * pad + 1, (B,), device=dev); dx = torch.randint(0, 2 * pad + 1, (B,), device=dev)
    ar = torch.arange(32, device=dev)
    rows = (dy[:, None] + ar)[:, None, :, None]
    cols = (dx[:, None] + ar)
    flip = torch.rand(B, device=dev) < 0.5
    cols = torch.where(flip[:, None], cols.flip(1), cols)[:, None, None, :]
    ch = torch.arange(3, device=dev)[None, :, None, None]
    return xp[idx[:, None, None, None], ch, rows, cols].contiguous(memory_format=torch.channels_last)


def train(images, labels, artifact_dir, info):
    dev = info["device"]; mode = info["mode"]
    cuda = dev.type == "cuda"
    epochs = {"full": 40, "smoke": 3, "cpu-smoke": 2}[mode]
    bs = 1024 if cuda else 256
    widths = (64, 128, 256, 512) if cuda else (16, 32, 64, 128)
    torch.backends.cudnn.benchmark = True
    model = Net(widths, info["n_classes"]).to(dev).to(memory_format=torch.channels_last)
    x = normalize(images)
    xp = F.pad(x, (4, 4, 4, 4), mode="reflect")
    N = x.shape[0]; steps_per_epoch = N // bs; total = epochs * steps_per_epoch
    decay, no_decay = [], []
    for n_, p in model.named_parameters():
        (no_decay if p.ndim <= 1 else decay).append(p)
    opt = torch.optim.SGD([{"params": decay, "weight_decay": 5e-4}, {"params": no_decay, "weight_decay": 0}],
                          lr=0.4, momentum=0.9, nesterov=True)
    warm = max(1, int(0.15 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else max(0.0, (total - s) / (total - warm)))
    step = 0; t_epoch = 0.0
    for ep in range(epochs):
        if time.time() + 1.5 * t_epoch + 20 > info["deadline"]:   # never overrun the budget
            break
        t0 = time.time(); model.train()
        perm = torch.randperm(N, device=dev)
        for i in range(steps_per_epoch):
            idx = perm[i * bs:(i + 1) * bs]
            xb = augment(xp, idx); yb = labels[idx]
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=cuda):
                loss = F.cross_entropy(model(xb), yb, label_smoothing=0.1)
            opt.zero_grad(set_to_none=True)
            loss.backward(); opt.step(); sched.step(); step += 1
        if cuda:
            torch.cuda.synchronize()
        t_epoch = time.time() - t0
        print(f"epoch {ep} loss {loss.item():.3f} {t_epoch:.1f}s", flush=True)
    torch.save({"widths": widths, "n_classes": info["n_classes"], "state": model.state_dict()}, f"{artifact_dir}/model.pt")


def load_predictor(artifact_dir, device):
    ck = torch.load(f"{artifact_dir}/model.pt", map_location=device)
    model = Net(ck["widths"], ck["n_classes"]).to(device).to(memory_format=torch.channels_last)
    model.load_state_dict(ck["state"]); model.eval()

    def predict(images_u8):
        x = normalize(images_u8)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            return (model(x).float().softmax(1) + model(x.flip(3)).float().softmax(1)) / 2
    return predict
