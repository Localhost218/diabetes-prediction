"""Chương 3 — Basic CNN và Improved CNN bằng PyTorch (GPU), tăng cường dữ liệu trên GPU, vòng huấn luyện, Grad-CAM."""
import copy
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicCNN(nn.Module):
    """Cùng kiến trúc với bản NumPy: Conv32 → Pool → Conv64 → Pool → FC128 → FC(K)."""

    def __init__(self, in_ch, n_classes, size):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(64 * (size // 4) ** 2, 128), nn.ReLU(), nn.Linear(128, n_classes))

    def forward(self, x):
        return self.classifier(self.features(x))


class ResBlock(nn.Module):
    """Khối dư (residual): Conv-BN-ReLU-Conv-BN cộng lối tắt (1×1 khi đổi số kênh), rồi ReLU."""

    def __init__(self, cin, cout):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.short = (nn.Identity() if cin == cout else
                      nn.Sequential(nn.Conv2d(cin, cout, 1, bias=False), nn.BatchNorm2d(cout)))

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return F.relu(y + self.short(x))


class ImprovedCNN(nn.Module):
    """Stem Conv-BN-ReLU → [ResBlock → MaxPool → Dropout tăng dần] × len(widths) → GAP → FC256 → Dropout → FC(K)."""

    def __init__(self, in_ch, n_classes, widths=(32, 64, 128)):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(in_ch, widths[0], 3, padding=1, bias=False),
                                  nn.BatchNorm2d(widths[0]), nn.ReLU())
        blocks, cin = [], widths[0]
        for i, w in enumerate(widths):
            blocks += [ResBlock(cin, w), nn.MaxPool2d(2), nn.Dropout(0.1 + 0.1 * i)]
            cin = w
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(cin, 256), nn.ReLU(),
                                  nn.Dropout(0.4), nn.Linear(256, n_classes))

    def forward(self, x):
        return self.head(self.blocks(self.stem(x)))


def gpu_augment(x, cfg):
    """Tăng cường dữ liệu theo lô ngay trên GPU: lật ngang, xoay, tịnh tiến, phóng to/thu nhỏ (một phép affine),
    và dao động độ sáng/độ tương phản. x: [N, C, H, W] trong [0, 1]."""
    n, dev = x.shape[0], x.device
    if cfg["flip"]:
        m = torch.rand(n, device=dev) < 0.5
        x = torch.where(m[:, None, None, None], x.flip(3), x)
    ang = (torch.rand(n, device=dev) * 2 - 1) * math.radians(cfg["rot"])
    sc = 1 + (torch.rand(n, device=dev) * 2 - 1) * cfg["zoom"]
    tx, ty = [(torch.rand(n, device=dev) * 2 - 1) * 2 * cfg["shift"] for _ in range(2)]
    theta = torch.stack([torch.stack([torch.cos(ang) / sc, -torch.sin(ang) / sc, tx], 1),
                         torch.stack([torch.sin(ang) / sc, torch.cos(ang) / sc, ty], 1)], 1)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    x = F.grid_sample(x, grid, padding_mode="reflection" if cfg["color"] else "zeros", align_corners=False)
    if cfg["color"]:
        b = 1 + (torch.rand(n, 1, 1, 1, device=dev) * 2 - 1) * cfg["color"]
        c = 1 + (torch.rand(n, 1, 1, 1, device=dev) * 2 - 1) * cfg["color"]
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        x = ((x - mean) * c + mean) * b
    return x.clamp(0, 1)


def to_tensor(x):
    """NHWC float32 numpy -> NCHW tensor."""
    return torch.from_numpy(np.ascontiguousarray(x.transpose(0, 3, 1, 2)))


@torch.no_grad()
def predict_proba(model, x, mean=None, std=None, batch=None, dev="cuda"):
    model.eval()
    out = []
    batch = batch or (512 if x.shape[1] <= 32 else 64)      # ảnh 128 px: lô 512 cần ~1 GB mỗi tensor kích hoạt
    for i in range(0, len(x), batch):
        xb = to_tensor(x[i:i + batch]).to(dev)
        if mean is not None:
            xb = (xb - mean) / std
        out.append(F.softmax(model(xb), 1).cpu())
    return torch.cat(out).numpy()


def train_torch(model, xtr, ytr, xva, yva, cfg, epochs, improved, dev="cuda", log=print):
    """Huấn luyện một mô hình; Basic: Adam 1e-3 hằng. Improved: AdamW + OneCycle, label smoothing 0,1,
    tăng cường dữ liệu và chuẩn hoá theo kênh. Giữ trọng số có val accuracy tốt nhất."""
    model.to(dev)
    xtr_t, ytr_t = to_tensor(xtr).to(dev), torch.from_numpy(ytr).to(dev)
    mean = std = None
    if improved:
        mean = xtr_t.mean(dim=(0, 2, 3), keepdim=True)
        std = xtr_t.std(dim=(0, 2, 3), keepdim=True) + 1e-6
    steps = math.ceil(len(xtr) / cfg["batch"])
    if improved:
        opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=5e-4)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=epochs * steps, pct_start=0.25)
        crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    else:
        opt, sched, crit = torch.optim.Adam(model.parameters(), lr=1e-3), None, nn.CrossEntropyLoss()
    hist = {k: [] for k in ("loss", "acc", "val_loss", "val_acc", "lr", "time")}
    best, best_acc, best_ep = None, -1.0, 0
    for ep in range(epochs):
        model.train()
        t0, perm = time.perf_counter(), torch.randperm(len(xtr_t), device=dev)
        tl = tc = 0.0
        for s in range(0, len(perm), cfg["batch"]):
            b = perm[s:s + cfg["batch"]]
            xb, yb = xtr_t[b], ytr_t[b]
            if improved:
                xb = (gpu_augment(xb, cfg) - mean) / std
            out = model(xb)
            loss = crit(out, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if sched:
                sched.step()
            tl += loss.item() * len(b)
            tc += (out.argmax(1) == yb).sum().item()
        if dev == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        pv = predict_proba(model, xva, mean, std, dev=dev)
        vl = float(-np.log(pv[np.arange(len(yva)), yva] + 1e-12).mean())
        va = float((pv.argmax(1) == yva).mean())
        for k, v in zip(hist, (tl / len(perm), tc / len(perm), vl, va, opt.param_groups[0]["lr"], dt)):
            hist[k].append(v)
        if va > best_acc:
            best_acc, best_ep, best = va, ep + 1, copy.deepcopy(model.state_dict())
        log(f"epoch {ep + 1}/{epochs} loss {hist['loss'][-1]:.4f} acc {hist['acc'][-1]:.4f} "
            f"val_loss {vl:.4f} val_acc {va:.4f} {dt:.1f}s")
    model.load_state_dict(best)
    norm = None if mean is None else (mean.cpu(), std.cpu())
    return hist, best_ep, norm


class GradCAM:
    """Grad-CAM (Selvaraju và cộng sự, 2017): trọng số kênh = trung bình gradient của điểm lớp theo bản đồ đặc trưng
    tầng tích chập cuối; bản đồ nhiệt = ReLU(Σ_k α_k · A_k), nội suy về kích thước ảnh."""

    def __init__(self, model, layer):
        self.model, self.acts, self.grads = model.eval(), None, None
        layer.register_forward_hook(lambda m, i, o: setattr(self, "acts", o))
        layer.register_full_backward_hook(lambda m, gi, go: setattr(self, "grads", go[0]))

    def __call__(self, x, cls=None):
        x = x.requires_grad_(True)
        out = self.model(x)
        cls = out.argmax(1) if cls is None else cls
        self.model.zero_grad()
        out.gather(1, cls.view(-1, 1)).sum().backward()
        alpha = self.grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((alpha * self.acts).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)
        cam = cam / (cam.amax(dim=(2, 3), keepdim=True) + 1e-8)
        return cam[:, 0].detach().cpu().numpy(), cls.cpu().numpy()


def n_params(model):
    return int(sum(p.numel() for p in model.parameters()))
