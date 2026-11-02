"""
Chương 3 — CNN phân độ bệnh võng mạc đái tháo đường (DR) trên RetinaMNIST (MedMNIST v2, 128×128).

RetinaMNIST lấy từ DeepDRiD (ISBI 2020): 1.600 ảnh đáy mắt, nhãn 0–4 theo thang ICDR.
Bốn mô hình, cùng dữ liệu tăng cường và giao thức (chọn epoch tốt nhất theo QWK trên validation):
  1. MLP (đối chứng — không có tích chập)       3. ResNet-18 học chuyển giao từ ImageNet
  2. CNN tự thiết kế (4 khối Conv-BN-ReLU-Pool) 4. MobileNetV3-Large học chuyển giao (nhẹ, để deploy CPU)

    .venv/bin/python lab/ch3_cnn.py
"""

import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from torchvision.transforms import v2 as T

from common import ACCENT, DATA, GREY, MODELS, PALETTE, plt, r4, save_fig, save_results, seed_everything

torch.set_num_threads(6)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # huấn luyện trên GPU nếu có
CLASSES = ["0 – Không DR", "1 – Nhẹ", "2 – Trung bình", "3 – Nặng", "4 – Tăng sinh"]
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]  # chuẩn hóa ImageNet (bắt buộc cho mô hình tiền huấn luyện)


class RetinaDataset(Dataset):
    def __init__(self, images, labels, train):
        self.images, self.labels = images, labels.ravel().astype(np.int64)
        aug = [T.RandomHorizontalFlip(), T.RandomVerticalFlip(), T.RandomRotation(20),
               T.ColorJitter(0.2, 0.2, 0.1)] if train else []
        self.tf = T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True), *aug, T.Normalize(MEAN, STD)])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.tf(self.images[i]), self.labels[i]


# ------------------------------------------------------------------ mô hình
class MLPBaseline(nn.Module):
    """Ảnh 3×64×64 → vector 12.288 chiều → 512 → 128 → 5. Mất cấu trúc không gian."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(3 * 64 * 64, 512), nn.ReLU(), nn.Dropout(0.3),
                                 nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 5))

    def forward(self, x):
        return self.net(F.interpolate(x, size=64, mode="bilinear", align_corners=False))


def conv_block(c_in, c_out):
    """Conv 3×3 (đệm 1, giữ kích thước) → BatchNorm → ReLU → MaxPool 2×2 (giảm một nửa)."""
    return nn.Sequential(nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out),
                         nn.ReLU(inplace=True), nn.MaxPool2d(2))


class RetinaCNN(nn.Module):
    """3×128×128 → [32,64,128,256] kênh qua 4 khối (128→64→32→16→8) → GAP → Dropout → FC 5."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(conv_block(3, 32), conv_block(32, 64), conv_block(64, 128),
                                      conv_block(128, 256))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.4), nn.Linear(256, 5))

    def forward(self, x):
        return self.head(self.features(x))


def resnet18():
    m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    m.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(512, 5))  # thay lớp phân loại 1000 lớp ImageNet
    return m


def mobilenet():
    m = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2)
    m.classifier[3] = nn.Linear(m.classifier[3].in_features, 5)
    return m


SPECS = {
    "mlp": ("MLP (đối chứng)", MLPBaseline, 1e-3, 40),
    "cnn": ("CNN tự thiết kế", RetinaCNN, 1e-3, 40),
    "resnet18": ("ResNet-18 (chuyển giao)", resnet18, 3e-4, 20),
    "mobilenet": ("MobileNetV3-L (chuyển giao)", mobilenet, 5e-4, 20),
}


# ------------------------------------------------------------------ huấn luyện
@torch.no_grad()
def predict(model, loader):
    model.eval()
    probs, ys = [], []
    for x, y in loader:
        probs.append(F.softmax(model(x.to(DEVICE)), 1).cpu())
        ys.append(y)
    return torch.cat(probs).numpy(), torch.cat(ys).numpy()


def evaluate(probs, y):
    pred = probs.argmax(1)
    return {"accuracy": r4(accuracy_score(y, pred)), "macro_f1": r4(f1_score(y, pred, average="macro")),
            "qwk": r4(cohen_kappa_score(y, pred, weights="quadratic")),
            "auc_ovr": r4(roc_auc_score(y, probs, multi_class="ovr", average="macro"))}


def train(key, loaders, class_weights):
    name, factory, lr, epochs = SPECS[key]
    seed_everything(0)
    model = factory().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    # entropy chéo có trọng số lớp (bù lệch lớp) + làm mịn nhãn 0,05
    loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE), label_smoothing=0.05)
    history, best = [], (-1, None, 0)
    t0 = time.perf_counter()
    for ep in range(epochs):
        model.train()
        total, n = 0.0, 0
        for x, y in loaders["train"]:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            total, n = total + loss.item() * len(y), n + len(y)
        sched.step()
        pv, yv = predict(model, loaders["val"])
        ev = evaluate(pv, yv)
        history.append({"epoch": ep + 1, "train_loss": r4(total / n), **{f"val_{k}": v for k, v in ev.items()}})
        if ev["qwk"] > best[0]:
            best = (ev["qwk"], {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, ep + 1)
        print(f"  [{key}] epoch {ep + 1:2d} loss={total / n:.3f} val acc={ev['accuracy']:.3f} qwk={ev['qwk']:.3f}")
    train_s = time.perf_counter() - t0
    model.load_state_dict({k: v.to(DEVICE) for k, v in best[1].items()})
    pt, yt = predict(model, loaders["test"])
    model = model.cpu()  # đo độ trễ trên CPU — môi trường deploy không có GPU
    x1 = next(iter(loaders["test"]))[0][:1]
    model.eval()
    t1 = time.perf_counter()
    with torch.no_grad():
        for _ in range(20):
            model(x1)
    latency = (time.perf_counter() - t1) / 20 * 1000
    res = {"name": name, "params": n_params, "lr": lr, "epochs": epochs, "best_epoch": best[2],
           "train_s": round(train_s, 1), "cpu_latency_ms": round(latency, 1), "test": evaluate(pt, yt),
           "confusion": confusion_matrix(yt, pt.argmax(1)).tolist()}
    print(f"  ==> {name}: test {res['test']}")
    return model, res, history, pt, yt


# ------------------------------------------------------------------ giải thích
def grad_cam(model, layer, x, cls=None):
    """Grad-CAM (Selvaraju 2017): trọng số kênh = trung bình gradient của logit lớp theo bản đồ đặc trưng."""
    acts, grads = {}, {}

    def keep(module, inputs, output):  # hook trên tensor đầu ra — tránh xung đột với ReLU in-place của ResNet
        acts["a"] = output
        output.register_hook(lambda g: grads.setdefault("g", g))

    handle = layer.register_forward_hook(keep)
    model.eval()
    out = model(x)
    cls = int(out.argmax(1)) if cls is None else cls
    model.zero_grad()
    out[0, cls].backward()
    handle.remove()
    w = grads["g"].mean((2, 3), keepdim=True)
    cam = F.relu((w * acts["a"]).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    return (cam / (cam.max() + 1e-9)).detach().numpy(), cls


def plot_samples(data):
    fig, axes = plt.subplots(3, 5, figsize=(10, 6.2))
    y = data["train_labels"].ravel()
    for c in range(5):
        idx = np.where(y == c)[0][:3]
        for r, i in enumerate(idx):
            axes[r, c].imshow(data["train_images"][i])
            axes[r, c].axis("off")
        axes[0, c].set_title(CLASSES[c], fontsize=9)
    fig.suptitle("Ảnh mẫu RetinaMNIST 128×128 theo 5 mức ICDR", y=0.99)
    fig.tight_layout()
    save_fig(fig, "ch3_samples")


def plot_distribution(data):
    fig, ax = plt.subplots(figsize=(7, 2.6))
    x = np.arange(5)
    for j, split in enumerate(["train", "val", "test"]):
        cnt = np.bincount(data[f"{split}_labels"].ravel(), minlength=5)
        ax.bar(x + (j - 1) * 0.27, cnt, 0.27, label=f"{split} ({cnt.sum()})", color=PALETTE[j])
        for xi, v in zip(x, cnt):
            ax.text(xi + (j - 1) * 0.27, v + 5, str(v), ha="center", fontsize=7)
    ax.set_xticks(x, CLASSES, fontsize=8)
    ax.set_title("Phân bố nhãn theo tập — lệch lớp mạnh (lớp 4 chỉ 4–5%)")
    ax.legend()
    save_fig(fig, "ch3_distribution")


def plot_history(histories, results):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3))
    for i, (k, h) in enumerate(histories.items()):
        ep = [r["epoch"] for r in h]
        axes[0].plot(ep, [r["train_loss"] for r in h], color=PALETTE[i], label=results[k]["name"])
        axes[1].plot(ep, [r["val_accuracy"] for r in h], color=PALETTE[i])
        axes[2].plot(ep, [r["val_qwk"] for r in h], color=PALETTE[i])
    axes[0].set(title="Loss huấn luyện", xlabel="epoch")
    axes[1].set(title="Accuracy (validation)", xlabel="epoch")
    axes[2].set(title="Quadratic Weighted Kappa (validation)", xlabel="epoch")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    save_fig(fig, "ch3_history")


def plot_confusion(res):
    cm = np.array(res["confusion"])
    norm = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v}\n{norm[i, j]:.0%}", ha="center", va="center", fontsize=7.5,
                color="white" if norm[i, j] > 0.5 else "black")
    ax.set_xticks(range(5), range(5))
    ax.set_yticks(range(5), range(5))
    ax.set(xlabel="Mức dự đoán", ylabel="Mức thật", title=f"Ma trận nhầm lẫn — {res['name']}")
    ax.grid(False)
    save_fig(fig, "ch3_confusion")


def plot_feature_maps(cnn, x_img):
    """Bản đồ đặc trưng sau khối 1 và khối 3 của CNN tự thiết kế."""
    cnn.eval()
    with torch.no_grad():
        f1 = cnn.features[0](x_img)
        f3 = cnn.features[2](cnn.features[1](f1))
    fig, axes = plt.subplots(2, 7, figsize=(11, 3.4))
    img = x_img[0].permute(1, 2, 0).numpy() * STD + MEAN
    for r, (fm, title) in enumerate([(f1, "khối 1 (32×64×64)"), (f3, "khối 3 (128×16×16)")]):
        axes[r, 0].imshow(np.clip(img, 0, 1))
        axes[r, 0].set_title("ảnh vào" if r == 0 else "", fontsize=8)
        top = fm[0].mean((1, 2)).argsort(descending=True)[:6]
        for j, c in enumerate(top):
            axes[r, j + 1].imshow(fm[0, c], cmap="viridis")
            axes[r, j + 1].set_title(f"{title}\nkênh {int(c)}" if j == 0 else f"kênh {int(c)}", fontsize=7.5)
        for a in axes[r]:
            a.axis("off")
    fig.tight_layout()
    save_fig(fig, "ch3_feature_maps")


def plot_gradcam(model, ds, y):
    fig, axes = plt.subplots(2, 5, figsize=(10, 4.4))
    for c in range(5):
        i = int(np.where(y == c)[0][1])
        x, _ = ds[i]
        cam, pred = grad_cam(model, model.layer4, x[None])
        img = np.clip(x.permute(1, 2, 0).numpy() * STD + MEAN, 0, 1)
        axes[0, c].imshow(img)
        axes[0, c].set_title(f"thật: mức {c}", fontsize=9)
        axes[1, c].imshow(img)
        axes[1, c].imshow(cam, cmap="jet", alpha=0.45)
        axes[1, c].set_title(f"dự đoán: mức {pred}", fontsize=9)
        axes[0, c].axis("off")
        axes[1, c].axis("off")
    fig.suptitle("Grad-CAM trên layer4 của ResNet-18: vùng ảnh quyết định dự đoán", y=0.99)
    fig.tight_layout()
    save_fig(fig, "ch3_gradcam")


def main():
    seed_everything()
    data = np.load(DATA / "retina" / "retinamnist_128.npz", allow_pickle=False)
    ds = {s: RetinaDataset(data[f"{s}_images"], data[f"{s}_labels"], s == "train") for s in ["train", "val", "test"]}
    loaders = {s: DataLoader(d, batch_size=32, shuffle=(s == "train"), num_workers=4, persistent_workers=True) for s, d in ds.items()}
    counts = np.bincount(data["train_labels"].ravel(), minlength=5)
    class_weights = torch.tensor(len(ds["train"]) / (5 * counts), dtype=torch.float32)
    print("Trọng số lớp:", class_weights.numpy().round(2))

    plot_samples(data)
    plot_distribution(data)

    results, histories, trained = {}, {}, {}
    for key in SPECS:
        model, res, hist, pt, yt = train(key, loaders, class_weights)
        results[key], histories[key], trained[key] = res, hist, model

    best_key = max(["cnn", "resnet18", "mobilenet"], key=lambda k: results[k]["test"]["qwk"])
    for key in ["cnn", "resnet18", "mobilenet"]:
        torch.save(trained[key].state_dict(), MODELS / f"ch3_{key}.pt")
    # mô hình deploy: chọn theo QWK validation tốt nhất, không theo test (tránh chọn mô hình bằng tập kiểm tra)
    deploy = max(["cnn", "resnet18", "mobilenet"], key=lambda k: max(r["val_qwk"] for r in histories[k]))
    stats = {s: {"n": int(len(data[f"{s}_labels"])),
                 "counts": np.bincount(data[f"{s}_labels"].ravel(), minlength=5).tolist()} for s in ds}
    px = data["train_images"].reshape(-1, 3) / 255.0
    save_results("ch3", {
        "dataset": {"name": "RetinaMNIST (MedMNIST v2, 128×128)", "splits": stats,
                    "pixel_mean": px.mean(0).round(4).tolist(), "pixel_std": px.std(0).round(4).tolist(),
                    "class_weights": class_weights.numpy().round(3).tolist()},
        "models": results, "history": histories, "best_test_key": best_key, "deploy_key": deploy,
    })
    plot_history(histories, results)
    plot_confusion(results[best_key])
    plot_feature_maps(trained["cnn"], ds["test"][0][0][None])
    plot_gradcam(trained["resnet18"], ds["test"], data["test_labels"].ravel())


if __name__ == "__main__":
    main()
