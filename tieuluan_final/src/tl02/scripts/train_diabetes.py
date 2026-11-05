"""Huấn luyện MLP chẩn đoán nguy cơ đái tháo đường (GPU) và Logistic Regression đối chứng (CPU).

Dữ liệu: diabetes_prediction_dataset (Kaggle), 100.000 hồ sơ; chia 70/15/15 phân tầng theo nhãn.
Ngưỡng quyết định chọn trên tập kiểm định để tối đa F1, rồi giữ nguyên khi chấm tập kiểm thử.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tl2.common import MODELS, device, save_json, seed_all  # noqa: E402
from tl2.data import diabetes_features, load_diabetes  # noqa: E402
from tl2.dl_models import DiabetesMLP  # noqa: E402

HIDDEN, DROPOUT, EPOCHS, BATCH, LR, PATIENCE = (128, 64), 0.2, 60, 1024, 1e-3, 8


def best_threshold(y, p):
    pr, rc, th = precision_recall_curve(y, p)
    f1 = 2 * pr * rc / np.maximum(pr + rc, 1e-12)
    return float(th[np.argmax(f1[:-1])])


def scores(y, p, thr):
    yh = (p >= thr).astype(int)
    return {"acc": float((yh == y).mean()), "precision": precision_score(y, yh), "recall": recall_score(y, yh),
            "f1": f1_score(y, yh), "roc_auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p),
            "threshold": thr, "confusion": confusion_matrix(y, yh).tolist()}


def fit_mlp(xtr, ytr, xva, yva, d_in, dev):
    """Huấn luyện MLP bằng BCE có pos_weight (bù lớp dương chỉ ~9%), dừng sớm theo PR-AUC trên tập kiểm định."""
    model = DiabetesMLP(d_in, HIDDEN, DROPOUT).to(dev)
    pos_weight = (ytr == 0).sum() / (ytr == 1).sum()
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=3)
    hist, best, best_state, wait = [], -1, None, 0
    for ep in range(1, EPOCHS + 1):
        model.train()
        perm = torch.randperm(len(xtr), device=dev)
        tot = 0.0
        for i in range(0, len(xtr), BATCH):
            b = perm[i:i + BATCH]
            opt.zero_grad()
            loss = loss_fn(model(xtr[b]), ytr[b])
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
        model.eval()
        with torch.no_grad():
            lva = model(xva)
            val_loss = loss_fn(lva, torch.tensor(yva, dtype=torch.float32, device=dev)).item()
            pva = torch.sigmoid(lva).cpu().numpy()
        ap = average_precision_score(yva, pva)
        sched.step(ap)
        hist.append({"epoch": ep, "train_loss": tot / len(xtr), "val_loss": val_loss, "val_pr_auc": ap,
                     "val_roc_auc": roc_auc_score(yva, pva), "lr": opt.param_groups[0]["lr"]})
        print(f"ep {ep:2d} loss {tot / len(xtr):.4f} val {val_loss:.4f} PR-AUC {ap:.4f}", flush=True)
        if ap > best:
            best, wait = ap, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, hist


def main():
    seed_all()
    dev = device()
    df = load_diabetes()
    X = diabetes_features(df)
    cols = list(X.columns)
    y = df["diabetes"].to_numpy()
    Xtr, Xtmp, ytr, ytmp = train_test_split(X.to_numpy(), y, test_size=0.30, stratify=y, random_state=42)
    Xva, Xte, yva, yte = train_test_split(Xtmp, ytmp, test_size=0.50, stratify=ytmp, random_state=42)
    mean, std = Xtr.mean(0), Xtr.std(0) + 1e-8
    norm = lambda a: (a - mean) / std  # noqa: E731

    # ------------------------------------------------ Logistic Regression (CPU)
    t0 = time.perf_counter()
    lr = LogisticRegression(max_iter=2000, class_weight="balanced").fit(norm(Xtr), ytr)
    lr_time = time.perf_counter() - t0
    p_va_lr, p_te_lr = lr.predict_proba(norm(Xva))[:, 1], lr.predict_proba(norm(Xte))[:, 1]
    lr_res = scores(yte, p_te_lr, best_threshold(yva, p_va_lr)) | {"train_s": lr_time, "device": "cpu"}

    # ------------------------------------------------ MLP (GPU)
    T = lambda a: torch.tensor(a, dtype=torch.float32, device=dev)  # noqa: E731
    xtr, xva, xte = T(norm(Xtr)), T(norm(Xva)), T(norm(Xte))
    t0 = time.perf_counter()
    model, hist = fit_mlp(xtr, T(ytr), xva, yva, len(cols), dev)
    train_s = time.perf_counter() - t0
    with torch.no_grad():
        p_va = torch.sigmoid(model(xva)).cpu().numpy()
        p_te = torch.sigmoid(model(xte)).cpu().numpy()
    thr = best_threshold(yva, p_va)
    mlp_res = scores(yte, p_te, thr) | {"train_s": train_s, "epochs": len(hist), "device": dev,
                                        "params": sum(p.numel() for p in model.parameters())}

    # ------------------------------------------------ hoán vị đặc trưng: giảm PR-AUC khi xáo trộn một cột
    rng = np.random.default_rng(0)
    base = average_precision_score(yte, p_te)
    groups = {"Tuổi": ["age"], "BMI": ["bmi"], "HbA1c": ["HbA1c_level"], "Glucose": ["blood_glucose_level"],
              "Giới tính": ["male"], "Tăng huyết áp": ["hypertension"], "Bệnh tim": ["heart_disease"],
              "Hút thuốc": [c for c in cols if c.startswith("smoke_")]}
    imp = {}
    for name, gcols in groups.items():
        Xp = Xte.copy()
        idx = [cols.index(c) for c in gcols]
        Xp[:, idx] = Xp[rng.permutation(len(Xp))][:, idx]
        with torch.no_grad():
            pp = torch.sigmoid(model(T(norm(Xp)))).cpu().numpy()
        imp[name] = float(base - average_precision_score(yte, pp))

    # ------------------------------------------------ đường cong và hiệu chỉnh xác suất
    fpr, tpr, _ = roc_curve(yte, p_te)
    fpr_l, tpr_l, _ = roc_curve(yte, p_te_lr)
    pr, rc, _ = precision_recall_curve(yte, p_te)
    sel = np.linspace(0, len(fpr) - 1, 200).astype(int)
    sel_l = np.linspace(0, len(fpr_l) - 1, 200).astype(int)
    sel_p = np.linspace(0, len(pr) - 1, 200).astype(int)
    bins = np.linspace(0, 1, 11)
    which = np.clip(np.digitize(p_te, bins) - 1, 0, 9)
    calib = [{"pred": float(p_te[which == b].mean()), "obs": float(yte[which == b].mean()), "n": int((which == b).sum())}
             for b in range(10) if (which == b).any()]
    hba = np.linspace(4.0, 9.0, 26)
    curve = []
    for g in (100, 140, 200):
        rows = []
        for h in hba:
            x = np.zeros(len(cols))
            v = dict(zip(cols, x))
            v.update(age=55, bmi=27.0, HbA1c_level=h, blood_glucose_level=g, male=1, smoke_never=1)
            rows.append([v[c] for c in cols])
        with torch.no_grad():
            curve.append(torch.sigmoid(model(T(norm(np.array(rows))))).cpu().numpy().tolist())

    MODELS.mkdir(exist_ok=True)
    torch.save(model.state_dict(), MODELS / "diabetes_mlp.pt")
    (MODELS / "diabetes_mlp.json").write_text(json.dumps(
        {"columns": cols, "mean": mean.tolist(), "std": std.tolist(), "threshold": thr, "hidden": list(HIDDEN),
         "dropout": DROPOUT}, indent=1), encoding="utf-8")
    save_json("diabetes", {
        "data": {"n": len(df), "pos_rate": float(y.mean()), "n_train": len(ytr), "n_val": len(yva), "n_test": len(yte),
                 "n_features": len(cols)},
        "mlp": mlp_res, "logreg": lr_res, "history": hist, "importance": imp,
        "roc": {"mlp": [fpr[sel].tolist(), tpr[sel].tolist()], "logreg": [fpr_l[sel_l].tolist(), tpr_l[sel_l].tolist()]},
        "pr": [rc[sel_p].tolist(), pr[sel_p].tolist()], "calibration": calib,
        "hba1c_curve": {"hba1c": hba.tolist(), "glucose": [100, 140, 200], "prob": curve}})
    print(json.dumps({"mlp": {k: v for k, v in mlp_res.items() if k != "confusion"},
                      "logreg_f1": lr_res["f1"]}, indent=1))


if __name__ == "__main__":
    main()
