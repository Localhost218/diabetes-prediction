"""Hình phân tích bổ sung cần nạp lại mô hình đã huấn luyện (không huấn luyện lại).

    python scripts/make_extra.py [ch2 ch3 ch4]

Chương 2: ROC/PR tiểu đường, hệ số Logistic, độ quan trọng đặc trưng Random Forest, phần dư giá nhà.
Chương 3: ảnh bị phân loại sai, bản đồ CAM, bộ lọc và bản đồ đặc trưng ResNet-18, F1 theo lớp.
Chương 4: lợi suất dự báo - thực tế, ROC/PR rời bỏ, phân bố xác suất rời bỏ.
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_style import C, W, plt, save, vn  # noqa: E402

from tl import data_tabular as dt  # noqa: E402
from tl.common import METRICS, MODELS, SEED, merge_json  # noqa: E402

torch.set_num_threads(8)
ARCH = {"custom": ("Custom CNN", C["gray"]), "vgg11": ("VGG-11", C["blue"]),
        "mobilenetv2": ("MobileNetV2", C["orange"]), "resnet18": ("ResNet-18", C["green"])}
CELL = {"rnn": ("Simple RNN", C["gray"]), "lstm": ("LSTM", C["blue"]), "gru": ("GRU", C["orange"]), "bilstm": ("BiLSTM", C["green"])}


VN = {"age": "Tuổi", "bmi": "BMI", "HbA1c_level": "HbA1c", "blood_glucose_level": "Đường huyết",
      "hypertension": "Tăng huyết áp", "heart_disease": "Bệnh tim", "log_area": "log diện tích", "Frontage": "Mặt tiền",
      "Access Road": "Đường vào", "Floors": "Số tầng", "Bedrooms": "Phòng ngủ", "Bathrooms": "Phòng tắm"}


def nice(feature):
    """Tên cột sau ColumnTransformer ('num__bmi', 'cat__province_Hà Nội') -> nhãn tiếng Việt."""
    f = feature.split("__", 1)[1]
    f = f.replace("missingindicator_", "thiếu ").replace("_infrequent_sklearn", " (nhóm hiếm)")
    for pre, vi in (("smoking_history_", "hút thuốc: "), ("gender_", "giới: "), ("province_", "tỉnh: "),
                    ("district_", "quận: "), ("Legal status_", "pháp lý: "), ("Furniture state_", "nội thất: "),
                    ("House direction_", "hướng: ")):
        if f.startswith(pre):
            return vi + f[len(pre):]
    for k, v in VN.items():
        f = f.replace(k, v) if f == k or f.startswith("thiếu ") else f
    return VN.get(f, f)


def roc_pr(curves, name, pos_rate):
    """curves: [(nhãn, màu, y_true, điểm)] -> hình ROC (trái) và Precision-Recall (phải)."""
    fig, axs = plt.subplots(1, 2, figsize=(W, 3.4))
    out = {}
    for lab, c, y, s in curves:
        fpr, tpr, _ = roc_curve(y, s)
        pr, rc, _ = precision_recall_curve(y, s)
        auc, ap = roc_auc_score(y, s), average_precision_score(y, s)
        axs[0].plot(fpr, tpr, color=c, lw=2, label=f"{lab} ({vn(auc, 3)})")
        axs[1].plot(rc, pr, color=c, lw=2, label=f"{lab} ({vn(ap, 3)})")
        out[lab] = {"auc": auc, "ap": ap}
    axs[0].plot([0, 1], [0, 1], color="k", ls=":", lw=1)
    axs[1].axhline(pos_rate, color="k", ls=":", lw=1)
    axs[0].set_xlabel("Tỉ lệ dương tính giả")
    axs[0].set_ylabel("Tỉ lệ dương tính thật")
    axs[1].set_xlabel("Recall")
    axs[1].set_ylabel("Precision")
    axs[0].set_title("ROC (AUC)", fontsize=11)
    axs[1].set_title("Precision–Recall (AP)", fontsize=11)
    axs[0].legend(fontsize=8.5, loc="lower right")
    axs[1].legend(fontsize=8.5, loc="lower left")
    fig.tight_layout()
    save(fig, name)
    return out


# ================================================================= Chương 2
def ch2():
    from tl.mlp_torch import TorchMLP
    df = dt.diabetes_frame()
    X, y = df.drop(columns="diabetes"), df.diabetes.to_numpy()
    _, Xte, _, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    m = {k: joblib.load(MODELS / "ch2" / f"diabetes_{k}.joblib") for k in ("logreg", "svm", "rf")}
    d = dt.load_diabetes()
    assert (d["y_test"] == yte).all()
    mlp = TorchMLP(d["X_train"].shape[1])
    mlp.load_state_dict(torch.load(MODELS / "ch2" / "diabetes_mlp_torch.pt", map_location="cpu"))
    mlp.eval()
    with torch.no_grad():
        p_mlp = torch.sigmoid(mlp(torch.as_tensor(d["X_test"]))).numpy().ravel()
    res = roc_pr([("Logistic", C["sky"], yte, m["logreg"].predict_proba(Xte)[:, 1]),
                  ("SVM RBF", C["purple"], yte, m["svm"].decision_function(Xte)),
                  ("Random Forest", C["orange"], yte, m["rf"].predict_proba(Xte)[:, 1]),
                  ("MLP PyTorch", C["red"], yte, p_mlp)], "f2_roc_diab", yte.mean())
    # ngưỡng: F1 của MLP khi hạ ngưỡng
    ths = np.linspace(0.05, 0.95, 91)
    f1s = [f1_score(yte, p_mlp >= t) for t in ths]
    rec = [((p_mlp >= t) & (yte == 1)).sum() / yte.sum() for t in ths]
    prec = [((p_mlp >= t) & (yte == 1)).sum() / max((p_mlp >= t).sum(), 1) for t in ths]
    fig, ax = plt.subplots(figsize=(W, 3.2))
    ax.plot(ths, f1s, color=C["red"], lw=2.2, label="F1")
    ax.plot(ths, rec, color=C["blue"], lw=2, label="Recall")
    ax.plot(ths, prec, color=C["green"], lw=2, label="Precision")
    ax.axvline(0.5, color="k", ls=":", lw=1)
    ax.set_xlabel("Ngưỡng quyết định của MLP PyTorch")
    ax.legend(ncol=3)
    save(fig, "f2_threshold_diab")
    best = int(np.argmax(f1s))
    res["mlp_best_thr"] = float(ths[best])
    res["mlp_best_f1"] = float(f1s[best])
    res["mlp_rec_at_03"] = float(rec[int(np.argmin(abs(ths - 0.3)))])
    res["mlp_prec_at_03"] = float(prec[int(np.argmin(abs(ths - 0.3)))])

    # cây quyết định nông (độ sâu 2) trên đặc trưng gốc: minh họa cho sơ đồ cây ở mục 2.1.3
    from sklearn.tree import DecisionTreeClassifier
    cols = ["age", "hypertension", "heart_disease", "bmi", "HbA1c_level", "blood_glucose_level"]
    Xtr_raw, _, ytr_raw, _ = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    tree = DecisionTreeClassifier(max_depth=2, random_state=SEED).fit(Xtr_raw[cols], ytr_raw)
    tr = tree.tree_
    nodes = []
    for i in range(tr.node_count):
        v = tr.value[i][0]
        nodes.append({"feature": cols[tr.feature[i]] if tr.feature[i] >= 0 else "la", "thr": float(tr.threshold[i]),
                      "gini": float(tr.impurity[i]), "n": int(tr.n_node_samples[i]), "p1": float(v[1] / v.sum())})
    yt_tree = tree.predict(Xte[cols])
    res["tree"] = {"nodes": nodes, "acc": float((yt_tree == yte).mean()), "f1": float(f1_score(yte, yt_tree))}

    lr = m["logreg"]
    names = [nice(n) for n in lr.named_steps["pre"].get_feature_names_out()]
    coef = lr.named_steps["model"].coef_.ravel()
    o = np.argsort(coef)
    fig, ax = plt.subplots(figsize=(W, 4.6))
    ax.barh(np.array(names)[o], coef[o], color=[C["red"] if v > 0 else C["blue"] for v in coef[o]])
    ax.set_xlabel("Hệ số Logistic (đặc trưng số đã chuẩn hóa)")
    ax.axvline(0, color="k", lw=1)
    save(fig, "f2_lr_coef")
    res["lr_coef"] = dict(zip(names, coef))

    hd = dt.house_frame()
    Xh, yh = hd[dt.HOUSE_NUM + dt.HOUSE_CAT], hd.Price.to_numpy(float)
    _, Xht, _, yht = train_test_split(Xh, yh, test_size=0.2, random_state=SEED)
    fig, axs = plt.subplots(2, 1, figsize=(W, 7.4))
    for ax, (pipe, title, c) in zip(axs, [(m["rf"], "Tiểu đường", C["orange"]),
                                          (joblib.load(MODELS / "ch2" / "house_rf.joblib"), "Giá nhà", C["green"])]):
        fn = pipe.named_steps["pre"].get_feature_names_out()
        imp = pipe.named_steps["model"].feature_importances_
        o = np.argsort(imp)[-12:]
        lab = [nice(f) for f in fn[o]]
        ax.barh(lab, imp[o], color=c)
        ax.set_title(f"{title}: 12 đặc trưng quan trọng nhất (giảm Gini / phương sai)", fontsize=11)
    fig.tight_layout()
    save(fig, "f2_rf_importance")

    house_rf = joblib.load(MODELS / "ch2" / "house_rf.joblib")
    yp = house_rf.predict(Xht)
    resid = yht - yp
    fig, axs = plt.subplots(1, 2, figsize=(W, 3.3), gridspec_kw={"width_ratios": [1.5, 1]})
    axs[0].scatter(yp, resid, s=4, alpha=0.25, color=C["green"])
    axs[0].axhline(0, color="k", lw=1)
    axs[0].set_xlabel("Giá dự báo (tỷ)")
    axs[0].set_ylabel("Phần dư (tỷ)")
    axs[1].hist(resid, bins=50, color=C["green"], alpha=0.8, orientation="horizontal")
    axs[1].set_xlabel("Số tin")
    fig.tight_layout()
    save(fig, "f2_house_resid")
    res["house_resid_within1"] = float((abs(resid) <= 1).mean())
    res["house_resid_within2"] = float((abs(resid) <= 2).mean())
    merge_json("extra", "ch2", res)


# ================================================================= Chương 3
def load_cnn(ds, arch):
    from tl.cnn_torch import ARCHS
    ck = torch.load(MODELS / "ch3" / f"{ds}_{arch}.pt", map_location="cpu")
    mdl = ARCHS[arch](10, **ck.get("kwargs", {}))
    mdl.load_state_dict(ck["state"])
    return mdl.eval(), ck


def ch3():
    from tl import data_vision as dv
    res = {}
    for ds in ("svhn", "eurosat"):
        d = dv.load(ds)
        y = d["y_test"]
        # F1 theo lớp
        fig, ax = plt.subplots(figsize=(W, 3.6))
        x = np.arange(10)
        for k, (a, (lab, c)) in enumerate(ARCH.items()):
            pred = np.load(METRICS / f"ch3_pred_{ds}_{a}.npz")["pred"]
            f = f1_score(y, pred, average=None)
            ax.bar(x + (k - 1.5) * 0.2, f * 100, 0.2, color=c, label=lab)
        lo = 88 if ds == "svhn" else 90
        ax.set_ylim(lo, 100)
        ax.set_xticks(x, d["classes"], rotation=35 if ds == "eurosat" else 0, ha="right" if ds == "eurosat" else "center")
        ax.set_ylabel("F1 theo lớp (%)")
        ax.legend(ncol=4, fontsize=9, loc="lower center", bbox_to_anchor=(0.5, 1.0))
        save(fig, f"f3_perclass_{ds}")

        pred = np.load(METRICS / f"ch3_pred_{ds}_resnet18.npz")["pred"]
        prob = np.load(METRICS / f"ch3_pred_{ds}_resnet18.npz")["prob"].astype(float)
        wrong = np.flatnonzero(pred != y)
        rng = np.random.default_rng(3)
        pick = rng.choice(wrong, 16, replace=False)
        sv = ds == "svhn"
        # tên lớp EuroSAT dài: lưới 4x4 với tên đầy đủ, SVHN giữ 2x8
        fig, axs = plt.subplots(2, 8, figsize=(W, 2.35)) if sv else plt.subplots(4, 4, figsize=(W, 8.2))
        for ax, i in zip(axs.ravel(), pick):
            ax.imshow(d["x_test"][i].transpose(1, 2, 0))
            ax.axis("off")
            pct = f"{vn(100 * prob[i, pred[i]], 0)} %"
            if sv:
                ax.set_title(f"{d['classes'][y[i]]}→{d['classes'][pred[i]]}\n{pct}", fontsize=8,
                             fontweight="normal", color=C["red"])
            else:
                ax.set_title(f"{d['classes'][y[i]]}\n→ {d['classes'][pred[i]]}\n{pct}", fontsize=9.5,
                             fontweight="normal", color=C["red"])
        fig.subplots_adjust(wspace=0.08 if sv else 0.9, hspace=0.45 if sv else 0.55)
        save(fig, f"f3_wrong_{ds}")
        res[f"{ds}_n_wrong"] = int(len(wrong))

        # CAM của ResNet-18
        mdl, ck = load_cnn(ds, "resnet18")
        idx = rng.choice(np.flatnonzero(pred == y), 6, replace=False)
        x = (torch.as_tensor(d["x_test"][idx]).float() / 255 - ck["mean"]) / ck["std"]
        with torch.no_grad():
            fmap = mdl.feature_maps(x)
            cls = mdl(x).argmax(1)
            cams = torch.relu(torch.einsum("bc,bchw->bhw", mdl.fc.weight[cls], fmap))
            cams = torch.nn.functional.interpolate(cams[:, None], size=x.shape[-1], mode="bilinear", align_corners=False)[:, 0]
        fig, axs = plt.subplots(2, 6, figsize=(W, 2.6))
        for j in range(6):
            img = d["x_test"][idx[j]].transpose(1, 2, 0)
            cam = cams[j].numpy()
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
            axs[0, j].imshow(img)
            axs[1, j].imshow(img)
            axs[1, j].imshow(cam, cmap="jet", alpha=0.45)
            axs[0, j].set_title(dv.short(d["classes"][int(cls[j])]), fontsize=9.5, fontweight="normal")
            for a in (axs[0, j], axs[1, j]):
                a.axis("off")
        fig.subplots_adjust(wspace=0.06, hspace=0.12)
        save(fig, f"f3_cam_{ds}")

    # bộ lọc tầng đầu và bản đồ đặc trưng (ResNet-18 EuroSAT)
    mdl, ck = load_cnn("eurosat", "resnet18")
    w = mdl.stem[0].weight.detach().numpy()                     # (64, 3, 3, 3)
    fig, axs = plt.subplots(4, 16, figsize=(W, 1.75))
    for i, ax in enumerate(axs.ravel()):
        f = w[i].transpose(1, 2, 0)
        ax.imshow((f - f.min()) / (f.max() - f.min() + 1e-8), interpolation="nearest")
        ax.axis("off")
    fig.subplots_adjust(wspace=0.08, hspace=0.08)
    save(fig, "f3_filters")
    d = dv.load("eurosat")
    i0 = int(np.flatnonzero(d["y_test"] == 7)[3])               # một ảnh khu dân cư
    x = (torch.as_tensor(d["x_test"][i0:i0 + 1]).float() / 255 - ck["mean"]) / ck["std"]
    with torch.no_grad():
        a1 = mdl.layer1(mdl.stem(x))[0]
        a2 = mdl.layer2(mdl.layer1(mdl.stem(x)))[0]
        a4 = mdl.feature_maps(x)[0]
    fig, axs = plt.subplots(3, 7, figsize=(W, 3.1))
    axs[0, 0].imshow(d["x_test"][i0].transpose(1, 2, 0))
    axs[0, 0].set_title("Ảnh vào", fontsize=9.5, fontweight="normal")
    rows = [(a1, "layer1\n64×64"), (a2, "layer2\n32×32"), (a4, "layer4\n8×8")]
    for r, (a, lab) in enumerate(rows):
        top = torch.argsort(a.mean((1, 2)), descending=True)[:6]
        for j, ch in enumerate(top):
            axs[r, j + 1].imshow(a[ch].numpy(), cmap="magma")
        if r > 0:                                   # cột đầu của hàng 2, 3 để trống: ghi tên tầng vào đó
            axs[r, 0].text(0.5, 0.5, lab, ha="center", va="center", fontsize=10, transform=axs[r, 0].transAxes)
    axs[0, 1].set_title("layer1 64×64 (6 kênh mạnh nhất)", fontsize=9.5, fontweight="normal", loc="left")
    for ax in axs.ravel():
        ax.axis("off")
    fig.subplots_adjust(wspace=0.06, hspace=0.1)
    save(fig, "f3_featmaps")
    merge_json("extra", "ch3", res)


# ================================================================= Chương 4
def ch4():
    from tl.data_seq import churn_data
    p = np.load(METRICS / "ch4_stock_pred_torch.npz")
    ret_true = (p["y_true"] / p["naive"] - 1) * 100
    ret_pred = (p["lstm"] / p["naive"] - 1) * 100
    fig, axs = plt.subplots(1, 2, figsize=(W, 3.3), gridspec_kw={"width_ratios": [1.3, 1]})
    axs[0].scatter(ret_true, ret_pred, s=8, alpha=0.5, color=C["blue"])
    axs[0].axhline(0, color="k", lw=0.8)
    axs[0].axvline(0, color="k", lw=0.8)
    axs[0].set_xlabel("Lợi suất thực tế (%)")
    axs[0].set_ylabel("Lợi suất LSTM dự báo (%)")
    axs[1].hist(ret_true, bins=40, color=C["gray"], alpha=0.7, label="Thực tế", density=True)
    axs[1].hist(ret_pred, bins=40, color=C["blue"], alpha=0.8, label="Dự báo", density=True)
    axs[1].set_xlabel("Lợi suất (%)")
    axs[1].legend()
    fig.tight_layout()
    save(fig, "f4_ret_scatter")
    corr = float(np.corrcoef(ret_true, ret_pred)[0, 1])

    d = churn_data()
    y = d["y_test"]
    curves = [(lab, c, y, np.load(METRICS / f"ch4_churn_prob_torch_{k}.npy")) for k, (lab, c) in CELL.items()]
    res = roc_pr(curves, "f4_roc_churn", y.mean())
    pl = np.load(METRICS / "ch4_churn_prob_torch_lstm.npy")
    fig, ax = plt.subplots(figsize=(W, 3.1))
    ax.hist(pl[y == 0], bins=50, range=(0, 0.8), density=True, alpha=0.5, color=C["green"], label="Ở lại")
    ax.hist(pl[y == 1], bins=50, range=(0, 0.8), density=True, alpha=0.5, color=C["red"], label="Rời bỏ")
    ax.set_xlabel("Xác suất rời bỏ do LSTM dự đoán")
    ax.set_ylabel("Mật độ")
    ax.legend()
    save(fig, "f4_prob_churn")
    res["ret_corr_lstm"] = corr
    res["pred_ret_std"] = float(ret_pred.std())
    res["true_ret_std"] = float(ret_true.std())
    merge_json("extra", "ch4", res)


if __name__ == "__main__":
    for w in sys.argv[1:] or ["ch2", "ch3", "ch4"]:
        globals()[w]()
