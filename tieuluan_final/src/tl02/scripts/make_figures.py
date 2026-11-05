"""Vẽ toàn bộ hình dữ liệu của báo cáo từ outputs/metrics/*.json (không huấn luyện lại gì).

Chạy:  python scripts/make_figures.py [nhóm ...]   nhóm ∈ dia ret kg mk fc adv grag agent  (mặc định: tất cả)
Hình PDF vector ghi vào Report/TieuLuan 02/figures/.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_style import C, H1, SEQ, W, save, vn  # noqa: E402
from tl2.common import METRICS, load_json  # noqa: E402

STRATS = ["fixed", "recursive", "structure", "semantic", "parent_child"]
STRAT_VN = {"fixed": "Cố định", "recursive": "Đệ quy", "structure": "Theo cấu trúc", "semantic": "Ngữ nghĩa",
            "parent_child": "Cha–con"}
RETS = ["tfidf", "bm25", "dense", "hybrid", "hybrid_rerank"]
RET_VN = {"tfidf": "TF-IDF", "bm25": "BM25", "dense": "Dense e5", "hybrid": "Hybrid RRF", "hybrid_rerank": "Hybrid+Rerank"}
DIVERGE = LinearSegmentedColormap.from_list("rb", [C["red"], "#FFFFFF", C["blue"]])
GREENS = LinearSegmentedColormap.from_list("g", ["#F0FDF4", C["teal"], C["navy"]])


def has(name):
    return (METRICS / f"{name}.json").exists()


def annotate_heat(ax, M, fmt="{:.2f}", thr=None):
    thr = np.nanmean(M) if thr is None else thr
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, fmt.format(M[i, j]), ha="center", va="center", fontsize=10.5,
                    color="white" if M[i, j] > thr else "#111827", fontweight="bold")


# ===================================================================== Chương 4: chẩn đoán đái tháo đường
def fig_diabetes():
    from tl2.data import load_diabetes
    d = load_json("diabetes")
    df = load_diabetes()
    pos, neg = df[df.diabetes == 1], df[df.diabetes == 0]

    fig, axs = plt.subplots(2, 2, figsize=(W, 5.0))
    for ax, col, title, bins in zip(axs.flat, ["HbA1c_level", "blood_glucose_level", "bmi", "age"],
                                    ["HbA1c (%)", "Glucose máu (mg/dL)", "BMI (kg/m²)", "Tuổi"],
                                    [np.arange(3.4, 9.2, 0.2), np.arange(70, 310, 10), 40, 40]):
        ax.hist(neg[col], bins=bins, density=True, alpha=0.55, color=C["blue"], label="Không mắc")
        ax.hist(pos[col], bins=bins, density=True, alpha=0.6, color=C["red"], label="Mắc ĐTĐ")
        ax.set_title(title)
        ax.set_yticks([])
    axs[0, 0].axvline(6.5, color=C["navy"], ls="--", lw=1.6)
    axs[0, 0].text(6.6, axs[0, 0].get_ylim()[1] * 0.85, "6,5 %", color=C["navy"], fontsize=10.5)
    axs[0, 1].axvline(126, color=C["navy"], ls="--", lw=1.6)
    axs[0, 0].legend(loc="upper left", fontsize=10)
    fig.tight_layout()
    save(fig, "dia_dist")

    h = pd.DataFrame(d["history"])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W, 2.9))
    a1.plot(h.epoch, h.train_loss, label="Huấn luyện", color=C["blue"])
    a1.plot(h.epoch, h.val_loss, label="Kiểm định", color=C["orange"])
    a1.set_xlabel("Epoch"), a1.set_title("Hàm mất mát BCE (có pos_weight)"), a1.legend()
    a2.plot(h.epoch, h.val_pr_auc, label="PR-AUC", color=C["green"])
    a2.plot(h.epoch, h.val_roc_auc, label="ROC-AUC", color=C["purple"])
    best = h.val_pr_auc.idxmax()
    a2.scatter([h.epoch[best]], [h.val_pr_auc[best]], s=80, color=C["red"], zorder=5)
    a2.annotate(f"tốt nhất\n{h.val_pr_auc[best]:.3f}", (h.epoch[best], h.val_pr_auc[best]),
                xytext=(-55, -38), textcoords="offset points", fontsize=10, arrowprops=dict(arrowstyle="->"))
    a2.set_xlabel("Epoch"), a2.set_title("Chỉ số trên tập kiểm định"), a2.legend(loc="lower right")
    fig.tight_layout()
    save(fig, "dia_history")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W, 3.1))
    for key, name, col in (("mlp", "MLP (GPU)", C["blue"]), ("logreg", "Logistic (CPU)", C["orange"])):
        fpr, tpr = d["roc"][key]
        a1.plot(fpr, tpr, color=col, label=f"{name}  AUC={d[key]['roc_auc']:.3f}")
    a1.plot([0, 1], [0, 1], ls=":", color=C["gray"])
    a1.set_xlabel("Tỉ lệ dương tính giả"), a1.set_ylabel("Độ nhạy"), a1.set_title("Đường ROC"), a1.legend(fontsize=9.5)
    rc, pr = d["pr"]
    a2.plot(rc, pr, color=C["green"])
    a2.fill_between(rc, pr, alpha=0.15, color=C["green"])
    a2.axhline(d["data"]["pos_rate"], ls=":", color=C["gray"])
    a2.scatter([d["mlp"]["recall"]], [d["mlp"]["precision"]], s=90, color=C["red"], zorder=5)
    a2.annotate(f"ngưỡng {d['mlp']['threshold']:.2f}", (d["mlp"]["recall"], d["mlp"]["precision"]),
                xytext=(-95, -40), textcoords="offset points", fontsize=10, arrowprops=dict(arrowstyle="->"))
    a2.set_xlabel("Độ nhạy (recall)"), a2.set_ylabel("Độ chính xác (precision)")
    a2.set_title(f"Precision–Recall  AP={d['mlp']['pr_auc']:.3f}")
    fig.tight_layout()
    save(fig, "dia_roc_pr")

    fig, axs = plt.subplots(1, 2, figsize=(W, 2.9))
    for ax, key, name in zip(axs, ("mlp", "logreg"), ("MLP (GPU)", "Logistic Regression")):
        M = np.array(d[key]["confusion"])
        ax.imshow(M / M.sum(1, keepdims=True), cmap=GREENS, vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, vn(M[i, j]), ha="center", va="center", fontsize=14, fontweight="bold",
                        color="white" if M[i, j] / M[i].sum() > 0.5 else "#111827")
        ax.set_xticks([0, 1], ["Dự đoán 0", "Dự đoán 1"]), ax.set_yticks([0, 1], ["Thực 0", "Thực 1"])
        ax.set_title(f"{name}\nF1 = {d[key]['f1']:.3f}"), ax.grid(False)
    fig.tight_layout()
    save(fig, "dia_confusion")

    imp = pd.Series(d["importance"]).sort_values()
    fig, ax = plt.subplots(figsize=(W, 3.0))
    cols = [C["red"] if v > 0.1 else C["orange"] if v > 0.02 else C["sky"] for v in imp.values]
    ax.barh(imp.index, imp.values, color=cols)
    for i, v in enumerate(imp.values):
        ax.text(v + 0.006, i, f"{v:.3f}", va="center", fontsize=10.5)
    ax.set_xlabel("Mức giảm PR-AUC khi xáo trộn đặc trưng"), ax.set_xlim(0, imp.max() * 1.15)
    ax.set_title("Độ quan trọng hoán vị của MLP trên tập kiểm thử")
    fig.tight_layout()
    save(fig, "dia_importance")

    cal = pd.DataFrame(d["calibration"])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W, 2.9), gridspec_kw={"width_ratios": [1.3, 1]})
    a1.plot([0, 1], [0, 1], ls=":", color=C["gray"], label="Hiệu chỉnh hoàn hảo")
    a1.plot(cal.pred, cal.obs, marker="o", ms=7, color=C["purple"], label="MLP")
    a1.set_xlabel("Xác suất dự đoán"), a1.set_ylabel("Tần suất thực tế"), a1.set_title("Biểu đồ tin cậy"), a1.legend(fontsize=9.5)
    a2.bar(range(len(cal)), cal.n, color=C["teal"])
    a2.set_yscale("log"), a2.set_xticks(range(len(cal)), [f"{p:.1f}" for p in cal.pred], rotation=60, fontsize=9)
    a2.set_title("Số hồ sơ mỗi ngăn"), a2.set_xlabel("Xác suất TB của ngăn")
    fig.tight_layout()
    save(fig, "dia_calibration")

    hc = d["hba1c_curve"]
    fig, ax = plt.subplots(figsize=(W, 3.1))
    for g, p, col in zip(hc["glucose"], hc["prob"], (C["green"], C["orange"], C["red"])):
        ax.plot(hc["hba1c"], p, color=col, lw=2.6, label=f"glucose {g} mg/dL")
    ax.axhline(d["mlp"]["threshold"], ls="--", color=C["navy"], lw=1.4)
    ax.text(4.05, d["mlp"]["threshold"] + 0.03, f"ngưỡng quyết định {d['mlp']['threshold']:.2f}", color=C["navy"], fontsize=10.5)
    ax.axvspan(5.7, 6.4, color=C["orange"], alpha=0.12)
    ax.axvspan(6.5, 9.0, color=C["red"], alpha=0.08)
    ax.text(5.75, 0.55, "tiền ĐTĐ", fontsize=10.5, color=C["brown"])
    ax.text(7.4, 0.55, "ngưỡng chẩn đoán ĐTĐ", fontsize=10.5, color=C["red"])
    ax.set_xlabel("HbA1c (%)"), ax.set_ylabel("P(đái tháo đường)")
    ax.set_title("Đáp ứng của MLP theo HbA1c (nam 55 tuổi, BMI 27)"), ax.legend(loc="center right", fontsize=10)
    fig.tight_layout()
    save(fig, "dia_hba1c_curve")

    keys = ["acc", "precision", "recall", "f1", "roc_auc", "pr_auc"]
    names = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"]
    x = np.arange(len(keys))
    fig, ax = plt.subplots(figsize=(W, 2.9))
    for k, (key, name, col) in enumerate((("mlp", "MLP (GPU)", C["blue"]), ("logreg", "Logistic (CPU)", C["orange"]))):
        v = [d[key][m] for m in keys]
        ax.bar(x + (k - 0.5) * 0.38, v, 0.38, color=col, label=name)
        for xi, vi in zip(x, v):
            ax.text(xi + (k - 0.5) * 0.38, vi + 0.01, f"{vi:.2f}", ha="center", fontsize=9.5)
    ax.set_xticks(x, names), ax.set_ylim(0.5, 1.05), ax.legend(ncol=2, loc="upper center")
    ax.set_title("So sánh MLP và Logistic Regression trên tập kiểm thử")
    fig.tight_layout()
    save(fig, "dia_compare")


# ===================================================================== Chương 3: truy xuất
def fig_retrieval():
    g = load_json("retrieval_grid")
    for metric, fname, title in (("hit@1", "ret_grid_hit1", "Hit@1"), ("mrr", "ret_grid_mrr", "MRR")):
        M = np.array([[g["grid"][s][r][metric] for r in RETS] for s in STRATS])
        fig, ax = plt.subplots(figsize=(W, 3.4))
        ax.imshow(M, cmap=GREENS, vmin=M.min() - 0.05, vmax=1)
        annotate_heat(ax, M, "{:.3f}", thr=(M.min() + 1) / 2)
        ax.set_xticks(range(len(RETS)), [RET_VN[r] for r in RETS], rotation=12)
        ax.set_yticks(range(len(STRATS)), [STRAT_VN[s] for s in STRATS])
        ax.set_title(f"{title} trên 114 câu hỏi: chiến lược chia đoạn × bộ truy xuất"), ax.grid(False)
        fig.tight_layout()
        save(fig, fname)

    fig, ax = plt.subplots(figsize=(W, 3.2))
    data = [g["chunk_stats"][s]["lengths"] for s in STRATS]
    parts = ax.violinplot(data, showmedians=True, widths=0.8)
    for pc, col in zip(parts["bodies"], SEQ):
        pc.set_facecolor(col), pc.set_alpha(0.55)
    for i, s in enumerate(STRATS):
        st = g["chunk_stats"][s]
        ax.text(i + 1, max(data[i]) + 25, f"{st['n_chunks']} đoạn", ha="center", fontsize=10)
    ax.set_xticks(range(1, 6), [STRAT_VN[s] for s in STRATS]), ax.set_ylabel("Độ dài đoạn (ký tự)")
    ax.set_title("Phân bố độ dài đoạn của 5 chiến lược (cha–con: đoạn cha)")
    fig.tight_layout()
    save(fig, "ret_chunk_lengths")

    sw = pd.DataFrame(load_json("retrieval_sweep"))
    fig, ax = plt.subplots(figsize=(W, 3.2))
    ax.plot(sw["size"], sw["bm25_hit@1"], marker="o", color=C["orange"], label="BM25 Hit@1")
    ax.plot(sw["size"], sw["dense_hit@1"], marker="s", color=C["blue"], label="Dense Hit@1")
    ax.plot(sw["size"], sw["dense_hit@5"], marker="^", ls="--", color=C["blue"], alpha=0.6, label="Dense Hit@5")
    ax.set_xscale("log"), ax.set_xticks(sw["size"], [str(s) for s in sw["size"]])
    ax.set_xlabel("Kích thước đoạn tối đa (ký tự, chia đệ quy)"), ax.set_ylabel("Tỉ lệ trúng")
    ax2 = ax.twinx()
    ax2.bar(sw["size"], sw["ctx_tokens_top5"], width=sw["size"] * 0.25, color=C["gray"], alpha=0.25)
    ax2.set_ylabel("Ký tự ngữ cảnh top-5", color=C["gray"]), ax2.grid(False)
    ax.set_zorder(ax2.get_zorder() + 1), ax.patch.set_visible(False)
    ax.legend(loc="lower right"), ax.set_title("Đánh đổi kích thước đoạn: độ trúng tăng, ngữ cảnh phình to")
    fig.tight_layout()
    save(fig, "ret_sweep")

    e = load_json("retrieval_embed")
    names = {"minilm": "MiniLM-L12\n(384 chiều)", "e5-small": "e5-small\n(384 chiều)", "e5-base": "e5-base\n(768 chiều)"}
    fig, ax = plt.subplots(figsize=(W, 3.0))
    x = np.arange(3)
    for k, (m, lab, col) in enumerate((("hit@1", "Hit@1", C["blue"]), ("hit@5", "Hit@5", C["green"]),
                                        ("mrr", "MRR", C["purple"]), ("health_hit@1", "Hit@1 y tế", C["red"]))):
        v = [e[n][m] for n in names]
        ax.bar(x + (k - 1.5) * 0.2, v, 0.2, color=col, label=lab)
        for xi, vi in zip(x, v):
            ax.text(xi + (k - 1.5) * 0.2, vi + 0.01, f"{vi:.2f}", ha="center", fontsize=8.5, rotation=90)
    ax.set_xticks(x, list(names.values())), ax.set_ylim(0.4, 1.13), ax.legend(ncol=4, fontsize=9.5, loc="upper left")
    ax.set_title("So sánh ba mô hình embedding đa ngôn ngữ (đoạn theo cấu trúc)")
    fig.tight_layout()
    save(fig, "ret_embedders")

    fig, ax = plt.subplots(figsize=(W, 3.5))
    marks = dict(zip(STRATS, "osD^v"))
    for r, col in zip(RETS, SEQ):
        for s in STRATS:
            v = g["grid"][s][r]
            ax.scatter(v["latency_ms"], v["hit@1"], color=col, marker=marks[s], s=70, edgecolor="white", lw=0.8)
    for r, col in zip(RETS, SEQ):
        ax.scatter([], [], color=col, s=60, label=RET_VN[r])
    for s in STRATS:
        ax.scatter([], [], color=C["gray"], marker=marks[s], s=60, label=STRAT_VN[s])
    ax.set_xscale("log"), ax.set_xlabel("Độ trễ trung bình mỗi truy vấn (ms, thang log)"), ax.set_ylabel("Hit@1")
    ax.legend(ncol=2, fontsize=9, loc="lower right"), ax.set_title("Chất lượng và chi phí: rerank đắt hơn ~20–40 lần")
    fig.tight_layout()
    save(fig, "ret_latency")

    fig, ax = plt.subplots(figsize=(W, 3.0))
    x = np.arange(len(RETS))
    for k, (dom, col) in enumerate((("health", C["red"]), ("finance", C["blue"]))):
        v = [g["grid"]["structure"][r][f"{dom}_hit@1"] for r in RETS]
        ax.bar(x + (k - 0.5) * 0.38, v, 0.38, color=col, label="Y tế (62 câu)" if dom == "health" else "Tài chính (52 câu)")
        for xi, vi in zip(x, v):
            ax.text(xi + (k - 0.5) * 0.38, vi + 0.01, f"{vi:.2f}", ha="center", fontsize=9.5)
    ax.set_xticks(x, [RET_VN[r] for r in RETS]), ax.set_ylim(0.5, 1.08), ax.legend(ncol=2, loc="upper left")
    ax.set_title("Hit@1 theo miền (đoạn theo cấu trúc)")
    fig.tight_layout()
    save(fig, "ret_domain")

    fig, ax = plt.subplots(figsize=(W, 3.0))
    ks = [1, 3, 5, 10]
    for r, col in zip(RETS, SEQ):
        ax.plot(ks, [g["grid"]["structure"][r][f"hit@{k}"] for k in ks], marker="o", color=col, label=RET_VN[r])
    ax.set_xticks(ks), ax.set_xlabel("k"), ax.set_ylabel("Hit@k"), ax.legend(fontsize=9.5, ncol=2)
    ax.set_title("Đường Hit@k (đoạn theo cấu trúc)")
    fig.tight_layout()
    save(fig, "ret_hitk")

    v = load_json("retrieval_viz")
    se = v["semantic_example"]
    fig, ax = plt.subplots(figsize=(W, 2.8))
    dist = np.array(se["dist"])
    ax.bar(range(len(dist)), dist, color=[C["red"] if x > se["threshold"] else C["sky"] for x in dist])
    ax.axhline(se["threshold"], ls="--", color=C["navy"])
    ax.text(len(dist) - 0.5, se["threshold"] + 0.004, "ngưỡng = phân vị 80", ha="right", color=C["navy"], fontsize=10.5)
    ax.set_xlabel("Vị trí giữa hai câu liên tiếp"), ax.set_ylabel("Khoảng cách cosin")
    ax.set_title(f"Chia đoạn ngữ nghĩa tài liệu {se['doc']}: cột đỏ là điểm cắt")
    fig.tight_layout()
    save(fig, "ret_semantic")

    pca = v["embedding_pca"]
    fig, ax = plt.subplots(figsize=(W, 3.9))
    docs = sorted(set(pca["doc"]))
    cmap = plt.get_cmap("tab20")
    for i, dname in enumerate(docs):
        idx = [k for k, x in enumerate(pca["doc"]) if x == dname]
        mk = "o" if dname.startswith("h") else "^"
        ax.scatter(np.array(pca["x"])[idx], np.array(pca["y"])[idx], s=55, marker=mk, color=cmap(i % 20),
                   edgecolor="white", lw=0.6, label=dname.split("-")[0])
    ax.legend(ncol=6, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.15), frameon=False)
    ax.set_xlabel(f"PC1 ({100 * pca['var_ratio'][0]:.1f} %)"), ax.set_ylabel(f"PC2 ({100 * pca['var_ratio'][1]:.1f} %)")
    ax.set_title("101 đoạn trong không gian e5-base (PCA 2D): ● y tế, ▲ tài chính")
    fig.tight_layout()
    save(fig, "ret_pca")

    sm = v["sim_matrix_sample"]
    M = np.array(sm["s"])
    fig, ax = plt.subplots(figsize=(W, 3.4))
    im = ax.imshow(M, cmap="magma", aspect="auto")
    ax.set_yticks(range(len(sm["q"])), sm["q"]), ax.set_xticks(range(len(sm["c"])), [c.split(":")[0][:3] + ":" + c.split(":")[1] for c in sm["c"]], rotation=70, fontsize=8.5)
    for i in range(M.shape[0]):
        j = int(M[i].argmax())
        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec=C["lime"], lw=2.4))
    fig.colorbar(im, ax=ax, fraction=0.03), ax.grid(False)
    ax.set_title("Độ tương đồng cosin câu hỏi × đoạn (ô xanh: đoạn gần nhất)")
    fig.tight_layout()
    save(fig, "ret_simmatrix")


# ===================================================================== KG
def radial_layout(G):
    """Nút bậc cao nhất ở tâm; các nút còn lại xếp trên một vòng, mỗi lớp thực thể chiếm một cung."""
    deg = dict(G.degree())
    hub = max(deg, key=deg.get)
    groups = {}
    for n in sorted(G, key=lambda n: (G.nodes[n]["type"], -deg[n])):
        if n != hub:
            groups.setdefault(G.nodes[n]["type"], []).append(n)
    total, pos, a = len(G) - 1, {hub: (0.0, 0.0)}, 0.0
    for members in groups.values():
        span = 2 * np.pi * len(members) / total
        for i, n in enumerate(members):
            th = a + span * (i + 0.5) / len(members)
            r = 1.0
            pos[n] = (r * np.cos(th), r * np.sin(th))
        a += span
    return pos, hub


def draw_kg(kg, fname, title, edge_color):
    G = nx.DiGraph()
    for n, dd in kg.G.nodes(data=True):
        G.add_node(n, **dd)
    for u, v_, k in kg.G.edges(keys=True):
        G.add_edge(u, v_, rel=k)
    pos, hub = radial_layout(G)
    types = sorted({dd["type"] for _, dd in G.nodes(data=True)})
    cmap = plt.get_cmap("tab20")
    col = {t: cmap(i % 20) for i, t in enumerate(types)}
    fig, ax = plt.subplots(figsize=(W, 6.9))
    deg = dict(G.degree())
    size = [90 + 45 * min(deg[n], 12) for n in G]
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=[edge_color(d["rel"]) for _, _, d in G.edges(data=True)],
                           width=1.2, arrowsize=9, alpha=0.7, node_size=size, connectionstyle="arc3,rad=0.12",
                           arrows=True)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=[col[G.nodes[n]["type"]] for n in G], node_size=size,
                           edgecolors="white", linewidths=1.2)
    for n, (x, y) in pos.items():
        lab = kg.label(n).split(" (")[0]
        lab = lab if len(lab) <= 22 else lab[:21] + "…"
        if n == hub:
            ax.text(0, -0.1, lab, ha="center", va="top", fontsize=9, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=C["navy"]))
            continue
        th = np.arctan2(y, x)
        r = np.hypot(x, y)
        tx, ty = (r + 0.06) * np.cos(th), (r + 0.06) * np.sin(th)
        deg_ = np.degrees(th)
        rot = deg_ if -90 <= deg_ <= 90 else deg_ + 180
        ax.text(tx, ty, lab, ha="left" if -90 <= deg_ <= 90 else "right", va="center", rotation=rot,
                rotation_mode="anchor", fontsize=7.6, color="#111827")
    for t in types:
        ax.scatter([], [], color=col[t], s=50, label=t)
    ax.legend(ncol=4, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 0.02), frameon=False)
    ax.set_xlim(-1.8, 1.8), ax.set_ylim(-1.75, 1.8), ax.set_aspect("equal")
    ax.set_title(title), ax.axis("off")
    fig.tight_layout()
    save(fig, fname)


def fig_kg():
    from tl2.kg_finance import SIGN, build_finance_kg
    from tl2.kg_health import build_health_kg
    k = load_json("kg")
    draw_kg(build_health_kg(), "kg_health_graph", "VietnamDiabetesKG: 46 thực thể, 59 quan hệ",
            lambda r: C["red"] if r in ("TUONG_TAC_NGUY_HIEM", "CHONG_CHI_DINH_KHI", "HAN_CHE_AN", "TANG_NGUY_CO") else "#94A3B8")
    draw_kg(build_finance_kg(), "kg_finance_graph", "VietnamFinanceKG: cạnh xanh (+) cùng chiều, đỏ (−) ngược chiều",
            lambda r: {1: C["green"], -1: C["red"]}.get(SIGN.get(r), "#94A3B8"))

    fig, axs = plt.subplots(1, 2, figsize=(W, 3.8))
    for ax, dom, title, col in zip(axs, ("health", "finance"), ("KG y tế", "KG tài chính"), (C["red"], C["blue"])):
        r = pd.Series(k[dom]["relations"]).sort_values()
        ax.barh(r.index, r.values, color=col, alpha=0.85)
        for i, v in enumerate(r.values):
            ax.text(v + 0.15, i, str(v), va="center", fontsize=9.5)
        ax.tick_params(axis="y", labelsize=8.5), ax.set_title(f"{title}: số cạnh theo vị từ")
    fig.tight_layout()
    save(fig, "kg_relations")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W, 3.0), gridspec_kw={"width_ratios": [1, 1.5]})
    x = np.arange(2)
    a1.bar(x - 0.2, [k["health"]["rdf_triples"], k["finance"]["rdf_triples"]], 0.4, color=C["sky"], label="Tường minh")
    a1.bar(x + 0.2, [k["health"]["rdf_after"], k["finance"]["rdf_after"]], 0.4, color=C["purple"], label="Sau suy diễn")
    for xi, (b, a) in enumerate(((k["health"]["rdf_triples"], k["health"]["rdf_after"]),
                                 (k["finance"]["rdf_triples"], k["finance"]["rdf_after"]))):
        a1.text(xi + 0.2, a + 8, f"+{a - b}", ha="center", fontsize=10.5, color=C["purple"], fontweight="bold")
    a1.set_xticks(x, ["Y tế", "Tài chính"]), a1.set_title("Số bộ ba RDF"), a1.legend(fontsize=9.5, loc="upper right")
    q = k["sparql"]
    names = list(q)
    a2.barh(np.arange(len(names)) + 0.2, [q[n]["rows_before_inference"] for n in names], 0.4, color=C["sky"], label="Trước suy diễn")
    a2.barh(np.arange(len(names)) - 0.2, [len(q[n]["rows"]) for n in names], 0.4, color=C["purple"], label="Sau suy diễn")
    a2.set_yticks(range(len(names)), [n.split("_", 1)[0] + " " + n.split("_", 1)[1].replace("_", " ")[:16] for n in names], fontsize=9)
    a2.invert_yaxis(), a2.set_title("Số dòng kết quả SPARQL"), a2.legend(fontsize=9.5, loc="lower right")
    fig.tight_layout()
    save(fig, "kg_inference")

    t = k["tabular"]["numeric_query"]
    fig, ax = plt.subplots(figsize=(W, 2.6))
    bars = ax.barh(["Truy vấn KG (lọc thuộc tính)", "TF-IDF top-20 trên văn bản"],
                   [t["kg_hits"] / t["truth"], t["tfidf_top20_correct"] / 20], color=[C["green"], C["red"]])
    for b, txt in zip(bars, (f"{t['kg_hits']}/{t['truth']} đúng, {t['kg_ms']:.0f} ms", f"{t['tfidf_top20_correct']}/20 đúng")):
        ax.text(b.get_width() + 0.01, b.get_y() + b.get_height() / 2, txt, va="center", fontsize=11)
    ax.set_xlim(0, 1.35), ax.set_xlabel("Tỉ lệ kết quả đúng")
    ax.set_title("Câu hỏi số liệu trên 96.128 hồ sơ: KG chính xác, tìm văn bản thì không")
    fig.tight_layout()
    save(fig, "kg_tabular")

    s = pd.Series(k["shock_dxy"])
    fig, ax = plt.subplots(figsize=(W, 3.0))
    ax.barh(s.index, s.values, color=[C["green"] if v > 0 else C["red"] for v in s.values])
    ax.axvline(0, color="#111827", lw=1)
    ax.set_xticks([-1, 0, 1], ["giảm (−)", "", "tăng (+)"]), ax.tick_params(axis="y", labelsize=9.5)
    ax.set_title("Lan truyền cú sốc DXY tăng trên KG tài chính (dấu tích theo đường)")
    fig.tight_layout()
    save(fig, "kg_shock")


# ===================================================================== Chương 5: thị trường và dự báo
MK_VN = {"FPT": "FPT", "HPG": "HPG", "VCB": "VCB", "VIC": "VIC", "VN30ETF": "VN30 ETF", "GOLD_VND": "Vàng quy đổi",
         "XAU": "XAU/USD", "USDVND": "USD/VND", "DXY": "DXY"}


def fig_market():
    m = load_json("market")
    p = pd.DataFrame(m["prices"])
    p["date"] = pd.to_datetime(p["date"])
    fig, ax = plt.subplots(figsize=(W, 3.4))
    for c, col in zip(["FPT", "HPG", "VCB", "VIC", "VN30ETF", "GOLD_VND"], SEQ):
        ax.plot(p.date, 100 * p[c] / p[c].iloc[0], color=col, lw=1.7, label=MK_VN[c])
    ax.set_yscale("log"), ax.set_ylabel("Chỉ số (đầu kỳ = 100, thang log)"), ax.legend(ncol=3, fontsize=9.5)
    ax.set_title("Diễn biến giá điều chỉnh 2015–2026")
    fig.tight_layout()
    save(fig, "mk_prices")

    gd = pd.DataFrame({k_: v_ for k_, v_ in m["gold"].items() if k_ != "last"})
    gd["date"] = pd.to_datetime(gd["date"])
    fig, axs = plt.subplots(3, 1, figsize=(W, 4.6), sharex=True)
    for ax, c, lab, col in zip(axs, ("xau", "usdvnd", "gold_vnd"), ("XAU/USD (USD/oz)", "USD/VND", "Vàng quy đổi (tr.đ/lượng)"),
                               (C["orange"], C["green"], C["red"])):
        ax.plot(gd.date, gd[c], color=col)
        ax.fill_between(gd.date, gd[c], gd[c].min(), color=col, alpha=0.12)
        ax.set_ylabel(lab, fontsize=10)
    last = m["gold"]["last"]
    axs[2].annotate(f"{vn(last['gold_vnd'], 1)} tr.đ\n({last['date']})", (gd.date.iloc[-1], last["gold_vnd"]),
                    xytext=(-110, -10), textcoords="offset points", fontsize=10, arrowprops=dict(arrowstyle="->"))
    axs[0].set_title("Giá vàng quy đổi = XAU × USD/VND × 1,2057 (oz → lượng)")
    fig.tight_layout()
    save(fig, "mk_gold")

    cols = m["cols"]
    fig, axs = plt.subplots(1, 2, figsize=(W, 3.5))
    for ax, key, title in zip(axs, ("corr_daily", "corr_20d"), ("Lợi suất ngày", "Lợi suất 20 phiên")):
        M = np.array([[m[key][a][b] for b in cols] for a in cols])
        ax.imshow(M, cmap=DIVERGE, vmin=-1, vmax=1)
        for i in range(len(cols)):
            for j in range(len(cols)):
                v_ = round(M[i, j], 1)
                ax.text(j, i, "0" if v_ == 0 else f"{v_:.1f}".replace("0.", ".").replace("-", "−"), ha="center",
                        va="center", fontsize=7)
        ax.set_xticks(range(len(cols)), [MK_VN[c].split()[0] for c in cols], rotation=70, fontsize=8.5)
        ax.set_yticks(range(len(cols)), [MK_VN[c].split()[0] for c in cols], fontsize=8.5)
        ax.set_title(f"Tương quan: {title}", fontsize=11.5), ax.grid(False)
    fig.tight_layout()
    save(fig, "mk_corr")

    fig, ax = plt.subplots(figsize=(W, 3.2))
    for (k_, v_), col in zip(m["rolling_corr"].items(), SEQ):
        ax.plot(pd.to_datetime(v_["date"]), v_["corr"], color=col, lw=1.6, label=k_.replace("-", " → ").replace("USDVND", "USD/VND"))
    ax.axhline(0, color="#111827", lw=0.8)
    ax.set_ylabel("Tương quan 252 phiên"), ax.legend(ncol=2, fontsize=9), ax.set_ylim(-1, 1.05)
    ax.set_title("Tương quan trượt của các cạnh KG theo thời gian (lợi suất 20 phiên)")
    fig.tight_layout()
    save(fig, "mk_rolling")

    lc = m["lag_corr"]
    fig, ax = plt.subplots(figsize=(W, 2.9))
    for (s, v_), col in zip(lc["series"].items(), SEQ):
        ax.plot(lc["lags"], v_, marker="o", color=col, label=s)
    ax.axhline(0, color="#111827", lw=0.8)
    ax.set_xlabel("Độ trễ (phiên) của cổ phiếu so với USD/VND"), ax.set_ylabel("Tương quan"), ax.legend(ncol=4)
    ax.set_title("Tương quan trễ: biến động tỷ giá đi trước cổ phiếu")
    fig.tight_layout()
    save(fig, "mk_lag")

    ev = pd.DataFrame(m["kg_edges_evidence"])
    fig, ax = plt.subplots(figsize=(W, 3.0))
    lab = [f"{a} → {b}" for a, b in zip(ev.source, ev.target)]
    y = np.arange(len(ev))
    ax.barh(y, ev.corr_20d, color=[C["green"] if a else C["red"] for a in ev.agree])
    ax.scatter(ev.theory_sign * 0.9, y, marker="D", s=60, color=C["navy"], zorder=5, label="Dấu lý thuyết trong KG")
    ax.set_yticks(y, lab, fontsize=9.5), ax.axvline(0, color="#111827", lw=0.8), ax.set_xlim(-1.05, 1.15)
    ax.set_xlabel("Tương quan thực nghiệm (lợi suất 20 phiên)"), ax.legend(loc="lower right", fontsize=9.5)
    ax.set_title("Kiểm chứng dấu cạnh KG bằng dữ liệu: xanh khớp, đỏ lệch")
    fig.tight_layout()
    save(fig, "mk_edges")

    ds = m["describe"]
    fig, ax = plt.subplots(figsize=(W, 3.3))
    for c, col in zip(cols, SEQ):
        d_ = ds[c]
        ax.scatter(d_["ann_vol"], d_["ann_ret"], s=2500 * abs(d_["max_dd"]) ** 1.5, color=col, alpha=0.6, edgecolor="white")
        ax.text(d_["ann_vol"], d_["ann_ret"], MK_VN[c].split()[0], ha="center", va="center", fontsize=9.5, fontweight="bold")
    ax.set_xlabel("Biến động năm hoá"), ax.set_ylabel("Lợi suất năm hoá")
    ax.set_title("Lợi suất – rủi ro 2015–2026 (bong bóng: mức sụt giảm tối đa)")
    fig.tight_layout()
    save(fig, "mk_risk")


def fig_forecast():
    f = load_json("forecast")
    targets = ["FPT", "HPG", "VCB", "VIC", "GOLD_VND"]
    fig, axs = plt.subplots(2, 1, figsize=(W, 4.6))
    for ax, t, unit in zip(axs, ("FPT", "GOLD_VND"), ("VND", "tr.đ/lượng")):
        s = f[t]["test_series"]
        dt = pd.to_datetime(s["date"])
        ax.plot(dt, s["true"], color="#111827", lw=1.5, label="Thực tế")
        ax.plot(dt, s["bilstm"], color=C["red"], lw=1.3, alpha=0.85, label="BiLSTM (dự báo 1 phiên)")
        ax.set_ylabel(unit), ax.set_title(f"{MK_VN[t]}: tập kiểm thử 2025–2026", fontsize=11.5), ax.legend(fontsize=9.5)
    fig.tight_layout()
    save(fig, "fc_test")

    fig, ax = plt.subplots(figsize=(W, 3.0))
    models = [("bilstm", "BiLSTM", C["red"]), ("lstm", "LSTM", C["orange"]), ("ridge", "Ridge", C["purple"])]
    x = np.arange(len(targets))
    for k, (mk, lab, col) in enumerate(models):
        v = [f[t][mk]["rmse_ratio"] for t in targets]
        ax.bar(x + (k - 1) * 0.27, v, 0.27, color=col, label=lab)
    ax.axhline(1, color="#111827", ls="--", lw=1.4)
    ax.text(4.45, 1.003, "naive", fontsize=10)
    ax.set_xticks(x, [MK_VN[t] for t in targets]), ax.set_ylim(0.95, 1.17), ax.legend(ncol=3)
    ax.set_ylabel("RMSE / RMSE naive"), ax.set_title("Không mô hình nào thắng rõ dự báo ngây thơ (< 1 là tốt hơn)")
    fig.tight_layout()
    save(fig, "fc_ratio")

    fig, ax = plt.subplots(figsize=(W, 2.9))
    for k, (mk, lab, col) in enumerate(models):
        v = [f[t][mk]["dir_acc"] for t in targets]
        ax.bar(x + (k - 1) * 0.27, v, 0.27, color=col, label=lab)
    ax.axhline(0.5, color="#111827", ls="--", lw=1.4)
    ax.set_xticks(x, [MK_VN[t] for t in targets]), ax.set_ylim(0.35, 0.65), ax.legend(ncol=3)
    ax.set_ylabel("Tỉ lệ đúng chiều"), ax.set_title("Độ chính xác hướng tăng/giảm quanh mức tung đồng xu")
    fig.tight_layout()
    save(fig, "fc_diracc")

    fig, axs = plt.subplots(1, 2, figsize=(W, 2.8))
    for ax, t in zip(axs, ("FPT", "GOLD_VND")):
        h = pd.DataFrame(f[t]["bilstm"]["history"])
        ax.plot(h.epoch, h.train, color=C["blue"], label="Huấn luyện")
        ax.plot(h.epoch, h.val, color=C["orange"], label="Kiểm định")
        ax.set_xlabel("Epoch"), ax.set_title(f"BiLSTM {MK_VN[t]}: MSE chuẩn hoá", fontsize=11.5), ax.legend(fontsize=9.5)
    fig.tight_layout()
    save(fig, "fc_history")

    # hai kiểu thất bại: FPT dự báo dao động mạnh, lệch dương; vàng co hẳn về 0 (gần như dự báo ngây thơ)
    fig, axs = plt.subplots(1, 2, figsize=(W, 3.2))
    for ax, t, col in zip(axs, ("FPT", "GOLD_VND"), (C["blue"], C["orange"])):
        s = f[t]["test_series"]
        rt, rb = 100 * np.array(s["r_true"]), 100 * np.array(s["r_bilstm"])
        ax.scatter(rt, rb, s=12, alpha=0.55, color=col)
        lim = 1.05 * max(abs(rt).max(), abs(rb).max())
        ax.plot([-lim, lim], [-lim, lim], color="#111827", ls="--", lw=1, label="dự báo hoàn hảo")
        ax.axhline(0, color=C["gray"], lw=0.8), ax.axvline(0, color=C["gray"], lw=0.8)
        ax.set_xlim(-lim, lim), ax.set_ylim(-lim, lim)
        ax.set_xlabel("Lợi suất thực tế (%)")
        ax.set_title(f"{MK_VN[t]}\nσ dự báo {vn(rb.std(), 2)}% · σ thực tế {vn(rt.std(), 2)}%", fontsize=11)
        ax.legend(fontsize=9, loc="upper left")
    axs[0].set_ylabel("Lợi suất dự báo (%)")
    fig.tight_layout()
    save(fig, "fc_scatter")


# ===================================================================== Advanced RAG, GraphRAG, Agent
def fig_advanced():
    a = load_json("advanced")["agg"]
    names = {"dense": "Dense gốc", "hyde": "HyDE", "hyde_plus_q": "HyDE + câu hỏi", "multiquery": "Multi-Query"}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W, 3.0), gridspec_kw={"width_ratios": [1.5, 1]})
    x = np.arange(len(names))
    for k, (m, lab, col) in enumerate((("hit@1", "Hit@1", C["blue"]), ("mrr", "MRR", C["purple"]), ("hit@5", "Hit@5", C["green"]))):
        v = [a[n][m] for n in names]
        a1.bar(x + (k - 1) * 0.27, v, 0.27, color=col, label=lab)
    a1.set_xticks(x, list(names.values()), fontsize=9.5), a1.set_ylim(0.6, 1.05), a1.legend(ncol=3, fontsize=9.5)
    a1.set_title("Tối ưu tiền truy xuất bằng LLM")
    w_ = [a[n]["wins_vs_dense"] for n in list(names)[1:]]
    l_ = [a[n]["losses_vs_dense"] for n in list(names)[1:]]
    y = np.arange(3)
    a2.barh(y, w_, color=C["green"], label="thắng")
    a2.barh(y, [-v for v in l_], color=C["red"], label="thua")
    a2.set_yticks(y, list(names.values())[1:], fontsize=9.5), a2.axvline(0, color="#111827", lw=0.8)
    a2.set_title("Số câu thắng/thua MRR so với gốc", fontsize=11), a2.legend(fontsize=9)
    fig.tight_layout()
    save(fig, "adv_rag")


GRAG_VN = {"naive": "Naive RAG\n(cố định, dense)", "advanced": "Advanced RAG\n(hybrid + rerank)",
           "graph_local": "GraphRAG\ncục bộ", "graph_local_global": "GraphRAG\ncục bộ + toàn cục"}


def fig_graphrag():
    g = load_json("graphrag")
    A = g["agg"]
    names = list(GRAG_VN)
    short = ["Naive", "Advanced", "Graph\ncục bộ", "Graph\n+ toàn cục"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W, 3.5), gridspec_kw={"width_ratios": [1.55, 1.15]})
    x = np.arange(len(names))
    for k, (m, lab, col) in enumerate((("ctx_cov", "Độ phủ ngữ cảnh", C["blue"]),
                                        ("ans_cov", "Độ phủ câu trả lời", C["orange"]))):
        v = [A[n][m] for n in names]
        a1.bar(x + (k - 0.5) * 0.4, v, 0.4, color=col, label=lab)
        for xi, vi in zip(x, v):
            a1.text(xi + (k - 0.5) * 0.4, vi + 0.02, f"{vi:.2f}".replace(".", ","), ha="center", fontsize=9.5)
    a1.set_xticks(x, short, fontsize=9), a1.set_ylim(0, 1.32), a1.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    a1.legend(loc="upper center", ncol=2, fontsize=10, bbox_to_anchor=(0.5, 1.02))
    a1.set_title(f"Độ phủ dữ kiện ({len(g['rows'])} câu hỏi đa bước)", fontsize=11.5)
    v = [A[n]["ctx_chars"] for n in names]
    a2.bar(x, v, 0.62, color=SEQ[:4])
    for xi, vi in zip(x, v):
        a2.text(xi, vi + 60, vn(vi, 0), ha="center", fontsize=9.5)
    a2.set_xticks(x, ["Naive", "Adv.", "Graph\ncục bộ", "Graph\n+ t.cục"], fontsize=9), a2.set_ylim(0, max(v) * 1.15)
    a2.set_title("Ký tự ngữ cảnh trung bình", fontsize=11.5)
    fig.tight_layout()
    save(fig, "grag_coverage")

    # bố cục riêng cho từng cộng đồng: tâm cố định (y tế bên trái, tài chính bên phải), nút trải bằng spring trong
    # một hình tròn bán kính tỉ lệ căn bậc hai kích thước cộng đồng
    lay = g["layout"]
    G = nx.Graph()
    G.add_nodes_from((n["id"], n) for n in lay["nodes"])
    G.add_edges_from(map(tuple, lay["edges"]))
    rep = {r["id"]: r for r in g["reports"]}
    centers = {0: (0.0, 0.3), 2: (-1.2, -4.2), 5: (1.65, -3.95), 6: (2.95, -1.35), 1: (6.6, 0.3), 3: (5.35, -3.95), 4: (7.95, -3.95)}
    cmap = [C[k] for k in ("blue", "orange", "green", "red", "purple", "brown", "pink")]
    xy = {}
    for cid, r in rep.items():
        sub = G.subgraph(r["members"])
        rad = 0.36 * np.sqrt(r["size"])
        pos = nx.spring_layout(sub, seed=7, k=1.6 / np.sqrt(max(2, r["size"])))
        P = np.array(list(pos.values()))
        P = (P - P.mean(0)) / max(1e-9, np.abs(P - P.mean(0)).max())
        cx, cy = centers.get(cid, (3.9, 2.5))
        for nid, p in zip(pos, P):
            xy[nid] = (cx + rad * p[0], cy + rad * p[1])
    fig, ax = plt.subplots(figsize=(W, 5.2))
    for a, b in G.edges():
        same = G.nodes[a]["comm"] == G.nodes[b]["comm"]
        ax.plot([xy[a][0], xy[b][0]], [xy[a][1], xy[b][1]], color="#CBD5E1" if same else "#64748B",
                lw=0.7 if same else 0.9, ls="-" if same else (0, (3, 2)), zorder=1)
    for nid, d in G.nodes(data=True):
        ax.scatter(*xy[nid], s=28 + 9 * d["deg"], color=cmap[d["comm"] % len(cmap)], edgecolor="white", lw=0.7,
                   marker="o" if d["kg"] == "VietnamDiabetesKG" else "s", zorder=2)
    for cid, r in rep.items():
        cx, cy = centers.get(cid, (3.9, 2.5))
        rad = 0.36 * np.sqrt(r["size"])
        col = cmap[cid % len(cmap)]
        ax.add_patch(plt.Circle((cx, cy), rad + 0.22, fc=col, alpha=0.07, ec=col, lw=1.0, zorder=0))
        hub = max(r["members"], key=lambda m: G.nodes[m]["deg"])
        lab = G.nodes[hub]["label"].split(" (")[0]
        lab = lab if len(lab) <= 24 else lab[:24].rsplit(" ", 1)[0] + "…"
        ax.text(cx, cy - rad - 0.3, f"C{cid} · {r['size']} nút\n{lab}", ha="center", va="top", fontsize=8.5,
                color=col, fontweight="bold")
    ax.axvline(3.95, color="#94A3B8", ls=(0, (4, 3)), lw=1)
    ax.text(0.5, 2.75, "VietnamDiabetesKG (●)", ha="center", fontsize=10.5, color=C["red"], fontweight="bold")
    ax.text(6.6, 2.75, "VietnamFinanceKG (■)", ha="center", fontsize=10.5, color=C["teal"], fontweight="bold")
    ax.set_xlim(-2.6, 9.4), ax.set_ylim(-6.5, 3.1), ax.set_aspect("equal"), ax.axis("off")
    ax.set_title(f"{g['n_comms']} cộng đồng Louvain, modularity {g['modularity']:.2f}".replace(".", ","), fontsize=12)
    fig.tight_layout()
    save(fig, "grag_communities")


AGENT_VN = {"tool_recall": "Recall\ncông cụ", "tool_precision": "Precision\ncông cụ", "coverage": "Độ phủ\ndữ kiện",
            "faith": "Trung thực\nsố học", "llm_pass": "Qua\nkiểm chứng", "domain_acc": "Đúng\nmiền"}


def fig_agent():
    a = load_json("agent")
    G = a["agg"]
    keys = list(AGENT_VN)
    fig, ax = plt.subplots(figsize=(W, 3.4))
    x = np.arange(len(keys))
    for k, (dom, lab, col) in enumerate((("health", "Y tế", C["red"]), ("finance", "Tài chính", C["blue"]),
                                         ("all", "Chung", C["purple"]))):
        v = [G[dom][m] for m in keys]
        ax.bar(x + (k - 1) * 0.27, v, 0.27, color=col, label=lab)
        for xi, vi in zip(x, v):
            ax.text(xi + (k - 1) * 0.27, vi + 0.015, f"{vi:.2f}".replace(".", ","), ha="center", fontsize=7)
    ax.set_xticks(x, [AGENT_VN[m] for m in keys], fontsize=9.5), ax.set_ylim(0, 1.32), ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.03))
    ax.set_title(f"Đánh giá tác tử ReAct trên {len(a['rows'])} kịch bản")
    fig.tight_layout()
    save(fig, "agent_eval")

    fig, ax = plt.subplots(figsize=(W, 2.9))
    pairs = (("coverage", "coverage_llm_only", "Độ phủ dữ kiện bắt buộc"), ("faith", "faith_llm_only", "Trung thực số học"))
    x = np.arange(2)
    for k, (who, lab, col) in enumerate(((0, "Agent (công cụ + bằng chứng)", C["green"]), (1, "Chỉ LLM, không công cụ", C["gray"]))):
        v = [G["all"][p[who]] for p in pairs]
        ax.bar(x + (k - 0.5) * 0.38, v, 0.38, color=col, label=lab)
        for xi, vi in zip(x, v):
            ax.text(xi + (k - 0.5) * 0.38, vi + 0.015, f"{vi:.2f}".replace(".", ","), ha="center", fontsize=10.5, fontweight="bold")
    ax.set_xticks(x, [p[2] for p in pairs]), ax.set_ylim(0, 1.2), ax.legend(ncol=2, loc="upper center")
    ax.set_title("Cùng LLM Qwen2.5-1.5B: có và không có công cụ")
    fig.tight_layout()
    save(fig, "agent_vs_llm")

    calls = pd.Series(G["tool_calls"]).sort_values()
    lat = pd.Series(G["tool_ms"]).reindex(calls.index)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W, 3.0))
    a1.barh(calls.index, calls.values, color=C["teal"]), a1.set_title("Số lần gọi (16 kịch bản)", fontsize=11)
    a1.tick_params(axis="y", labelsize=9)
    a2.barh(lat.index, lat.values, color=C["orange"]), a2.set_xscale("log"), a2.set_yticks([])
    for i, v in enumerate(lat.values):
        a2.text(v * 1.1, i, vn(v, 0) if v >= 10 else vn(v, 1), va="center", fontsize=9)
    a2.set_title("Trung vị độ trễ (ms, log)", fontsize=11)
    fig.tight_layout()
    save(fig, "agent_tools")


GROUPS = {"dia": fig_diabetes, "ret": fig_retrieval, "kg": fig_kg, "mk": fig_market, "fc": fig_forecast,
          "adv": fig_advanced, "grag": fig_graphrag, "agent": fig_agent}
NEEDS = {"adv": "advanced", "grag": "graphrag", "agent": "agent"}

if __name__ == "__main__":
    for name in sys.argv[1:] or list(GROUPS):
        if name in NEEDS and not has(NEEDS[name]):
            sys.exit(f"thiếu outputs/metrics/{NEEDS[name]}.json: chạy thí nghiệm tương ứng trước")
        GROUPS[name]()
