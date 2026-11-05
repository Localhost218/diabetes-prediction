"""Hình dữ liệu Chương 6 (figures/ch6_*.pdf) và ch6_summary.json (các số suy ra cho lời văn)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tl3.common import load_json, save_json  # noqa: E402
from tl3.plot_style import C, SEQ, W, plt, save, vn  # noqa: E402

NAMES = {"logreg": "LogReg", "mlp": "MLP", "gcn": "GCN", "sage": "GraphSAGE", "gat": "GAT"}
MCOL = {"logreg": C["gray"], "mlp": C["brown"], "gcn": C["blue"], "sage": C["green"], "gat": C["orange"]}
DS = ["Cora", "CiteSeer"]


def fig_class_dist(data):
    fig, axes = plt.subplots(1, 2, figsize=(W, 3.4))
    for ax, ds in zip(axes, DS):
        d = data[ds]
        ax.barh(range(d["classes"]), d["class_counts"], color=SEQ[:d["classes"]])
        ax.set_yticks(range(d["classes"]), [n.replace("_", " ") for n in d["class_names"]], fontsize=9.5)
        ax.invert_yaxis()
        ax.set_title(f"{ds} ({vn(d['nodes'])} nút)")
        ax.set_xlabel("số nút")
        for i, v in enumerate(d["class_counts"]):
            ax.text(v, i, f" {v}", va="center", fontsize=9)
    fig.tight_layout()
    save(fig, "ch6_class_dist")


def fig_degree(data):
    fig, ax = plt.subplots(figsize=(W, 3.2))
    for ds, c in zip(DS, (C["blue"], C["orange"])):
        h = np.array(data[ds]["deg_hist"], float)
        h = np.where(h > 0, h, np.nan)   # bậc không có nút nào: bỏ điểm thay vì vẽ log(0)
        ax.plot(range(len(h)), h / np.nansum(h), "o-", color=c, label=f"{ds} (trung bình {vn(data[ds]['deg_mean'], 2)})")
    ax.set_yscale("log")
    ax.set_xlabel("bậc của nút (30 = từ 30 trở lên)")
    ax.set_ylabel("tỷ lệ nút (log)")
    ax.legend()
    save(fig, "ch6_degree_hist")


def fig_acc_bars(node):
    fig, ax = plt.subplots(figsize=(W, 3.6))
    kinds = list(NAMES)
    x = np.arange(len(DS))
    w = 0.16
    for i, k in enumerate(kinds):
        m = [100 * node[ds][k]["test_mean"] for ds in DS]
        s = [100 * node[ds][k]["test_std"] for ds in DS]
        b = ax.bar(x + (i - 2) * w, m, w, yerr=s, capsize=3, color=MCOL[k], label=NAMES[k])
        for r, v, e in zip(b, m, s):  # nhãn đặt trên đầu thanh lỗi để không đè lên nó
            ax.text(r.get_x() + r.get_width() / 2, v + e + 1.0, vn(v, 1), ha="center", va="bottom", fontsize=8.5,
                    rotation=90)
    ax.set_xticks(x, DS)
    ax.set_ylabel("accuracy test (%)")
    ax.set_ylim(40, 95)
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.15), fontsize=9.5)
    save(fig, "ch6_acc_bars")


def fig_curves(node):
    fig, axes = plt.subplots(1, 2, figsize=(W, 3.3))
    for k in ("mlp", "gcn", "sage", "gat"):
        h = node["Cora"][k]["history"]
        axes[0].plot(h["loss"], color=MCOL[k], label=NAMES[k])
        axes[1].plot(np.array(h["val"]) * 100, color=MCOL[k], label=NAMES[k])
    axes[0].set_title("loss huấn luyện")
    axes[1].set_title("accuracy validation (%)")
    for ax in axes:
        ax.set_xlabel("epoch")
    axes[0].set_yscale("log")
    axes[1].legend(fontsize=9)
    fig.tight_layout()
    save(fig, "ch6_curves")


def fig_seeds(node):
    fig, axes = plt.subplots(1, 2, figsize=(W, 3.2), sharey=False)
    for ax, ds in zip(axes, DS):
        kinds = list(NAMES)
        runs = [np.array(node[ds][k]["test_runs"]) * 100 for k in kinds]
        bp = ax.boxplot(runs, patch_artist=True, widths=0.55, medianprops={"color": "black"})
        for p, k in zip(bp["boxes"], kinds):
            p.set_facecolor(MCOL[k])
            p.set_alpha(0.75)
        for i, r in enumerate(runs, 1):
            ax.scatter(np.full(len(r), i) + np.linspace(-0.12, 0.12, len(r)), r, s=14, color="black", zorder=3)
        ax.set_xticks(range(1, 6), [NAMES[k] for k in kinds], rotation=30, fontsize=9.5)
        ax.set_title(ds)
        ax.set_ylabel("accuracy test (%)")
    fig.tight_layout()
    save(fig, "ch6_seed_box")


def fig_time_params(node):
    fig, ax = plt.subplots(figsize=(W, 3.4))
    for ds, mk in zip(DS, ("o", "s")):
        for k in ("mlp", "gcn", "sage", "gat"):
            r = node[ds][k]
            ax.scatter(r["params"] / 1000, r["seconds"], s=90, color=MCOL[k], marker=mk, edgecolor="black")
            # MLP và GCN trên Cora gần trùng toạ độ: tách nhãn lên/xuống
            off = {("mlp", "Cora"): (9, 4), ("gcn", "Cora"): (9, -11)}.get((k, ds), (6, 4))
            ax.annotate(f"{NAMES[k]}·{ds}", (r["params"] / 1000, r["seconds"]), textcoords="offset points",
                        xytext=off, fontsize=9.5)
    ax.margins(y=0.2)                      # chừa chỗ cho nhãn đặt dưới điểm GCN·Cora
    ax.set_xlabel("số tham số (nghìn)")
    ax.set_ylabel("thời gian huấn luyện (giây, CPU)")
    save(fig, "ch6_time_params")


def fig_depth(depth):
    rows = depth["rows"]
    fig, ax = plt.subplots(figsize=(W, 3.6))
    for res, c, lab in ((False, C["blue"], "GCN"), (True, C["green"], "GCN + nối tắt")):
        rr = [r for r in rows if r["residual"] == res]
        L = [r["layers"] for r in rr]
        m = np.array([r["test_mean"] for r in rr]) * 100
        s = np.array([r["test_std"] for r in rr]) * 100
        ax.plot(L, m, "o-", color=c, label=f"accuracy — {lab}")
        ax.fill_between(L, m - s, m + s, color=c, alpha=0.15)
    ax.set_xlabel("số lớp GCN")
    ax.set_ylabel("accuracy test (%)")
    ax2 = ax.twinx()
    rr = [r for r in rows if not r["residual"]]
    ax2.plot([r["layers"] for r in rr], [r["mad"] for r in rr], "s--", color=C["red"], label="MAD — GCN")
    ax2.set_ylabel("MAD biểu diễn ẩn", color=C["red"])
    ax2.spines["right"].set_visible(True)
    ax2.grid(False)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="lower left", fontsize=9.5)
    save(fig, "ch6_depth")


def fig_propagation(depth):
    p = depth["propagation"]
    fig, ax = plt.subplots(figsize=(W, 3.0))
    ax.plot([r["k"] for r in p], [r["mad"] for r in p], "o-", color=C["purple"])
    ax.set_xscale("symlog", base=2)
    ax.set_xticks([r["k"] for r in p], [str(r["k"]) for r in p])
    ax.set_xlabel("số lần nhân $\\hat A$ (lan truyền không học)")
    ax.set_ylabel("MAD của $\\hat A^k X$")
    save(fig, "ch6_propagation")


def fig_perturb(pert, node):
    fig, ax = plt.subplots(figsize=(W, 3.5))
    for mode, c, lab in (("drop", C["blue"], "bỏ cạnh ngẫu nhiên"), ("shuffle", C["red"], "nối lại đầu cạnh ngẫu nhiên")):
        rr = [r for r in pert["rows"] if r["mode"] == mode]
        p = [100 * r["p"] for r in rr]
        m = np.array([r["test_mean"] for r in rr]) * 100
        ax.errorbar(p, m, yerr=np.array([r["test_std"] for r in rr]) * 100, fmt="o-", color=c, capsize=3, label=lab)
    ax.axhline(100 * node["Cora"]["mlp"]["test_mean"], color=C["brown"], ls=":", label="MLP (không dùng cạnh)")
    ax.set_xlabel("tỷ lệ cạnh bị can thiệp (%)")
    ax.set_ylabel("accuracy GCN trên Cora (%)")
    ax.legend(fontsize=9.5)
    save(fig, "ch6_perturb")


def fig_tsne(extra):
    t = extra["tsne"]
    y = np.array(t["labels"]["series"])
    fig, axes = plt.subplots(1, 3, figsize=(W, 2.6))
    for ax, key, title in zip(axes, ("raw", "gcn", "gat"), ("đặc trưng từ", "ẩn GCN", "ẩn GAT")):
        z = np.array(t[key]["series"])
        for c in range(int(y.max()) + 1):
            ax.scatter(z[y == c, 0], z[y == c, 1], s=2.5, color=SEQ[c])
        ax.set_title(f"{title}\nsilhouette {vn(t[key]['silhouette'], 3)}", fontsize=10.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    fig.tight_layout()
    save(fig, "ch6_tsne", ext="png", dpi=220)


def fig_confusion(extra, data):
    m = np.array(extra["confusion"]["matrix"], float)
    m = m / m.sum(1, keepdims=True)
    names = [n.replace("_", "\n") for n in data["Cora"]["class_names"]]
    fig, ax = plt.subplots(figsize=(W, 5.0))
    im = ax.imshow(m, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(m)):
        for j in range(len(m)):
            ax.text(j, i, vn(100 * m[i, j], 0), ha="center", va="center", fontsize=11.5,
                    color="white" if m[i, j] > 0.5 else "black")
    ax.set_xticks(range(len(m)), names, fontsize=10, rotation=45, ha="right")
    ax.set_yticks(range(len(m)), names, fontsize=10)
    ax.set_xlabel("dự đoán")
    ax.set_ylabel("nhãn thật")
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.04)
    save(fig, "ch6_confusion")


def fig_gat_attn(extra, data):
    g = extra["gat_attention"]
    nb = sorted(g["neighbors"], key=lambda r: -r["alpha_mean"])
    fig, ax = plt.subplots(figsize=(W, 3.2))
    labels = ["chính nó" if r["id"] == g["node"] else f"nút {r['id']}" for r in nb]
    cols = [C["green"] if r["label"] == g["label"] else C["red"] for r in nb]
    ax.bar(range(len(nb)), [r["alpha_mean"] for r in nb], color=cols)
    heads = np.array([r["alpha_heads"] for r in nb])
    for h in range(heads.shape[1]):
        ax.scatter(range(len(nb)), heads[:, h], s=12, color="black", alpha=0.45, zorder=3)
    ax.set_xticks(range(len(nb)), labels, rotation=35, fontsize=9.5)
    ax.set_ylabel("hệ số attention $\\alpha$")
    ax.set_title(f"Nút {g['node']} — xanh: cùng nhãn, đỏ: khác nhãn; chấm đen: từng đầu")
    save(fig, "ch6_gat_attn")


def fig_degree_acc(extra):
    rows = extra["degree_acc"]["rows"]
    fig, ax = plt.subplots(figsize=(W, 3.3))
    x = np.arange(len(rows))
    for i, k in enumerate(("mlp", "gcn", "gat")):
        ax.bar(x + (i - 1) * 0.26, [100 * (r[k] or 0) for r in rows], 0.26, color=MCOL[k], label=NAMES[k])
    ax.set_xticks(x, [f"bậc {r['bin']}\n(n={r['n']})" for r in rows], fontsize=9.5)
    ax.set_ylabel("accuracy test (%)")
    ax.set_ylim(30, 100)
    ax.legend(ncol=3, fontsize=9.5)
    save(fig, "ch6_degree_acc")


def fig_link(link):
    h = link["history"]
    fig, ax = plt.subplots(figsize=(W, 3.0))
    ax.plot([r["epoch"] for r in h], [r["loss"] for r in h], color=C["blue"], label="loss huấn luyện")
    ax2 = ax.twinx()
    ax2.plot([r["epoch"] for r in h], [r["val_auc"] for r in h], color=C["orange"], label="AUC validation")
    ax2.spines["right"].set_visible(True)
    ax2.grid(False)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss", color=C["blue"])
    ax2.set_ylabel("AUC", color=C["orange"])
    save(fig, "ch6_link_curve")


def summary(data, node, extra, depth, pert, link):
    """Các số suy ra (chênh lệch, cực trị) để lời văn gọi \\SL{ch6_summary/...} thay vì gõ tay."""
    s = {}
    for ds in DS:
        n = node[ds]
        best = max(("gcn", "sage", "gat"), key=lambda k: n[k]["test_mean"])
        s[ds] = {"best": NAMES[best], "best_acc": n[best]["test_mean"],
                 "gain_gcn_mlp": n["gcn"]["test_mean"] - n["mlp"]["test_mean"],
                 "gain_gat_mlp": n["gat"]["test_mean"] - n["mlp"]["test_mean"],
                 "gain_best_logreg": n[best]["test_mean"] - n["logreg"]["test_mean"]}
    rows = [r for r in depth["rows"] if not r["residual"]]
    rres = {r["layers"]: r for r in depth["rows"] if r["residual"]}
    b = max(rows, key=lambda r: r["test_mean"])
    s["depth"] = {"best_layers": b["layers"], "best_acc": b["test_mean"], "acc8": rows[-1]["test_mean"],
                  "acc8_res": rres[8]["test_mean"], "drop_1_8": b["test_mean"] - rows[-1]["test_mean"],
                  "mad2": rows[1]["mad"], "mad8": rows[-1]["mad"], "res_gain8": rres[8]["test_mean"] - rows[-1]["test_mean"]}
    pr = {(r["mode"], r["p"]): r for r in pert["rows"]}
    s["perturb"] = {"full": pr[("drop", 0.0)]["test_mean"], "drop50": pr[("drop", 0.5)]["test_mean"],
                    "drop100": pr[("drop", 1.0)]["test_mean"], "shuf50": pr[("shuffle", 0.5)]["test_mean"],
                    "shuf100": pr[("shuffle", 1.0)]["test_mean"], "homo_shuf100": pr[("shuffle", 1.0)]["homophily"],
                    "shuf100_vs_mlp": pr[("shuffle", 1.0)]["test_mean"] - node["Cora"]["mlp"]["test_mean"]}
    g = extra["gat_attention"]
    same = [r["alpha_mean"] for r in g["neighbors"] if r["label"] == g["label"]]
    s["gat"] = {"n_neighbors": len(g["neighbors"]), "same_label_share": float(sum(same)),
                "max_alpha": max(r["alpha_mean"] for r in g["neighbors"])}
    da = extra["degree_acc"]["rows"]
    s["degree"] = {"low_gain": (da[0]["gcn"] or 0) - (da[0]["mlp"] or 0), "high_gain": (da[-1]["gcn"] or 0) - (da[-1]["mlp"] or 0)}
    s["tsne"] = {"gain_gcn": extra["tsne"]["gcn"]["silhouette"] - extra["tsne"]["raw"]["silhouette"]}
    s["pyg"] = {"diff_gcn": node["Cora"]["gcn"]["test_mean"] - extra["pyg"]["gcn"]["test_mean"],
                "diff_gat": node["Cora"]["gat"]["test_mean"] - extra["pyg"]["gat"]["test_mean"]}
    save_json("ch6_summary", s)


def main():
    data, node, extra = load_json("ch6_data"), load_json("ch6_node"), load_json("ch6_extra")
    depth, pert, link = load_json("ch6_depth"), load_json("ch6_perturb"), load_json("ch6_link")
    fig_class_dist(data)
    fig_degree(data)
    fig_acc_bars(node)
    fig_curves(node)
    fig_seeds(node)
    fig_time_params(node)
    fig_depth(depth)
    fig_propagation(depth)
    fig_perturb(pert, node)
    fig_tsne(extra)
    fig_confusion(extra, data)
    fig_gat_attn(extra, data)
    fig_degree_acc(extra)
    fig_link(link)
    summary(data, node, extra, depth, pert, link)


if __name__ == "__main__":
    main()
