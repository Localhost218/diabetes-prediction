"""Chương 5 — vẽ mọi hình dữ liệu (PDF vector, rộng 6,3 in) vào Report/TieuLuan 03/figures/ch5_*.pdf.

    python scripts/ch5_figures.py kb      # khảo sát dữ liệu, kho tài liệu, đồ thị, embedding
    python scripts/ch5_figures.py ret     # kết quả truy xuất (cần outputs/metrics/ch5_retrieval.json)
    python scripts/ch5_figures.py agent   # kết quả tác tử (cần outputs/metrics/ch5_agent_*.json)
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402

from tl3.common import limit_threads, load_json  # noqa: E402
from tl3.plot_style import C, SEQ, W, plt, save, vn, vn_axis  # noqa: E402
from tl3.ch5_agent import MODELS, load_domain  # noqa: E402

AGE_ORDER = ["dưới 20", "20–39", "40–59", "60–79", "từ 80"]
DOM_VN = {"diabetes": "Tiểu đường", "stock": "Cổ phiếu Google", "gold": "Quỹ vàng GLD"}
RET_ORDER = ["TF-IDF", "BM25", "Dense minilm", "Dense e5-small", "Dense e5-base", "Graph", "RRF BM25+Dense",
             "RRF BM25+Dense+Graph", "RRF+Rerank (không đồ thị)", "RRF+Graph+Rerank"]
RET_SHORT = {"TF-IDF": "TF-IDF", "BM25": "BM25", "Dense minilm": "MiniLM", "Dense e5-small": "e5-small",
             "Dense e5-base": "e5-base", "Graph": "Đồ thị", "RRF BM25+Dense": "RRF(B+D)",
             "RRF BM25+Dense+Graph": "RRF(B+D+G)", "RRF+Rerank (không đồ thị)": "RRF(B+D)\n+Rerank",
             "RRF+Graph+Rerank": "RRF(B+D+G)\n+Rerank"}


# ================================================================ khảo sát dữ liệu
def fig_diabetes(D):
    df = D.df
    fig, ax = plt.subplots(1, 2, figsize=(W, 3.0), gridspec_kw={"width_ratios": [1, 2]})
    cnt = df.diabetes.value_counts().sort_index()
    ax[0].bar(["Không", "Có"], cnt.values, color=[C["blue"], C["red"]])
    for i, v in enumerate(cnt.values):
        ax[0].text(i, v, vn(v), ha="center", va="bottom", fontsize=10.5)
    ax[0].set_title("Nhãn tiểu đường")
    vn_axis(ax[0])
    r = df.groupby("age_group").diabetes.mean().reindex(AGE_ORDER) * 100
    ax[1].bar(AGE_ORDER, r.values, color=SEQ[:5])
    for i, v in enumerate(r.values):
        ax[1].text(i, v, vn(v, 1) + "%", ha="center", va="bottom", fontsize=10.5)
    ax[1].set_title("Tỷ lệ mắc theo nhóm tuổi (%)")
    fig.tight_layout()
    save(fig, "ch5_dia_overview")

    fig, axs = plt.subplots(2, 2, figsize=(W, 5.2))
    for a, (col, order, title) in zip(axs.ravel(), [
            ("hba1c_class", ["bình thường", "tiền tiểu đường", "ngưỡng chẩn đoán"], "Theo phân loại HbA1c (ADA)"),
            ("glucose_class", ["dưới 100", "100–125", "126–199", "từ 200"], "Theo nhóm đường huyết (mg/dL)"),
            ("bmi_class", ["thiếu cân", "bình thường", "thừa cân", "béo phì"], "Theo phân loại BMI (WHO)"),
            ("smoking_history", ["never", "No Info", "former", "current", "not current", "ever"], "Theo tiền sử hút thuốc")]):
        r = df.groupby(col).diabetes.mean().reindex(order) * 100
        labels = [o.replace("No Info", "không TT").replace("not current", "hiện không")
                  .replace("never", "chưa hút").replace("former", "đã bỏ").replace("current", "đang hút")
                  .replace("ever", "từng hút") for o in order]
        a.bar(range(len(order)), r.values, color=SEQ[:len(order)])
        a.set_xticks(range(len(order)), labels, rotation=20, ha="right", fontsize=9.5)
        a.set_title(title, fontsize=11.5)
        for i, v in enumerate(r.values):
            a.text(i, v, vn(v, 1), ha="center", va="bottom", fontsize=9.5)
    fig.supylabel("Tỷ lệ mắc tiểu đường (%)", fontsize=11.5)
    fig.tight_layout()
    save(fig, "ch5_dia_rates")

    fig, ax = plt.subplots(figsize=(W, 2.9))
    sub = df[df.gender != "Other"]
    piv = (sub.groupby(["gender", "hypertension", "age_group"]).diabetes.mean().unstack()[AGE_ORDER] * 100)
    rows = [f"{'Nữ' if g == 'Female' else 'Nam'} · {'THA' if h else 'không THA'}" for g, h in piv.index]
    im = ax.imshow(piv.values, cmap="Reds", aspect="auto")
    ax.set_xticks(range(5), AGE_ORDER)
    ax.set_yticks(range(len(rows)), rows)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            ax.text(j, i, vn(v, 1), ha="center", va="center", fontsize=10.5, color="white" if v > 25 else "black")
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="% mắc")
    ax.set_title("Tỷ lệ mắc tiểu đường theo giới × tăng huyết áp × nhóm tuổi")
    save(fig, "ch5_dia_heat")


def fig_market(S, G):
    df = S.df
    fig, ax = plt.subplots(2, 1, figsize=(W, 4.6), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
    ax[0].plot(df.Date, df.Close, color=C["blue"], lw=1.4)
    ax[0].fill_between(df.Date, df.Close, color=C["blue"], alpha=0.08)
    ax[0].set_ylabel("Đóng cửa (USD)")
    ax[0].set_title("Cổ phiếu Alphabet (Google) 2015–2024, giá điều chỉnh chia tách")
    for d, t in (("2020-03-16", "Covid-19"), ("2022-11-03", "đáy 2022")):
        i = S.by_date.get(d)
        if i is not None:
            ax[0].annotate(t, (df.Date.iloc[i], df.Close.iloc[i]), xytext=(0, -30), textcoords="offset points",
                           ha="center", arrowprops={"arrowstyle": "->", "color": C["red"]}, color=C["red"], fontsize=10.5)
    m = df.set_index("Date").Volume.resample("ME").mean() / 1e6
    ax[1].bar(m.index, m.values, width=25, color=C["orange"])
    ax[1].set_ylabel("KL TB (triệu)")
    fig.tight_layout()
    save(fig, "ch5_stock_price")

    fig, ax = plt.subplots(1, 2, figsize=(W, 2.9))
    for a, D, name, col in ((ax[0], S, "Google", C["blue"]), (ax[1], G, "GLD", C["orange"])):
        r = D.df.ret.dropna()
        a.hist(r.clip(-8, 8), bins=60, color=col, alpha=0.85)
        a.set_title(f"Lợi suất ngày {name} (%)")
        a.axvline(0, color="black", lw=1)
        a.text(0.02, 0.95, f"σ = {vn(r.std(), 2)}%\nmin = {vn(r.min(), 1)}%\nmax = {vn(r.max(), 1)}%",
               transform=a.transAxes, va="top", fontsize=10)
    fig.tight_layout()
    save(fig, "ch5_returns")

    fig, ax = plt.subplots(1, 2, figsize=(W, 2.8))
    order = ["giảm mạnh", "giảm", "đi ngang", "tăng", "tăng mạnh"]
    for a, D, name in ((ax[0], S, "Google"), (ax[1], G, "GLD")):
        c = D.df.move.value_counts().reindex(order).fillna(0)
        a.bar(range(5), c.values, color=[C["red"], "#F87171", C["gray"], "#4ADE80", C["green"]])
        a.set_xticks(range(5), order, rotation=25, ha="right", fontsize=9.5)
        a.set_title(f"Nút khái niệm biến động — {name}", fontsize=11)
        for i, v in enumerate(c.values):
            a.text(i, v, vn(v), ha="center", va="bottom", fontsize=9.5)
    fig.tight_layout()
    save(fig, "ch5_moves")

    gd = G.df
    fig, ax = plt.subplots(figsize=(W, 3.3))
    for col, lab, c in (("Close", "Quỹ vàng GLD", C["orange"]), ("SP_close", "S&P 500 (SPY)", C["blue"]),
                        ("OF_Price", "Dầu Brent", C["gray"]), ("USDI_Price", "Chỉ số USD", C["green"]),
                        ("GDX_Close", "Quỹ mỏ vàng GDX", C["purple"])):
        ax.plot(gd.Date, 100 * gd[col] / gd[col].iloc[0], label=lab, color=c, lw=1.5)
    ax.axhline(100, color="black", lw=0.8, ls="--")
    ax.set_ylabel("Chỉ số (15/12/2011 = 100)")
    ax.legend(ncol=3, loc="upper left", fontsize=9.5)
    ax.set_title("FINAL\\_USO.csv: vàng và các chỉ báo thị trường 2011–2018".replace("\\", ""))
    save(fig, "ch5_gold_price")

    cols = ["Close", "GDX_Close", "SP_close", "DJ_close", "EU_Price", "OF_Price", "OS_Price", "USB_Price",
            "USDI_Price", "PLT_Price", "PLD_Price", "USO_Close"]
    labs = ["GLD", "GDX", "SPY", "Dow", "EUR/USD", "Brent", "WTI", "Lợi suất", "USD idx", "Bạch kim", "Palađi", "USO"]
    r = gd[cols].pct_change().corr()
    fig, ax = plt.subplots(figsize=(W, 5.0))
    im = ax.imshow(r.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)), labs, rotation=45, ha="right")
    ax.set_yticks(range(len(cols)), labs)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, vn(r.values[i, j], 2), ha="center", va="center", fontsize=8,
                    color="white" if abs(r.values[i, j]) > 0.6 else "black")
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.04)
    ax.set_title("Tương quan lợi suất ngày giữa các cột FINAL_USO")
    save(fig, "ch5_gold_corr")


# ================================================================ kho tài liệu & đồ thị
def fig_kb(doms):
    kb = {n: load_json(f"ch5_kb_{n}") for n in doms}
    fig, ax = plt.subplots(1, 3, figsize=(W, 2.6), sharey=False)
    for a, n, c in zip(ax, doms, SEQ):
        h = np.array(kb[n]["doc_len_hist"])
        a.bar(np.arange(len(h)), h, color=c, width=1.0)
        a.set_title(DOM_VN[n], fontsize=11)
        a.set_xlabel("số từ / tài liệu")
        vn_axis(a)
    fig.tight_layout()
    save(fig, "ch5_doc_len")

    fig, ax = plt.subplots(1, 2, figsize=(W, 3.0))
    x = np.arange(3)
    nodes = [kb[n]["graph"]["nodes"] for n in doms]
    edges = [kb[n]["graph"]["edges"] for n in doms]
    ax[0].bar(x - 0.2, nodes, 0.4, label="Nút", color=C["blue"])
    ax[0].bar(x + 0.2, edges, 0.4, label="Cạnh", color=C["orange"])
    ax[0].set_yscale("log")
    ax[0].set_xticks(x, ["Tiểu đường", "Google", "Vàng"], fontsize=10)
    ax[0].legend()
    ax[0].set_title("Quy mô đồ thị (thang log)")
    for i in range(3):
        ax[0].text(i - 0.2, nodes[i] * 1.15, vn(nodes[i]), ha="center", va="bottom", fontsize=9, rotation=90)
        ax[0].text(i + 0.2, edges[i] * 1.15, vn(edges[i]), ha="center", va="bottom", fontsize=9, rotation=90)
    ax[0].set_ylim(top=max(edges) * 30)
    rel = kb["diabetes"]["graph"]["relations"]
    ks = list(rel)[:9]
    ax[1].barh(ks[::-1], [rel[k] for k in ks][::-1], color=C["teal"])
    ax[1].set_title("Quan hệ (cạnh) — KG tiểu đường", fontsize=11)
    ax[1].tick_params(axis="y", labelsize=9)
    vn_axis(ax[1], "x")
    fig.tight_layout()
    save(fig, "ch5_kg_stats")

    times = {n: [kb[n]["corpus_s"], kb[n]["build_kg_s"], kb[n]["tfidf_s"], kb[n]["bm25_s"], kb[n]["emb_minilm_s"],
                 kb[n]["emb_e5-small_s"], kb[n]["emb_e5-base_s"]] for n in doms}
    labs = ["Văn bản hoá", "Dựng KG", "TF-IDF", "BM25", "MiniLM", "e5-small", "e5-base"]
    fig, ax = plt.subplots(figsize=(W, 3.3))
    x = np.arange(len(labs))
    for i, n in enumerate(doms):
        ax.bar(x + (i - 1) * 0.27, times[n], 0.27, label=DOM_VN[n], color=SEQ[i])
    ax.set_yscale("log")
    ax.set_xticks(x, labs, rotation=20)
    ax.set_ylabel("giây (thang log)")
    ax.legend()
    ax.set_title("Thời gian dựng kho tri thức và chỉ mục")
    save(fig, "ch5_build_time")


def fig_graphs(doms):
    fig, ax = plt.subplots(figsize=(W, 3.4))
    for n, c in zip(doms, SEQ):
        D = load_domain(n)
        deg = np.array([d for _, d in D.g.in_degree()])
        deg = deg[deg > 0]
        vals, cnts = np.unique(deg, return_counts=True)
        ax.loglog(vals, cnts, "o", ms=4, color=c, label=f"{DOM_VN[n]} (max {vn(deg.max())})", alpha=0.8)
        if n == "diabetes":
            fig_sub_diabetes(D)
        else:
            fig_sub_market(D, n)
        del D
    ax.set_xlabel("bậc vào")
    ax.set_ylabel("số nút")
    ax.legend()
    ax.set_title("Phân bố bậc vào: vài nút khái niệm trung tâm, rất nhiều nút bậc thấp")
    save(fig, "ch5_kg_degree")


def _draw(g, pos, colors, name, title, size=None, labels=None, fs=9.5, outside=(), xlim=1.55):
    """Vẽ đồ thị con; nút trong `outside` đặt nhãn ra phía ngoài theo hướng bán kính để không chồng nhau."""
    fig, ax = plt.subplots(figsize=(W, 5.0))
    nx.draw_networkx_edges(g, pos, ax=ax, alpha=0.35, arrows=True, arrowsize=8, edge_color="#64748B", width=0.8)
    nx.draw_networkx_nodes(g, pos, ax=ax, node_color=[colors[n] for n in g], node_size=size or 260,
                           edgecolors="white", linewidths=0.8)
    labels = labels or {}
    inner = {n: t for n, t in labels.items() if n not in outside}
    nx.draw_networkx_labels(g, pos, labels=inner, ax=ax, font_size=fs,
                            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75))
    for n in outside:
        x, y = pos[n]
        r = max(np.hypot(x, y), 1e-9)
        ax.text(x * (1 + 0.13 / r), y * (1 + 0.13 / r), labels[n], fontsize=fs, va="center",
                ha="left" if x > 0.05 else "right" if x < -0.05 else "center")
    ax.set_xlim(-xlim, xlim)
    ax.set_ylim(-1.17, 1.17)
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.93)
    ax.set_title(title)
    ax.axis("off")
    save(fig, name)


def fig_sub_diabetes(D):
    rng = np.random.default_rng(3)
    persons = rng.choice([n for n in D.g if n.startswith("Person_")], 40, replace=False)
    sub = D.g.subgraph(set(persons) | {o for p in persons for o in D.g.successors(p)} |
                       {"Diabetes_yes"}).copy()
    pal = {"Person": "#93C5FD", "Gender": C["orange"], "AgeGroup": C["green"], "BMI": C["teal"], "HbA1c": C["purple"],
           "Glucose": C["sky"], "Smoking": C["brown"], "Diabetes": C["red"], "Condition": C["pink"]}
    colors = {n: pal.get(n.split("_")[0], C["gray"]) for n in sub}
    # khái niệm xếp trên vòng ngoài (không chồng nhãn), hồ sơ ở vòng trong
    concepts = sorted((n for n in sub if not n.startswith("Person_")), key=lambda n: n.split("_")[0])
    pos = nx.shell_layout(sub, [list(persons), concepts])
    labels = {n: n.split("_", 1)[1] for n in concepts}
    size =[60 if n.startswith("Person_") else 420 for n in sub]
    _draw(sub, pos, colors, "ch5_kg_sub_diabetes",
          "Đồ thị con: 40 hồ sơ (vòng trong) nối tới các nút khái niệm dùng chung", size, labels, fs=10.5,
          outside=concepts, xlim=1.5)


def fig_sub_market(D, n):
    pre = "TradingDay_" if n == "stock" else "MarketDay_"
    month = "2020-03" if n == "stock" else "2013-04"
    days = [x for x in D.g if x.startswith(pre) and D.g.nodes[x].get("has_date", "").startswith(month)]
    sub = D.g.subgraph(set(days) | {o for d in days for o in D.g.successors(d)}).copy()
    pal = {pre[:-1]: C["blue"], "Month": C["green"], "Year": C["purple"], "Move": C["orange"]}
    colors = {x: pal.get(x.split("_")[0], C["gray"]) for x in sub}
    # ngày giao dịch xếp theo thứ tự thời gian trên vòng ngoài, nút tháng/kiểu biến động ở vòng trong
    hubs = sorted(x for x in sub if not x.startswith(pre))
    pos = nx.shell_layout(sub, [hubs, sorted(x for x in sub if x.startswith(pre))])
    pos.update({x: 0.9 * pos[x] for x in hubs})
    labels = {x: (x.split("_", 1)[1][-5:] if x.startswith(pre) else x.split("_", 1)[1]) for x in sub}
    size = [520 if x.startswith(pre) else 1300 for x in sub]
    _draw(sub, pos, colors, f"ch5_kg_sub_{n}",
          f"Đồ thị con tháng {month[5:]}/{month[:4]} — {DOM_VN[n]}: chuỗi next_day, tháng, kiểu biến động",
          size, labels, fs=11, outside=[x for x in sub if x.startswith(pre)], xlim=1.3)


def fig_embeddings():
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    D = load_domain("stock")
    E = np.load(MODELS / "stock_emb_e5-small.npy").astype(np.float32)
    rec = np.array([d["kind"] == "record" for d in D.docs])
    Z = TSNE(n_components=2, perplexity=40, random_state=0, init="pca").fit_transform(PCA(50).fit_transform(E))
    years = np.array([int(i[1:5]) if i.startswith("D") else int(i[2:6]) for i in D.ids])
    fig, ax = plt.subplots(figsize=(W, 4.2))
    sc = ax.scatter(Z[rec, 0], Z[rec, 1], c=years[rec], cmap="viridis", s=6, alpha=0.8)
    ax.scatter(Z[~rec, 0], Z[~rec, 1], c="red", marker="*", s=40, label="tài liệu tổng hợp tháng/năm")
    fig.colorbar(sc, ax=ax, label="năm")
    ax.legend(loc="lower right")
    ax.set_title("t-SNE của embedding e5-small — kho tài liệu cổ phiếu")
    ax.set_xticks([])
    ax.set_yticks([])
    save(fig, "ch5_emb_tsne")


# ================================================================ truy xuất
def fig_retrieval():
    R = load_json("ch5_retrieval")
    br = R["by_retriever"]
    names = [n for n in RET_ORDER if n in br]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(W, 3.6))
    for i, (m, lab) in enumerate((("r@1", "Recall@1"), ("r@5", "Recall@5"), ("mrr", "MRR@10"), ("ndcg", "nDCG@10"))):
        ax.bar(x + (i - 1.5) * 0.2, [br[n][m] for n in names], 0.2, label=lab, color=SEQ[i])
    ax.set_xticks(x, [RET_SHORT[n] for n in names], rotation=35, ha="right", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.legend(ncol=4, loc="upper left", fontsize=9.5)
    ax.set_title("Chất lượng truy xuất trên toàn bộ câu hỏi ba miền")
    save(fig, "ch5_ret_overall")

    bd = R["by_domain"]
    M = np.array([[bd.get(f"{d}|{n}", {}).get("mrr", np.nan) for d in ("diabetes", "stock", "gold")] for n in names])
    fig, ax = plt.subplots(figsize=(W, 4.4))
    im = ax.imshow(M, cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
    ax.set_yticks(range(len(names)), [RET_SHORT[n].replace("\n", " ") for n in names])
    ax.set_xticks(range(3), [DOM_VN[d] for d in ("diabetes", "stock", "gold")])
    for i in range(M.shape[0]):
        for j in range(3):
            ax.text(j, i, vn(M[i, j], 3), ha="center", va="center", fontsize=10, color="white" if M[i, j] > 0.6 else "black")
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="MRR@10")
    ax.set_title("MRR@10 theo miền và bộ truy xuất")
    save(fig, "ch5_ret_domain")

    bs = R["by_style"]
    fig, ax = plt.subplots(1, 3, figsize=(W, 4.0), sharey=True)
    for a, d in zip(ax, ("diabetes", "stock", "gold")):
        lex = [bs.get(f"{d}|lex|{n}", {}).get("r@1", 0) for n in names]
        para = [bs.get(f"{d}|para|{n}", {}).get("r@1", 0) for n in names]
        y = np.arange(len(names))
        a.barh(y + 0.2, lex, 0.4, color=C["blue"], label="đúng mặt chữ")
        a.barh(y - 0.2, para, 0.4, color=C["orange"], label="diễn đạt khác")
        a.set_title(DOM_VN[d], fontsize=11)
        a.set_xlim(0, 1)
    ax[0].set_yticks(np.arange(len(names)), [RET_SHORT[n].replace("\n", " ") for n in names], fontsize=8.5)
    fig.legend(*ax[0].get_legend_handles_labels(), ncol=2, loc="lower center", fontsize=9.5, frameon=False)
    fig.supxlabel("Recall@1", fontsize=11, y=0.08)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    save(fig, "ch5_ret_style")

    bt = R["by_type"]
    types = [t for t in ("record", "group", "month", "year", "indicator") if any(k.startswith(t + "|") for k in bt)]
    tvn = {"record": "bản ghi", "group": "nhóm", "month": "tháng", "year": "năm", "indicator": "chỉ báo"}
    M = np.array([[bt.get(f"{t}|{n}", {}).get("r@1", np.nan) for t in types] for n in names])
    fig, ax = plt.subplots(figsize=(W, 4.4))
    im = ax.imshow(M, cmap="PuRd", vmin=0, vmax=1, aspect="auto")
    ax.set_yticks(range(len(names)), [RET_SHORT[n].replace("\n", " ") for n in names])
    ax.set_xticks(range(len(types)), [tvn[t] for t in types])
    for i in range(M.shape[0]):
        for j in range(len(types)):
            ax.text(j, i, vn(M[i, j], 2), ha="center", va="center", fontsize=10, color="white" if M[i, j] > 0.6 else "black")
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="Recall@1")
    ax.set_title("Recall@1 theo loại câu hỏi")
    save(fig, "ch5_ret_type")

    fig, ax = plt.subplots(figsize=(W, 3.6))
    for i, n in enumerate(names):
        ax.scatter(br[n]["ms"], br[n]["mrr"], s=120, color=SEQ[i % len(SEQ)], edgecolor="black", zorder=3)
        right = br[n]["ms"] < 1000
        ax.annotate(RET_SHORT[n].replace("\n", " "), (br[n]["ms"], br[n]["mrr"]),
                    xytext=(9 if right else -9, 6 if i % 2 else -12), ha="left" if right else "right",
                    textcoords="offset points", fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("độ trễ trung bình mỗi truy vấn (ms, thang log)")
    ax.set_ylabel("MRR@10")
    ax.set_title("Đánh đổi chất lượng — độ trễ")
    save(fig, "ch5_ret_latency")

    fig, ax = plt.subplots(figsize=(W, 3.0))
    ks = [1, 3, 5, 10]
    for i, n in enumerate([n for n in names if n in ("BM25", "Dense e5-small", "Graph", "RRF BM25+Dense",
                                                      "RRF BM25+Dense+Graph", "RRF+Graph+Rerank")]):
        ax.plot(ks, [br[n][f"r@{k}"] for k in ks], "o-", label=RET_SHORT[n].replace("\n", " "), color=SEQ[i])
    ax.set_xticks(ks)
    ax.set_xlabel("k")
    ax.set_ylabel("Recall@k")
    ax.legend(ncol=2, fontsize=9)
    ax.set_title("Đường Recall@k")
    save(fig, "ch5_ret_recall")

    fig, ax = plt.subplots(figsize=(W, 2.9))
    emb = ["Dense minilm", "Dense e5-small", "Dense e5-base"]
    x = np.arange(3)
    for i, e in enumerate(emb):
        ax.bar(x + (i - 1) * 0.27, [bd[f"{d}|{e}"]["mrr"] for d in ("diabetes", "stock", "gold")], 0.27,
               label=RET_SHORT[e], color=SEQ[i + 3])
    ax.set_xticks(x, [DOM_VN[d] for d in ("diabetes", "stock", "gold")])
    ax.set_ylabel("MRR@10")
    ax.set_ylim(0, 0.8)
    ax.legend(ncol=3, loc="upper left")
    ax.set_title("So sánh ba mô hình embedding đa ngôn ngữ")
    save(fig, "ch5_ret_dense")

    sw = R["sweep_rows"]
    fig, ax = plt.subplots(1, 2, figsize=(W, 2.9))
    for i, d in enumerate(("diabetes", "stock", "gold")):
        rows = [r for r in sw if r["domain"] == d and r["param"] == "graph_weight"]
        ax[0].plot([r["value"] for r in rows], [r["mrr"] for r in rows], "o-", color=SEQ[i], label=DOM_VN[d])
        rows = [r for r in sw if r["domain"] == d and r["param"] == "rerank_n"]
        ax[1].plot([r["value"] for r in rows], [r["r@1"] for r in rows], "s-", color=SEQ[i], label=DOM_VN[d])
    ax[0].set_xlabel("trọng số nhánh đồ thị $w_G$")
    ax[0].set_ylabel("MRR@10")
    ax[0].set_title("RRF: trọng số đồ thị", fontsize=11)
    ax[1].set_xlabel("số ứng viên đưa vào rerank $n$")
    ax[1].set_ylabel("Recall@1")
    ax[1].set_title("Rerank: số ứng viên", fontsize=11)
    ax[1].legend(fontsize=9)
    fig.tight_layout()
    save(fig, "ch5_ret_sweep")


# ================================================================ tác tử
SYS = {"llm_only": "LLM thuần", "naive_rag": "Naive RAG", "advanced_rag": "Advanced RAG", "agent": "Tác tử"}


def fig_agent():
    A = {d: load_json(f"ch5_agent_{d}") for d in ("diabetes", "stock", "gold")}
    fig, ax = plt.subplots(figsize=(W, 3.6))
    x = np.arange(3)
    for i, s in enumerate(SYS):
        v = [A[d]["systems"][s]["acc"] * 100 for d in A]
        ax.bar(x + (i - 1.5) * 0.2, v, 0.2, label=SYS[s], color=SEQ[[9, 1, 0, 2][i]])
        for j, vv in enumerate(v):
            ax.text(x[j] + (i - 1.5) * 0.2, vv, vn(vv, 0), ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x, [DOM_VN[d] for d in A])
    ax.set_ylabel("Độ chính xác đáp án (%)")
    ax.set_ylim(0, 110)
    ax.legend(ncol=4, fontsize=10, loc="upper center", bbox_to_anchor=(0.5, -0.1), frameon=False)
    ax.set_title("Bốn hệ hỏi đáp trên câu hỏi có đáp án")
    fig.tight_layout()
    save(fig, "ch5_agent_acc")

    fig, ax = plt.subplots(figsize=(W, 3.0))
    data = [[r["ms"] / 1000 for d in A for r in A[d]["rows"] if r["system"] == s] for s in SYS]
    bp = ax.boxplot(data, tick_labels=list(SYS.values()), patch_artist=True, showfliers=False)
    for p, c in zip(bp["boxes"], [SEQ[9], SEQ[1], SEQ[0], SEQ[2]]):
        p.set_facecolor(c)
        p.set_alpha(0.7)
    ax.set_yscale("log")
    ax.set_ylabel("độ trễ (giây, log)")
    ax.set_title("Độ trễ một câu hỏi")
    save(fig, "ch5_agent_latency")

    fig, ax = plt.subplots(1, 2, figsize=(W, 3.6), gridspec_kw={"width_ratios": [1.5, 1.2]})
    types = {}
    for d in A:
        for r in A[d]["rows"]:
            types.setdefault((r["type"], r["system"]), []).append(r["correct"])
    tl = [t for t in ("record", "group", "month", "year", "indicator") if (t, "agent") in types]
    tvn = {"record": "bản ghi", "group": "nhóm", "month": "tháng", "year": "năm", "indicator": "chỉ báo"}
    for i, s in enumerate(("naive_rag", "advanced_rag", "agent")):
        ax[0].bar(np.arange(len(tl)) + (i - 1) * 0.27, [100 * np.mean(types[(t, s)]) for t in tl], 0.27,
                  label=SYS[s], color=SEQ[[1, 0, 2][i]])
    ax[0].set_xticks(np.arange(len(tl)), [tvn[t] for t in tl], rotation=25, ha="right")
    ax[0].set_ylabel("% đúng")
    ax[0].set_ylim(0, 105)
    ax[0].legend(fontsize=9, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.27), frameon=False)
    ax[0].set_title("Theo loại câu hỏi", fontsize=11)
    tu = {}
    for d in A:
        for k, v in A[d]["tool_use"].items():
            tu[k] = tu.get(k, 0) + v
    ax[1].pie(list(tu.values()), labels=list(tu.keys()), autopct="%1.0f%%", colors=SEQ[:len(tu)], radius=1.25,
              textprops={"fontsize": 9.5})
    ax[1].set_title("Công cụ tác tử đã chọn", fontsize=11)
    fig.tight_layout()
    save(fig, "ch5_agent_type")

    fig, ax = plt.subplots(figsize=(W, 3.6))
    off = {"diabetes": (-8, 6, "right"), "stock": (8, -12, "left"), "gold": (-8, -14, "right")}
    for i, s in enumerate(("naive_rag", "advanced_rag", "agent")):
        g = [100 * A[d]["systems"][s]["gold_in_sources"] for d in A]
        a = [100 * A[d]["systems"][s]["acc"] for d in A]
        ax.scatter(g, a, s=140, color=SEQ[[1, 0, 2][i]], label=SYS[s], edgecolor="black", zorder=3)
        if s == "agent":
            ax.annotate("tác tử: cả ba miền", (100, 100), xytext=(-10, 8), ha="right", textcoords="offset points",
                        fontsize=9)
            continue
        for j, d in enumerate(A):
            dx, dy, ha = off[d]
            ax.annotate(DOM_VN[d], (g[j], a[j]), xytext=(dx, dy), ha=ha, textcoords="offset points", fontsize=9)
    ax.plot([0, 100], [0, 100], "--", color="gray")
    ax.set_xlabel("% câu có tài liệu đúng trong nguồn")
    ax.set_ylabel("% đáp án đúng")
    ax.set_xlim(30, 108)
    ax.set_ylim(30, 110)
    ax.legend(fontsize=9.5, loc="lower right")
    ax.set_title("Truy xuất đúng là điều kiện cần của trả lời đúng")
    save(fig, "ch5_agent_goldsrc")


if __name__ == "__main__":
    limit_threads(4)
    what = sys.argv[1]
    if what == "kb":
        S, G = load_domain("stock"), load_domain("gold")
        fig_market(S, G)
        fig_kb(("diabetes", "stock", "gold"))
        D = load_domain("diabetes")
        fig_diabetes(D)
        del D
        fig_graphs(("diabetes", "stock", "gold"))
        fig_embeddings()
    elif what == "ret":
        fig_retrieval()
    elif what == "agent":
        fig_agent()
