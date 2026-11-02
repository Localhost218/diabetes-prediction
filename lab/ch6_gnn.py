"""
Chương 6 — Học sâu trên đồ thị với Pubmed-Diabetes (LINQS): 19.717 bài báo PubMed về đái tháo đường,
44.338 cạnh trích dẫn, mỗi bài là vector TF-IDF 500 từ; nhãn 3 lớp: ĐTĐ thực nghiệm (trên động vật),
ĐTĐ type 1, ĐTĐ type 2.

Ứng dụng trong hệ thống: tự động gán nhãn type 1 / type 2 cho tài liệu y văn trước khi nạp vào Knowledge Base.

Mô hình (PyTorch thuần, không dùng PyG — để thấy rõ phép truyền thông điệp):
  MLP, hồi quy logistic (chỉ đặc trưng) · Lan truyền nhãn (chỉ đồ thị) · GCN · GraphSAGE-mean · GAT
Hai giao thức: bán giám sát kiểu Planetoid (20 nhãn/lớp, 500 val, 1.000 test, 10 hạt giống) và có giám sát
đầy đủ (60/20/20).

    .venv/bin/python lab/ch6_gnn.py
"""

import copy
import time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.manifold import TSNE
from sklearn.metrics import f1_score

from common import ACCENT, DATA, GREY, MODELS, PALETTE, plt, r4, save_fig, save_results, seed_everything

torch.set_num_threads(6)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # huấn luyện trên GPU nếu có
ROOT_DIR = DATA / "pubmed" / "Pubmed-Diabetes" / "data"
CLASSES = ["ĐTĐ thực nghiệm", "ĐTĐ type 1", "ĐTĐ type 2"]


# ------------------------------------------------------------------ dữ liệu
def load_pubmed():
    lines = (ROOT_DIR / "Pubmed-Diabetes.NODE.paper.tab").read_text().splitlines()
    vocab = [c.split(":")[1] for c in lines[1].split("\t")[1:-1]]  # numeric:w-rat:0.0 → w-rat
    widx = {w: i for i, w in enumerate(vocab)}
    ids, labels, X = [], [], np.zeros((len(lines) - 2, len(vocab)), np.float32)
    for r, line in enumerate(lines[2:]):
        parts = line.split("\t")
        ids.append(parts[0])
        labels.append(int(parts[1].split("=")[1]) - 1)
        for kv in parts[2:-1]:
            k, v = kv.split("=")
            X[r, widx[k]] = float(v)
    nid = {p: i for i, p in enumerate(ids)}
    edges = set()
    for line in (ROOT_DIR / "Pubmed-Diabetes.DIRECTED.cites.tab").read_text().splitlines()[2:]:
        parts = line.split("\t")
        a, b = nid[parts[1].split(":")[1]], nid[parts[3].split(":")[1]]
        if a != b:
            edges.add((min(a, b), max(a, b)))  # đồ thị vô hướng, bỏ cạnh lặp và khuyên
    E = np.array(sorted(edges)).T
    return ids, X, np.array(labels), E, [w[2:] for w in vocab]


def build_graph(n, E):
    src = np.concatenate([E[0], E[1]])
    dst = np.concatenate([E[1], E[0]])
    deg = np.bincount(src, minlength=n)
    # Â = D̃^{-1/2} (A + I) D̃^{-1/2} cho GCN
    s_loop, d_loop = np.concatenate([src, np.arange(n)]), np.concatenate([dst, np.arange(n)])
    dt = deg + 1.0
    w = 1 / np.sqrt(dt[s_loop] * dt[d_loop])
    A_gcn = torch.sparse_coo_tensor(np.stack([d_loop, s_loop]), torch.tensor(w, dtype=torch.float32), (n, n)).coalesce()
    # D^{-1} A cho GraphSAGE-mean (trung bình láng giềng, không có khuyên)
    w_mean = 1 / np.maximum(deg[dst], 1)
    A_mean = torch.sparse_coo_tensor(np.stack([dst, src]), torch.tensor(w_mean, dtype=torch.float32), (n, n)).coalesce()
    edge_index = torch.tensor(np.stack([s_loop, d_loop]), dtype=torch.long)  # GAT: có khuyên
    return A_gcn, A_mean, edge_index, deg


# ------------------------------------------------------------------ mô hình
class MLP(nn.Module):
    def __init__(self, d, h, c, p=0.5):
        super().__init__()
        self.l1, self.l2, self.p = nn.Linear(d, h), nn.Linear(h, c), p

    def forward(self, x, g):
        x = F.dropout(x, self.p, self.training)
        self.hidden = F.relu(self.l1(x))
        return self.l2(F.dropout(self.hidden, self.p, self.training))


class GCN(nn.Module):
    """Kipf & Welling (2017): H^{l+1} = σ(Â H^l W^l). Cho phép nhiều tầng để khảo sát over-smoothing."""

    def __init__(self, d, h, c, layers=2, p=0.5):
        super().__init__()
        dims = [d] + [h] * (layers - 1) + [c]
        self.lins = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:]))
        self.p = p

    def forward(self, x, g):
        for i, lin in enumerate(self.lins):
            x = F.dropout(x, self.p, self.training)
            x = torch.sparse.mm(g["A_gcn"], lin(x))       # biến đổi đặc trưng rồi gộp có chuẩn hóa bậc
            if i < len(self.lins) - 1:
                x = F.relu(x)
                self.hidden = x
        return x


class SAGE(nn.Module):
    """Hamilton et al. (2017), bộ gộp mean: h_v' = σ(W_self·h_v + W_neigh·mean_{u∈N(v)} h_u)."""

    def __init__(self, d, h, c, p=0.5):
        super().__init__()
        self.s1, self.n1 = nn.Linear(d, h), nn.Linear(d, h, bias=False)
        self.s2, self.n2 = nn.Linear(h, c), nn.Linear(h, c, bias=False)
        self.p = p

    def forward(self, x, g):
        x = F.dropout(x, self.p, self.training)
        x = F.relu(self.s1(x) + self.n1(torch.sparse.mm(g["A_mean"], x)))
        self.hidden = x
        x = F.dropout(x, self.p, self.training)
        return self.s2(x) + self.n2(torch.sparse.mm(g["A_mean"], x))


class GATLayer(nn.Module):
    """Veličković et al. (2018): α_vu = softmax_u(LeakyReLU(aᵀ[W h_v ‖ W h_u])), h_v' = ‖_k Σ_u α^k_vu W^k h_u."""

    def __init__(self, d, h, heads, concat=True, p=0.6):
        super().__init__()
        self.W = nn.Linear(d, h * heads, bias=False)
        self.a_src = nn.Parameter(torch.empty(1, heads, h))
        self.a_dst = nn.Parameter(torch.empty(1, heads, h))
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)
        self.heads, self.h, self.concat, self.p = heads, h, concat, p
        self.bias = nn.Parameter(torch.zeros(h * heads if concat else h))

    def forward(self, x, edge_index):
        n = x.shape[0]
        src, dst = edge_index
        z = self.W(x).view(n, self.heads, self.h)
        e = F.leaky_relu((z * self.a_src).sum(-1)[src] + (z * self.a_dst).sum(-1)[dst], 0.2)  # [E, heads]
        # softmax theo từng nút đích: trừ max để ổn định số, rồi chuẩn hóa bằng tổng exp của láng giềng
        e_max = torch.full((n, self.heads), -1e9, device=x.device).index_reduce_(0, dst, e, "amax", include_self=True)
        ex = torch.exp(e - e_max[dst])
        denom = torch.zeros(n, self.heads, device=x.device).index_add_(0, dst, ex)
        alpha = ex / (denom[dst] + 1e-16)
        self.alpha = alpha.detach()
        alpha = F.dropout(alpha, self.p, self.training)
        out = torch.zeros(n, self.heads, self.h, device=x.device).index_add_(0, dst, alpha[..., None] * z[src])
        out = out.reshape(n, -1) if self.concat else out.mean(1)
        return out + self.bias


class GAT(nn.Module):
    def __init__(self, d, h, c, heads=8, p=0.6):
        super().__init__()
        self.g1 = GATLayer(d, h, heads, True, p)
        self.g2 = GATLayer(h * heads, c, 8, False, p)   # Pubmed: 8 đầu ở tầng ra, lấy trung bình
        self.p = p

    def forward(self, x, g):
        x = F.dropout(x, self.p, self.training)
        x = F.elu(self.g1(x, g["edge_index"]))
        self.hidden = x
        return self.g2(F.dropout(x, self.p, self.training), g["edge_index"])


def label_propagation(g, y, train, alpha=0.9, iters=50):
    """Zhou et al. (2004): F ← α·Â·F + (1−α)·Y — chỉ dùng cấu trúc đồ thị, không dùng đặc trưng."""
    Y = torch.zeros(len(y), 3, device=y.device)
    Y[train, y[train]] = 1
    Fm = Y.clone()
    for _ in range(iters):
        Fm = alpha * torch.sparse.mm(g["A_gcn"], Fm) + (1 - alpha) * Y
    return Fm


FACTORIES = {
    "logreg": ("Hồi quy logistic", lambda d: nn.Sequential(nn.Dropout(0.2), nn.Linear(d, 3)), 0.2, 5e-4),
    "mlp": ("MLP", lambda d: MLP(d, 64, 3), 0.01, 5e-4),
    "gcn": ("GCN", lambda d: GCN(d, 64, 3), 0.01, 5e-4),
    "sage": ("GraphSAGE-mean", lambda d: SAGE(d, 64, 3), 0.01, 5e-4),
    "gat": ("GAT (8 đầu)", lambda d: GAT(d, 8, 3), 0.01, 1e-3),
}


class Wrap(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x, g):
        return self.m(x)


def make(key, d):
    m = FACTORIES[key][1](d)
    return Wrap(m) if key == "logreg" else m


def fit(model, X, y, g, tr, va, lr, wd, epochs=300, patience=50):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    best, hist, wait = (-1, None, 0), [], 0
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        out = model(X, g)
        loss = F.cross_entropy(out[tr], y[tr])
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            pred = model(X, g).argmax(1)
        va_acc = (pred[va] == y[va]).float().mean().item()
        hist.append({"epoch": ep + 1, "loss": r4(loss.item()), "train_acc": r4((pred[tr] == y[tr]).float().mean()),
                     "val_acc": r4(va_acc)})
        if va_acc > best[0]:
            best, wait = (va_acc, copy.deepcopy(model.state_dict()), ep + 1), 0
        else:
            wait += 1
            if wait > patience:
                break
    model.load_state_dict(best[1])
    return model, hist


def evaluate(pred, y, idx):
    return {"acc": (pred[idx] == y[idx]).float().mean().item(),
            "macro_f1": f1_score(y[idx].cpu().numpy(), pred[idx].cpu().numpy(), average="macro")}


def planetoid_split(y, per_class, seed, n_val=500, n_test=1000):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(y))
    train = np.concatenate([perm[y[perm] == c][:per_class] for c in range(3)])
    rest = np.setdiff1d(perm, train, assume_unique=True)
    rest = rest[rng.permutation(len(rest))]
    return [torch.tensor(a, device=DEVICE) for a in (train, rest[:n_val], rest[n_val:n_val + n_test])]


def full_split(n, seed):
    perm = np.random.default_rng(seed).permutation(n)
    a, b = int(0.6 * n), int(0.8 * n)
    return [torch.tensor(perm[:a], device=DEVICE), torch.tensor(perm[a:b], device=DEVICE), torch.tensor(perm[b:], device=DEVICE)]


def run_protocol(X, y, g, splits, keys, epochs=300):
    out, timing = {k: [] for k in keys + ["lp"]}, {}
    for tr, va, te in splits:
        for k in keys:
            seed_everything(int(tr[0]))
            t0 = time.perf_counter()
            model, _ = fit(make(k, X.shape[1]).to(DEVICE), X, y, g, tr, va, FACTORIES[k][2], FACTORIES[k][3], epochs)
            timing[k] = time.perf_counter() - t0
            with torch.no_grad():
                out[k].append(evaluate(model(X, g).argmax(1), y, te))
        out["lp"].append(evaluate(label_propagation(g, y, tr).argmax(1), y, te))
    return {k: {m: {"mean": r4(np.mean([r[m] for r in v])), "std": r4(np.std([r[m] for r in v]))}
                for m in ["acc", "macro_f1"]} for k, v in out.items()}, {k: round(v, 2) for k, v in timing.items()}


# ------------------------------------------------------------------ hình
def plot_data(y, deg, X, vocab):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3))
    cnt = np.bincount(y)
    axes[0].bar(CLASSES, cnt, color=[PALETTE[2], PALETTE[1], ACCENT])
    for i, v in enumerate(cnt):
        axes[0].text(i, v + 150, f"{v:,}\n({v / len(y):.1%})", ha="center", fontsize=8)
    axes[0].set(title="Phân bố nhãn", ylim=(0, cnt.max() * 1.25))
    axes[0].tick_params(axis="x", labelsize=8)
    c = Counter(deg)
    ks = np.array(sorted(c))
    axes[1].loglog(ks[ks > 0], [c[k] for k in ks[ks > 0]], "o", ms=3, color=ACCENT)
    axes[1].set(title=f"Phân bố bậc (TB {deg.mean():.2f}, max {deg.max()})", xlabel="bậc", ylabel="số nút")
    nz = (X > 0).sum(1)
    axes[2].hist(nz, bins=40, color=GREY)
    axes[2].set(title=f"Số từ khác 0 mỗi bài (TB {nz.mean():.1f}/500)", xlabel="số từ")
    fig.tight_layout()
    save_fig(fig, "ch6_data")


def plot_label_rate(rows):
    fig, ax = plt.subplots(figsize=(6, 2.9))
    rates = [r["per_class"] for r in rows]
    for i, (k, name) in enumerate([("mlp", "MLP"), ("lp", "Lan truyền nhãn"), ("gcn", "GCN"), ("gat", "GAT")]):
        ax.errorbar(rates, [r[k]["mean"] for r in rows], yerr=[r[k]["std"] for r in rows], marker="o", capsize=2,
                    color=[GREY, PALETTE[1], ACCENT, PALETTE[4]][i], label=name)
    ax.set(xscale="log", xticks=rates, xticklabels=rates, xlabel="số nhãn huấn luyện mỗi lớp",
           ylabel="accuracy (test)", title="Đồ thị bù cho sự khan hiếm nhãn")
    ax.legend(fontsize=8)
    save_fig(fig, "ch6_label_rate")


def plot_depth(rows):
    fig, ax = plt.subplots(figsize=(6, 2.8))
    L = [r["layers"] for r in rows]
    ax.plot(L, [r["train_acc"] for r in rows], "o-", color=GREY, label="train")
    ax.plot(L, [r["test_acc"] for r in rows], "o-", color=ACCENT, label="test")
    ax2 = ax.twinx()
    ax2.plot(L, [r["mad"] for r in rows], "s--", color=PALETTE[3], label="MAD (độ khác biệt biểu diễn)")
    ax2.set_ylabel("MAD", color=PALETTE[3])
    ax2.grid(False)
    ax.set(title="Over-smoothing: GCN càng sâu, biểu diễn các nút càng giống nhau", xlabel="số tầng GCN",
           ylabel="accuracy", xticks=L)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7.5, loc="lower left")
    save_fig(fig, "ch6_depth")


def plot_tsne(X, y, gcn_hidden):
    idx = np.random.default_rng(0).permutation(len(y))[:3000]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (Z, title) in zip(axes, [(X[idx], "Đặc trưng TF-IDF gốc (500 chiều)"),
                                     (gcn_hidden[idx], "Biểu diễn ẩn của GCN (64 chiều)")]):
        emb = TSNE(2, init="pca", random_state=0, perplexity=30).fit_transform(Z)
        for c in range(3):
            m = y[idx] == c
            ax.scatter(emb[m, 0], emb[m, 1], s=3, alpha=0.6, color=[PALETTE[2], PALETTE[1], ACCENT][c],
                       label=CLASSES[c])
        ax.set(title=f"t-SNE: {title}", xticks=[], yticks=[])
    axes[0].legend(markerscale=4, fontsize=8)
    fig.tight_layout()
    save_fig(fig, "ch6_tsne")


def plot_history(histories):
    fig, axes = plt.subplots(1, 2, figsize=(10, 2.8))
    for i, (k, h) in enumerate(histories.items()):
        axes[0].plot([r["loss"] for r in h], color=PALETTE[i], label=FACTORIES[k][0])
        axes[1].plot([r["val_acc"] for r in h], color=PALETTE[i])
    axes[0].set(title="Loss huấn luyện (20 nhãn/lớp)", xlabel="epoch", yscale="log")
    axes[1].set(title="Accuracy validation", xlabel="epoch")
    axes[0].legend(fontsize=7.5)
    fig.tight_layout()
    save_fig(fig, "ch6_history")


def mad(h):
    """Mean Average Distance (Chen et al. 2020): 1 − cos trung bình giữa biểu diễn các cặp nút."""
    h = F.normalize(h[torch.randperm(len(h), device=h.device)[:2000]], dim=1)
    return float((1 - h @ h.T).mean())


def main():
    seed_everything()
    ids, Xn, yn, E, vocab = load_pubmed()
    n = len(yn)
    A_gcn, A_mean, edge_index, deg = build_graph(n, E)
    g = {"A_gcn": A_gcn.to(DEVICE), "A_mean": A_mean.to(DEVICE), "edge_index": edge_index.to(DEVICE)}
    X, y = torch.tensor(Xn, device=DEVICE), torch.tensor(yn, device=DEVICE)
    homophily = float((yn[E[0]] == yn[E[1]]).mean())
    print(f"Pubmed: {n:,} nút, {E.shape[1]:,} cạnh vô hướng, bậc TB {deg.mean():.2f}, homophily {homophily:.3f}")

    keys = ["logreg", "mlp", "gcn", "sage", "gat"]
    semi, t_semi = run_protocol(X, y, g, [planetoid_split(yn, 20, s) for s in range(10)], keys)
    print("Bán giám sát:", {k: v["acc"] for k, v in semi.items()})
    full, t_full = run_protocol(X, y, g, [full_split(n, s) for s in range(3)], keys, epochs=400)
    print("Có giám sát:", {k: v["acc"] for k, v in full.items()})

    label_rate = []
    for pc in [1, 3, 5, 10, 20, 50, 100]:
        r, _ = run_protocol(X, y, g, [planetoid_split(yn, pc, s) for s in range(5)], ["mlp", "gcn", "gat"])
        label_rate.append({"per_class": pc, **{k: r[k]["acc"] for k in r}})
        print("  nhãn/lớp", pc, {k: r[k]["acc"]["mean"] for k in r})

    depth = []
    tr, va, te = planetoid_split(yn, 20, 0)
    for L in [1, 2, 3, 4, 6, 8, 16]:
        seed_everything(0)
        m, _ = fit(GCN(500, 64, 3, layers=L).to(DEVICE), X, y, g, tr, va, 0.01, 5e-4)
        m.eval()
        with torch.no_grad():
            out = m(X, g)
        pred = out.argmax(1)
        depth.append({"layers": L, "train_acc": r4((pred[tr] == y[tr]).float().mean()),
                      "test_acc": r4((pred[te] == y[te]).float().mean()), "mad": r4(mad(out))})
    print("Độ sâu:", depth)

    histories, models_ = {}, {}
    for k in ["mlp", "gcn", "sage", "gat"]:
        seed_everything(0)
        models_[k], histories[k] = fit(make(k, 500).to(DEVICE), X, y, g, tr, va, FACTORIES[k][2], FACTORIES[k][3])
    models_["gcn"].eval()
    with torch.no_grad():
        models_["gcn"](X, g)
    plot_data(yn, deg, Xn, vocab)
    plot_history(histories)
    plot_label_rate(label_rate)
    plot_depth(depth)
    plot_tsne(Xn, yn, models_["gcn"].hidden.cpu().numpy())

    # mô hình deploy: GAT huấn luyện trên tập 60% (giao thức có giám sát) để gán nhãn tài liệu mới vào KB
    tr_f, va_f, te_f = full_split(n, 0)
    seed_everything(0)
    deploy, _ = fit(make("gat", 500).to(DEVICE), X, y, g, tr_f, va_f, FACTORIES["gat"][2], FACTORIES["gat"][3], 400)
    deploy.eval()
    with torch.no_grad():
        prob = F.softmax(deploy(X, g), 1)
    torch.save({"state": {k: v.cpu() for k, v in deploy.state_dict().items()}, "test_idx": te_f.cpu()},
               MODELS / "ch6_gat.pt")
    np.savez_compressed(MODELS / "ch6_pubmed_graph.npz", ids=np.array(ids), y=yn, E=E,
                        X=Xn.astype(np.float16), vocab=np.array(vocab), prob=prob.cpu().numpy().astype(np.float32))

    top_words = {CLASSES[c]: [vocab[i] for i in np.argsort(-Xn[yn == c].mean(0))[:10]] for c in range(3)}
    params = {k: sum(p.numel() for p in make(k, 500).parameters()) for k in keys}
    save_results("ch6", {
        "dataset": {"name": "Pubmed-Diabetes (LINQS)", "nodes": n, "edges": int(E.shape[1]),
                    "features": int(Xn.shape[1]), "classes": np.bincount(yn).tolist(),
                    "avg_degree": r4(deg.mean()), "max_degree": int(deg.max()), "isolated": int((deg == 0).sum()),
                    "homophily": r4(homophily), "density": r4(2 * E.shape[1] / (n * (n - 1)) * 1e4),
                    "nnz_per_doc": r4((Xn > 0).sum(1).mean()), "top_words": top_words,
                    "sample": [{"id": ids[i], "label": CLASSES[yn[i]], "degree": int(deg[i]),
                                "top_terms": [vocab[j] for j in np.argsort(-Xn[i])[:6]]} for i in range(5)]},
        "semi_supervised": semi, "full_supervised": full, "time_s": {"semi": t_semi, "full": t_full},
        "label_rate": label_rate, "depth": depth, "params": params,
        "deploy_test_acc": r4((prob.argmax(1)[te_f] == y[te_f]).float().mean()),
    })


if __name__ == "__main__":
    main()
