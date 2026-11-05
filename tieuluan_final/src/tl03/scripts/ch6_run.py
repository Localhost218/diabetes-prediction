"""Mọi thực nghiệm Chương 6 trên PubMed-Diabetes (mạch chính của tiểu luận final), Cora và CiteSeer; ghi outputs/metrics/ch6_*.json và models/ch6/*.pt."""
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tl3.ch6_threads import torch_threads  # noqa: E402
from tl3.common import LOGS, load_json, model_dir, save_json  # noqa: E402
from tl3 import gpu  # noqa: E402
from tl3.ch6_gnn import (CLASSES, GAE, gcn_norm_adj, homophily, load_planetoid, mad,  # noqa: E402
                         train_node)

DEV = os.environ.get("CH6_DEVICE", "cpu")   # GPU dùng chung bị các làn khác chiếm; Cora/CiteSeer đủ nhỏ cho CPU
SEEDS = [0, 1, 2, 3, 4]
KINDS = ["mlp", "gcn", "sage", "gat"]
CKPT = LOGS / "ch6_ckpt"
CKPT.mkdir(parents=True, exist_ok=True)


def dataset_stats(name, data):
    """Thống kê mô tả: số nút, cạnh vô hướng, đặc trưng, lớp, bậc, homophily, phân bố lớp, độ thưa đặc trưng."""
    ei = data.edge_index
    deg = torch.bincount(ei[1], minlength=data.num_nodes).float()
    raw_nnz = (data.x > 0).sum(1).float()
    return {"nodes": data.num_nodes, "edges": ei.size(1) // 2, "features": data.num_features,
            "classes": int(data.y.max()) + 1, "train": int(data.train_mask.sum()), "val": int(data.val_mask.sum()),
            "test": int(data.test_mask.sum()), "deg_mean": deg.mean().item(), "deg_max": int(deg.max()),
            "deg_median": deg.median().item(), "isolated": int((deg == 0).sum()), "homophily": homophily(ei, data.y),
            "feat_nnz_mean": raw_nnz.mean().item(), "density": ei.size(1) / (data.num_nodes * (data.num_nodes - 1)),
            "class_counts": torch.bincount(data.y).tolist(), "class_names": CLASSES[name],
            "deg_hist": torch.bincount(deg.long().clamp(max=30)).tolist()}


def logreg_baseline(data, seed):
    """Hồi quy logistic trên vector từ (bag-of-words) — không dùng cạnh."""
    from sklearn.linear_model import LogisticRegression
    x = data.x.numpy()
    clf = LogisticRegression(max_iter=2000, C=1.0, random_state=seed)
    t0 = time.perf_counter()
    clf.fit(x[data.train_mask], data.y[data.train_mask])
    return {"test": clf.score(x[data.test_mask], data.y[data.test_mask]),
            "val": clf.score(x[data.val_mask], data.y[data.val_mask]), "seconds": time.perf_counter() - t0,
            "params": int(clf.coef_.size + clf.intercept_.size)}


def node_classification(datasets):
    """5 mô hình × 5 seed × 2 bộ dữ liệu; lưu trung bình ± độ lệch, đường học của seed 0, mô hình Cora seed 0."""
    out, keep = {}, {}
    for name, data in datasets.items():
        out[name] = {}
        rs = [logreg_baseline(data, s) for s in SEEDS]
        out[name]["logreg"] = summarize(rs)
        for kind in KINDS:
            ck = CKPT / f"{name}_{kind}.pt"
            if not ck.exists():
                # mỗi (bộ dữ liệu, mô hình) huấn luyện trong tiến trình con riêng: trên máy dùng chung, một tiến trình
                # chạy lâu từng chết do lỗi truy cập bộ nhớ native; tách ra thì lỗi không kéo theo các phần đã xong
                subprocess.run([sys.executable, "-u", __file__, "--train", name, kind], check=True)
            out[name][kind], keep[(name, kind)] = torch.load(ck, weights_only=False)
            print(name, kind, out[name][kind]["test_mean"], flush=True)
    return out, keep


def train_one(name, kind):
    """5 hạt giống cho một (bộ dữ liệu, mô hình); lưu tóm tắt và lần chạy seed 0 vào outputs/logs/ch6_ckpt."""
    torch_threads(3)
    data = load_planetoid(name)
    rs = [train_node(kind, data, seed=s, device=DEV) for s in SEEDS]
    res = summarize(rs)
    res["history"] = rs[0]["history"]
    torch.save((res, rs[0]), CKPT / f"{name}_{kind}.pt")


def summarize(rs):
    t = np.array([r["test"] for r in rs])
    return {"test_mean": float(t.mean()), "test_std": float(t.std()), "test_runs": t.tolist(),
            "val_mean": float(np.mean([r["val"] for r in rs])), "seconds": float(np.mean([r["seconds"] for r in rs])),
            "params": int(rs[0]["params"]), "epoch": float(np.mean([r.get("epoch", 0) for r in rs]))}


def pyg_crosscheck(data):
    """Đối chiếu lớp tự cài với GCNConv/GATConv của PyTorch Geometric cùng siêu tham số (3 seed)."""
    from torch_geometric.nn import GATConv, GCNConv

    class PygNet(torch.nn.Module):
        def __init__(self, kind):
            super().__init__()
            if kind == "gcn":
                self.c1, self.c2 = GCNConv(data.num_features, 64), GCNConv(64, 7)
            else:
                self.c1, self.c2 = GATConv(data.num_features, 8, heads=8, dropout=0.6), GATConv(64, 7, dropout=0.6)
            self.p, self.kind = (0.5 if kind == "gcn" else 0.6), kind

        def forward(self, x, ei):
            x = F.dropout(x, self.p, self.training)
            x = self.c1(x, ei)
            x = F.elu(x) if self.kind == "gat" else F.relu(x)
            return self.c2(F.dropout(x, self.p, self.training), ei)

    res = {}
    x, y, ei = data.x.to(DEV), data.y.to(DEV), data.edge_index.to(DEV)
    for kind in ("gcn", "gat"):
        accs = []
        for s in SEEDS[:3]:
            torch.manual_seed(s)
            m = PygNet(kind).to(DEV)
            opt = torch.optim.Adam(m.parameters(), lr=0.01, weight_decay=5e-4)
            best_v, best_t = -1, 0
            for _ in range(300):
                m.train()
                opt.zero_grad()
                F.cross_entropy(m(x, ei)[data.train_mask], y[data.train_mask]).backward()
                opt.step()
                m.eval()
                with torch.no_grad():
                    p = m(x, ei).argmax(1)
                v = (p[data.val_mask] == y[data.val_mask]).float().mean().item()
                if v > best_v:
                    best_v, best_t = v, (p[data.test_mask] == y[data.test_mask]).float().mean().item()
            accs.append(best_t)
        res[kind] = {"test_mean": float(np.mean(accs)), "test_std": float(np.std(accs))}
    return res


def depth_study(data):
    """GCN 1–8 lớp, có/không nối tắt: accuracy test và MAD của biểu diễn ẩn cuối (đo over-smoothing)."""
    rows = []
    for layers in range(1, 9):
        for residual in (False, True):
            if residual and layers < 3:
                continue
            accs, mads, mads_e = [], [], []
            for s in SEEDS[:3]:
                r = train_node("gcn", data, seed=s, layers=layers, hidden=64, residual=residual, device=DEV)
                accs.append(r["test"])
                h = r["hidden"] if r["hidden"] is not None else r["logits"]
                mads.append(mad(h))
                mads_e.append(mad(h, data.edge_index))
            rows.append({"layers": layers, "residual": residual, "test_mean": float(np.mean(accs)),
                         "test_std": float(np.std(accs)), "mad": float(np.mean(mads)),
                         "mad_neighbors": float(np.mean(mads_e))})
            print("depth", layers, residual, rows[-1]["test_mean"], rows[-1]["mad"], flush=True)
    # làm trơn thuần (không học): Â^k X — thấy over-smoothing tách khỏi việc huấn luyện
    adj = gcn_norm_adj(data.edge_index, data.num_nodes)
    x = data.x.clone()
    smooth = []
    for k in range(0, 65):
        if k in (0, 1, 2, 4, 8, 16, 32, 64):
            smooth.append({"k": k, "mad": mad(x)})
        x = torch.sparse.mm(adj, x)
    return {"rows": rows, "propagation": smooth}


def edge_perturbation(data):
    """Giá trị của cấu trúc: bỏ ngẫu nhiên p% cạnh, hoặc xáo trộn đầu cạnh (giữ số cạnh) rồi huấn luyện lại GCN."""
    rows = []
    src, dst = data.edge_index
    und = src < dst
    pairs = torch.stack([src[und], dst[und]])
    for mode in ("drop", "shuffle"):
        for p in (0.0, 0.25, 0.5, 0.75, 1.0):
            accs = []
            for s in SEEDS[:3]:
                g = torch.Generator().manual_seed(s)
                e = pairs.clone()
                m = int(p * e.size(1))
                idx = torch.randperm(e.size(1), generator=g)[:m]
                if mode == "drop":
                    keep = torch.ones(e.size(1), dtype=torch.bool)
                    keep[idx] = False
                    e = e[:, keep]
                else:
                    e[1, idx] = torch.randint(0, data.num_nodes, (m,), generator=g)
                d2 = data.clone()
                d2.edge_index = torch.cat([e, e.flip(0)], 1)
                accs.append(train_node("gcn", d2, seed=s, device=DEV)["test"])
            rows.append({"mode": mode, "p": p, "test_mean": float(np.mean(accs)), "test_std": float(np.std(accs)),
                         "homophily": homophily(d2.edge_index, data.y)})
            print("perturb", mode, p, rows[-1]["test_mean"], flush=True)
    return rows


def tsne_points(data, keep):
    """Toạ độ t-SNE của đặc trưng thô và của biểu diễn ẩn GCN (Cora, seed 0) để vẽ hình."""
    from sklearn.manifold import TSNE
    from sklearn.metrics import silhouette_score
    out = {}
    for tag, mat in (("raw", data.x.numpy()), ("gcn", keep[("Cora", "gcn")]["hidden"].numpy()),
                     ("gat", keep[("Cora", "gat")]["hidden"].numpy())):
        z = TSNE(2, random_state=0, init="pca", perplexity=30).fit_transform(mat)
        out[tag] = {"silhouette": float(silhouette_score(mat, data.y.numpy(), metric="cosine")),
                    "series": np.round(z, 3).tolist()}
    out["labels"] = {"series": data.y.tolist()}
    return out


def gat_attention(data, keep):
    """Hệ số attention (trung bình 8 đầu, lớp 1) của GAT Cora cho một nút bậc vừa phải."""
    model = keep[("Cora", "gat")]["model"].cpu()
    with torch.no_grad():
        model(data.x, data.edge_index)        # mô hình phân loại nút đã huấn luyện trên CPU
    ei, alpha = model.layers[0].last_alpha
    ei, alpha = ei.cpu(), alpha.cpu()
    deg = torch.bincount(data.edge_index[1], minlength=data.num_nodes)
    cand = torch.nonzero((deg >= 6) & (deg <= 9)).flatten()
    node = int(cand[0])
    mask = ei[1] == node
    nbrs = ei[0, mask].tolist()
    a = alpha[mask]
    pred = keep[("Cora", "gat")]["logits"].argmax(1)
    return {"node": node, "label": int(data.y[node]), "pred": int(pred[node]),
            "neighbors": [{"id": int(u), "label": int(data.y[u]), "alpha_mean": float(a[i].mean()),
                           "alpha_heads": a[i].tolist()} for i, u in enumerate(nbrs)],
            "entropy_mean": float(-(alpha.clamp(min=1e-12) * alpha.clamp(min=1e-12).log()).sum(0).mean())}


def link_prediction(data, epochs=300):
    """GAE trên Cora: tách 85/5/10 % cạnh (RandomLinkSplit), mẫu âm ngẫu nhiên; đo AUC và AP trên cạnh test."""
    from sklearn.metrics import average_precision_score, roc_auc_score
    from torch_geometric.transforms import RandomLinkSplit
    from torch_geometric.utils import negative_sampling
    res = []
    hist = []
    for s in SEEDS[:3]:
        torch.manual_seed(s)
        split = RandomLinkSplit(num_val=0.05, num_test=0.10, is_undirected=True,
                                add_negative_train_samples=False)
        tr, va, te = split(data.clone())
        adj = gcn_norm_adj(tr.edge_index.to(DEV), data.num_nodes)
        model = GAE(data.num_features).to(DEV)
        opt = torch.optim.Adam(model.parameters(), lr=0.01)
        x = data.x.to(DEV)
        pos = tr.edge_label_index.to(DEV)
        best = (-1, None)
        for ep in range(epochs):
            model.train()
            opt.zero_grad()
            z = model.encode(x, adj)
            neg = negative_sampling(tr.edge_index.to(DEV), data.num_nodes, pos.size(1))
            logit = torch.cat([model.decode(z, pos), model.decode(z, neg)])
            lab = torch.cat([torch.ones(pos.size(1)), torch.zeros(neg.size(1))]).to(DEV)
            loss = F.binary_cross_entropy_with_logits(logit, lab)
            loss.backward()
            opt.step()
            if ep % 10 == 9:
                v = evaluate_links(model, x, adj, va, roc_auc_score, average_precision_score)
                if s == 0:
                    hist.append({"epoch": ep + 1, "loss": loss.item(), "val_auc": v[0]})
                if v[0] > best[0]:
                    best = (v[0], evaluate_links(model, x, adj, te, roc_auc_score, average_precision_score))
        res.append({"val_auc": best[0], "test_auc": best[1][0], "test_ap": best[1][1]})
        print("link", res[-1], flush=True)
    return {"test_auc_mean": float(np.mean([r["test_auc"] for r in res])),
            "test_auc_std": float(np.std([r["test_auc"] for r in res])),
            "test_ap_mean": float(np.mean([r["test_ap"] for r in res])),
            "test_ap_std": float(np.std([r["test_ap"] for r in res])),
            "train_edges": int(tr.edge_index.size(1) // 2), "val_edges": int(va.edge_label.sum()),
            "test_edges": int(te.edge_label.sum()), "history": hist}


def evaluate_links(model, x, adj, split, auc, ap):
    model.eval()
    with torch.no_grad():
        z = model.encode(x, adj)
        p = torch.sigmoid(model.decode(z, split.edge_label_index.to(DEV))).cpu().numpy()
    y = split.edge_label.numpy()
    return float(auc(y, p)), float(ap(y, p))


def confusion(data, keep, name="Cora", kind="gcn"):
    from sklearn.metrics import confusion_matrix, f1_score
    pred = keep[(name, kind)]["logits"].argmax(1).numpy()
    m = data.test_mask.numpy()
    y = data.y.numpy()
    return {"matrix": confusion_matrix(y[m], pred[m]).tolist(), "macro_f1": float(f1_score(y[m], pred[m], average="macro")),
            "per_class_recall": (confusion_matrix(y[m], pred[m]).diagonal()
                                 / np.maximum(confusion_matrix(y[m], pred[m]).sum(1), 1)).tolist()}


def degree_accuracy(data, keep):
    """Độ chính xác theo nhóm bậc của nút test: GCN so với MLP — đồ thị giúp nút nào nhiều nhất?"""
    deg = torch.bincount(data.edge_index[1], minlength=data.num_nodes).numpy()
    m = data.test_mask.numpy()
    y = data.y.numpy()
    bins = [(1, 1), (2, 2), (3, 4), (5, 8), (9, 1000)]
    out = []
    for lo, hi in bins:
        sel = m & (deg >= lo) & (deg <= hi)
        row = {"bin": f"{lo}" if lo == hi else (f"{lo}-{hi}" if hi < 1000 else f">={lo}"), "n": int(sel.sum())}
        for kind in ("mlp", "gcn", "gat"):
            pred = keep[("Cora", kind)]["logits"].argmax(1).numpy()
            row[kind] = float((pred[sel] == y[sel]).mean()) if sel.any() else None
        out.append(row)
    return out


def save_serving_models(datasets, keep):
    """Lưu trọng số Cora và PubMed seed 0 cho Web App (serve_ch6.py)."""
    d = model_dir("ch6")
    for ds, prefix in (("Cora", "cora"), ("PubMed", "pubmed")):
        for kind in ("gcn", "sage", "gat", "mlp"):
            if (ds, kind) in keep:
                r = keep[(ds, kind)]
                torch.save({"state": r["model"].cpu().state_dict(), "kind": kind, "hidden": 8 if kind == "gat" else 64,
                            "test": r["test"]}, d / f"{prefix}_{kind}.pt")


def done(name):
    try:
        return bool(load_json(name))
    except FileNotFoundError:
        return False


def main():
    global DEV
    import faulthandler
    faulthandler.enable()   # tiến trình chết do lỗi native thì vẫn in được ngăn xếp vào log
    torch_threads(3)
    if DEV == "cuda":
        gpu.cap(0.15)
    datasets = {n: load_planetoid(n) for n in ("PubMed", "Cora", "CiteSeer")}
    save_json("ch6_data", {n: dataset_stats(n, d) for n, d in datasets.items()})
    node, keep = node_classification(datasets)
    save_json("ch6_node", node)
    save_serving_models(datasets, keep)
    # Bảng thời gian/tham số ở trên đo trên CPU cho mọi mô hình; các phân tích sau (hàng trăm lần huấn luyện, không so
    # thời gian) chạy GPU với phần VRAM giới hạn, vì CPU của máy dùng chung đang bị các làn khác chiếm gần hết.
    DEV = os.environ.get("CH6_ANALYSIS_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    if DEV == "cuda":
        gpu.cap(0.15)
    cora = datasets["Cora"]
    stages = {
        "ch6_extra": lambda: {"pyg": pyg_crosscheck(cora), "confusion": confusion(cora, keep),
                              "gat_attention": gat_attention(cora, keep),
                              "degree_acc": {"rows": degree_accuracy(cora, keep)}, "tsne": tsne_points(cora, keep)},
        "ch6_depth": lambda: depth_study(cora),
        "ch6_perturb": lambda: {"rows": edge_perturbation(cora)},
        "ch6_link": lambda: link_prediction(cora),
    }
    for name, run in stages.items():
        if not done(name):
            save_json(name, run())
            print("đã ghi", name, flush=True)
    print("XONG", flush=True)


if __name__ == "__main__":
    if sys.argv[1:2] == ["--train"]:
        train_one(*sys.argv[2:4])
    else:
        main()
