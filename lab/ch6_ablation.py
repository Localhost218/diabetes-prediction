"""
Chương 6 — thí nghiệm bổ sung trên Pubmed-Diabetes (dùng lại mã của ch6_gnn.py):
  (a) Ablation các lựa chọn thiết kế của GCN: khuyên, cách chuẩn hóa ma trận kề, dropout, kích thước ẩn
  (b) Học quy nạp (inductive): bỏ toàn bộ nút kiểm tra (và cạnh của chúng) khỏi đồ thị khi huấn luyện,
      chỉ thêm lại lúc suy luận — so sánh với học chuyển dẫn (transductive) thông thường.

    .venv-gpu/bin/python lab/ch6_ablation.py
"""

import numpy as np
import torch

from ch6_gnn import (DEVICE, FACTORIES, GCN, build_graph, evaluate, fit, full_split, load_pubmed, make,
                     planetoid_split)
from common import r4, save_results, seed_everything


def adj(n, E, self_loop=True, norm="sym"):
    src = np.concatenate([E[0], E[1]])
    dst = np.concatenate([E[1], E[0]])
    if self_loop:
        src, dst = np.concatenate([src, np.arange(n)]), np.concatenate([dst, np.arange(n)])
    deg = np.bincount(dst, minlength=n).astype(float)
    if norm == "sym":
        w = 1 / np.sqrt(np.maximum(deg[src], 1) * np.maximum(deg[dst], 1))
    elif norm == "row":
        w = 1 / np.maximum(deg[dst], 1)
    else:  # không chuẩn hóa: tổng láng giềng
        w = np.ones(len(src))
    return torch.sparse_coo_tensor(np.stack([dst, src]), torch.tensor(w, dtype=torch.float32), (n, n)).coalesce().to(DEVICE)


def run_gcn(X, y, A, splits, hidden=64, p=0.5):
    accs = []
    for tr, va, te in splits:
        seed_everything(int(tr[0]))
        m, _ = fit(GCN(X.shape[1], hidden, 3, p=p).to(DEVICE), X, y, {"A_gcn": A}, tr, va, 0.01, 5e-4)
        m.eval()
        with torch.no_grad():
            accs.append(evaluate(m(X, {"A_gcn": A}).argmax(1), y, te)["acc"])
    return {"mean": r4(np.mean(accs)), "std": r4(np.std(accs))}


def ablation(X, y, E, yn):
    n = len(yn)
    splits = [planetoid_split(yn, 20, s) for s in range(10)]
    cfgs = [("GCN chuẩn (khuyên, chuẩn hóa đối xứng, ẩn 64, dropout 0,5)", dict(), dict()),
            ("Không có khuyên", dict(self_loop=False), dict()),
            ("Chuẩn hóa theo hàng D^-1 A (trung bình)", dict(norm="row"), dict()),
            ("Không chuẩn hóa (tổng láng giềng)", dict(norm="none"), dict()),
            ("Không dropout", dict(), dict(p=0.0)),
            ("Ẩn 16", dict(), dict(hidden=16)),
            ("Ẩn 256", dict(), dict(hidden=256))]
    out = []
    for name, akw, mkw in cfgs:
        r = run_gcn(X, y, adj(n, E, **akw), splits, **mkw)
        out.append({"config": name, **r})
        print(f"  {name:60s} {r}")
    return out


def inductive(X, y, E, yn):
    """Chia 60/20/20; khi huấn luyện, đồ thị chỉ gồm nút train + val (cạnh tới nút test bị xóa)."""
    n = len(yn)
    out = {k: {"transductive": [], "inductive": []} for k in ["mlp", "gcn", "sage", "gat"]}
    for seed in range(3):
        tr, va, te = full_split(n, seed)
        test = np.zeros(n, bool)
        test[te.cpu().numpy()] = True
        keep = ~(test[E[0]] | test[E[1]])
        g_full = build_graph(n, E)
        g_obs = build_graph(n, E[:, keep])
        G = lambda g: {"A_gcn": g[0].to(DEVICE), "A_mean": g[1].to(DEVICE), "edge_index": g[2].to(DEVICE)}
        for k in out:
            for mode, g_train in [("transductive", G(g_full)), ("inductive", G(g_obs))]:
                seed_everything(seed)
                m, _ = fit(make(k, X.shape[1]).to(DEVICE), X, y, g_train, tr, va, FACTORIES[k][2], FACTORIES[k][3], 400)
                m.eval()
                with torch.no_grad():  # lúc suy luận luôn dùng đồ thị đầy đủ (nút mới đã xuất hiện)
                    out[k][mode].append(evaluate(m(X, G(g_full)).argmax(1), y, te)["acc"])
        print("  seed", seed, {k: [round(v[-1], 4) for v in d.values()] for k, d in out.items()})
    return {k: {mode: {"mean": r4(np.mean(v)), "std": r4(np.std(v))} for mode, v in d.items()}
            for k, d in out.items()}, int(keep.sum()), int(E.shape[1])


def main():
    seed_everything()
    ids, Xn, yn, E, vocab = load_pubmed()
    X, y = torch.tensor(Xn, device=DEVICE), torch.tensor(yn, device=DEVICE)
    print("(a) Ablation GCN")
    abl = ablation(X, y, E, yn)
    print("(b) Quy nạp")
    ind, kept, total = inductive(X, y, E, yn)
    print(ind)
    save_results("ch6_extra", {"ablation": abl, "inductive": ind, "inductive_edges_kept": kept, "edges_total": total})


if __name__ == "__main__":
    main()
