"""Chương 6 — Học sâu trên đồ thị: GCN, GraphSAGE, GAT tự cài bằng PyTorch thuần (không dùng lớp của PyG).

Đồ thị được biểu diễn bằng edge_index (2 × |E|, đã đối xứng hoá). Mọi phép tổng hợp láng giềng viết bằng
index_add_ / ma trận thưa để thấy rõ cơ chế truyền thông điệp (message passing).
"""
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import DATA

CLASSES = {
    "Cora": ["Case_Based", "Genetic_Algorithms", "Neural_Networks", "Probabilistic_Methods",
             "Reinforcement_Learning", "Rule_Learning", "Theory"],
    "CiteSeer": ["Agents", "AI", "DB", "IR", "ML", "HCI"],
    # Mạch tiểu đường (tiểu luận final): 19.717 bài báo PubMed về đái tháo đường, 3 lớp theo MeSH
    "PubMed": ["ĐTĐ thực nghiệm", "ĐTĐ típ 1", "ĐTĐ típ 2"],
}


def load_planetoid(name):
    """Nạp Cora/CiteSeer/PubMed (Planetoid, phép chia chuẩn 20 nút/lớp, 500 val, 1000 test) và chuẩn hoá đặc trưng theo hàng."""
    from torch_geometric.datasets import Planetoid
    data = Planetoid(str(DATA / "ch6"), name)[0]
    x = data.x
    data.x = x / x.sum(1, keepdim=True).clamp(min=1)
    return data


# ---------------------------------------------------------------- tiện ích đồ thị
def add_self_loops(edge_index, n):
    loops = torch.arange(n, device=edge_index.device)
    return torch.cat([edge_index, torch.stack([loops, loops])], 1)


def gcn_norm_adj(edge_index, n):
    """Â = D̃^{-1/2}(A + I)D̃^{-1/2} dưới dạng ma trận thưa (Kipf & Welling, 2017)."""
    ei = add_self_loops(edge_index, n)
    deg = torch.zeros(n, device=ei.device).index_add_(0, ei[1], torch.ones(ei.size(1), device=ei.device))
    dinv = deg.pow(-0.5)
    w = dinv[ei[0]] * dinv[ei[1]]
    return torch.sparse_coo_tensor(ei.flip(0), w, (n, n)).coalesce()


def scatter_softmax(score, index, n):
    """Softmax của điểm cạnh theo từng nút đích: trừ max cục bộ để ổn định số rồi chuẩn hoá trên láng giềng."""
    smax = torch.full((n,) + score.shape[1:], -1e9, device=score.device)
    smax = smax.scatter_reduce(0, index.view(-1, *[1] * (score.dim() - 1)).expand_as(score), score, "amax")
    e = (score - smax[index]).exp()
    denom = torch.zeros_like(smax).index_add_(0, index, e)
    return e / denom[index].clamp(min=1e-16)


# ---------------------------------------------------------------- các lớp
class GCNLayer(nn.Module):
    """H' = Â H W: mỗi nút lấy trung bình có trọng số (theo bậc) của chính nó và láng giềng."""

    def __init__(self, d_in, d_out):
        super().__init__()
        self.lin = nn.Linear(d_in, d_out, bias=False)
        self.bias = nn.Parameter(torch.zeros(d_out))

    def forward(self, x, adj):
        return torch.sparse.mm(adj, self.lin(x)) + self.bias


class SAGELayer(nn.Module):
    """GraphSAGE-mean: h_v' = W_self h_v + W_neigh · mean_{u∈N(v)} h_u, có thể lấy mẫu tối đa k láng giềng."""

    def __init__(self, d_in, d_out):
        super().__init__()
        self.lin_self = nn.Linear(d_in, d_out)
        self.lin_neigh = nn.Linear(d_in, d_out, bias=False)

    def forward(self, x, edge_index):
        src, dst = edge_index
        n = x.size(0)
        agg = torch.zeros(n, x.size(1), device=x.device).index_add_(0, dst, x[src])
        deg = torch.zeros(n, device=x.device).index_add_(0, dst, torch.ones_like(dst, dtype=x.dtype))
        return self.lin_self(x) + self.lin_neigh(agg / deg.clamp(min=1).unsqueeze(1))


class GATLayer(nn.Module):
    """GAT: α_vu = softmax_u LeakyReLU(aᵀ[W h_v ‖ W h_u]); h_v' = ‖_k Σ_u α^k_vu W^k h_u (ghép hoặc lấy trung bình)."""

    def __init__(self, d_in, d_out, heads=8, concat=True, dropout=0.6):
        super().__init__()
        self.heads, self.d_out, self.concat, self.dropout = heads, d_out, concat, dropout
        self.lin = nn.Linear(d_in, heads * d_out, bias=False)
        self.a_src = nn.Parameter(torch.empty(heads, d_out))
        self.a_dst = nn.Parameter(torch.empty(heads, d_out))
        self.bias = nn.Parameter(torch.zeros(heads * d_out if concat else d_out))
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)
        self.last_alpha = None

    def forward(self, x, edge_index):
        n = x.size(0)
        ei = add_self_loops(edge_index, n)
        src, dst = ei
        h = self.lin(x).view(n, self.heads, self.d_out)
        score = F.leaky_relu((h * self.a_src).sum(-1)[src] + (h * self.a_dst).sum(-1)[dst], 0.2)   # |E| × heads
        alpha = scatter_softmax(score, dst, n)
        self.last_alpha = (ei.detach(), alpha.detach())
        alpha = F.dropout(alpha, self.dropout, self.training)
        out = torch.zeros(n, self.heads, self.d_out, device=x.device).index_add_(0, dst, h[src] * alpha.unsqueeze(-1))
        out = out.reshape(n, -1) if self.concat else out.mean(1)
        return out + self.bias


# ---------------------------------------------------------------- mô hình
class GNN(nn.Module):
    """Chồng L lớp cùng loại (gcn | sage | gat | mlp), ReLU/ELU + dropout giữa các lớp, tuỳ chọn nối tắt (residual)."""

    def __init__(self, kind, d_in, d_hid, n_cls, layers=2, dropout=0.5, heads=8, residual=False):
        super().__init__()
        self.kind, self.dropout, self.residual = kind, dropout, residual
        dims = [d_in] + [d_hid * (heads if kind == "gat" else 1)] * (layers - 1) + [n_cls]
        self.layers = nn.ModuleList()
        for i in range(layers):
            last = i == layers - 1
            if kind == "gcn":
                self.layers.append(GCNLayer(dims[i], dims[i + 1]))
            elif kind == "sage":
                self.layers.append(SAGELayer(dims[i], dims[i + 1]))
            elif kind == "gat":
                self.layers.append(GATLayer(dims[i], n_cls if last else d_hid, heads=1 if last else heads,
                                            concat=not last, dropout=dropout))
            else:
                self.layers.append(nn.Linear(dims[i], dims[i + 1]))
        self.hidden = None

    def forward(self, x, graph):
        for i, layer in enumerate(self.layers):
            x_in = x
            x = F.dropout(x, self.dropout, self.training)
            x = layer(x) if self.kind == "mlp" else layer(x, graph)
            if i < len(self.layers) - 1:
                x = F.elu(x) if self.kind == "gat" else F.relu(x)
                if self.residual and x.shape == x_in.shape:
                    x = x + x_in
                self.hidden = x
        return x


def graph_input(kind, data, device):
    """GCN cần Â chuẩn hoá (thưa); SAGE/GAT dùng edge_index; MLP không dùng đồ thị."""
    if kind == "gcn":
        return gcn_norm_adj(data.edge_index.to(device), data.num_nodes)
    return data.edge_index.to(device)


def accuracy(logits, y, mask):
    return (logits[mask].argmax(1) == y[mask]).float().mean().item()


def train_node(kind, data, seed=0, epochs=250, lr=0.01, wd=5e-4, hidden=None, layers=2, dropout=None,
               residual=False, patience=50, device="cuda"):
    """Huấn luyện bán giám sát (transductive): loss chỉ trên nút train; chọn epoch theo accuracy validation."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    hidden = hidden or (8 if kind == "gat" else 64)
    dropout = (0.6 if kind == "gat" else 0.5) if dropout is None else dropout
    model = GNN(kind, data.num_features, hidden, int(data.y.max()) + 1, layers=layers, dropout=dropout,
                residual=residual).to(device)
    x, y = data.x.to(device), data.y.to(device)
    g = graph_input(kind, data, device)
    tr, va, te = (m.to(device) for m in (data.train_mask, data.val_mask, data.test_mask))
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    best = {"val": -1}
    hist = {"loss": [], "train": [], "val": []}
    t0, bad = time.perf_counter(), 0
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        out = model(x, g)
        loss = F.cross_entropy(out[tr], y[tr])
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            out = model(x, g)
        v = accuracy(out, y, va)
        hist["loss"].append(loss.item())
        hist["train"].append(accuracy(out, y, tr))
        hist["val"].append(v)
        if v > best["val"]:
            best = {"val": v, "test": accuracy(out, y, te), "epoch": ep,
                    "state": {k: t.detach().clone() for k, t in model.state_dict().items()}}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    if device == "cuda":
        torch.cuda.synchronize()
    seconds = time.perf_counter() - t0
    model.load_state_dict(best["state"])
    model.eval()
    with torch.no_grad():
        logits = model(x, g)
    return {"model": model, "val": best["val"], "test": best["test"], "epoch": best["epoch"], "seconds": seconds,
            "epochs_run": ep + 1, "params": sum(p.numel() for p in model.parameters()), "history": hist,
            "logits": logits.cpu(), "hidden": model.hidden.detach().cpu() if model.hidden is not None else None}


def mad(h, edge_index=None):
    """Mean Average Distance (Chen et al., 2020): trung bình khoảng cách cosine giữa các cặp biểu diễn nút.
    MAD → 0 nghĩa là các nút đã bị làm trơn thành gần như một vector (over-smoothing)."""
    h = F.normalize(h, dim=1)
    if edge_index is None:
        sim = h @ h.T
        n = h.size(0)
        return float(((1 - sim).sum() / (n * (n - 1))).item())
    src, dst = edge_index
    return float((1 - (h[src] * h[dst]).sum(1)).mean().item())


def homophily(edge_index, y):
    """Tỷ lệ cạnh nối hai nút cùng nhãn (edge homophily)."""
    src, dst = edge_index
    return float((y[src] == y[dst]).float().mean().item())


# ---------------------------------------------------------------- dự đoán liên kết (GAE)
class GAE(nn.Module):
    """Graph Auto-Encoder: bộ mã hoá GCN 2 lớp -> z; bộ giải mã tích vô hướng σ(z_uᵀ z_v) cho xác suất có cạnh."""

    def __init__(self, d_in, d_hid=128, d_z=64):
        super().__init__()
        self.l1, self.l2 = GCNLayer(d_in, d_hid), GCNLayer(d_hid, d_z)

    def encode(self, x, adj):
        return self.l2(F.relu(self.l1(x, adj)), adj)

    @staticmethod
    def decode(z, pairs):
        return (z[pairs[0]] * z[pairs[1]]).sum(1)
