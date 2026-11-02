"""
Chương 6: Học sâu trên đồ thị — phân loại y văn về đái tháo đường trên đồ thị trích dẫn Pubmed-Diabetes.

Mạng chú ý trên đồ thị (GAT, Veličković 2018) cài bằng PyTorch thuần, huấn luyện ở lab/ch6_gnn.py
(data/models/ch6_gat.pt + ch6_pubmed_graph.npz). Dùng để gán nhãn "ĐTĐ thực nghiệm / típ 1 / típ 2"
cho tài liệu y văn trước khi đưa vào Knowledge Base, kèm các bài báo láng giềng có trọng số chú ý cao nhất.
"""

from functools import lru_cache
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import BASE_DIR

MODEL_PATH = BASE_DIR / "data" / "models" / "ch6_gat.pt"
GRAPH_PATH = BASE_DIR / "data" / "models" / "ch6_pubmed_graph.npz"
CLASSES = ["ĐTĐ thực nghiệm (động vật)", "ĐTĐ típ 1", "ĐTĐ típ 2"]


class GATLayer(nn.Module):
    """α_vu = softmax_u(LeakyReLU(a_srcᵀ W h_u + a_dstᵀ W h_v)); h_v' = ‖_k Σ_u α^k_vu W^k h_u."""

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
        e = F.leaky_relu((z * self.a_src).sum(-1)[src] + (z * self.a_dst).sum(-1)[dst], 0.2)
        e_max = torch.full((n, self.heads), -1e9).index_reduce_(0, dst, e, "amax", include_self=True)
        ex = torch.exp(e - e_max[dst])
        denom = torch.zeros(n, self.heads).index_add_(0, dst, ex)
        alpha = ex / (denom[dst] + 1e-16)
        self.alpha = alpha.detach()
        alpha = F.dropout(alpha, self.p, self.training)
        out = torch.zeros(n, self.heads, self.h).index_add_(0, dst, alpha[..., None] * z[src])
        out = out.reshape(n, -1) if self.concat else out.mean(1)
        return out + self.bias


class GAT(nn.Module):
    def __init__(self, d=500, h=8, c=3, heads=8, p=0.6):
        super().__init__()
        self.g1 = GATLayer(d, h, heads, True, p)
        self.g2 = GATLayer(h * heads, c, 8, False, p)
        self.p = p

    def forward(self, x, g):
        x = F.dropout(x, self.p, self.training)
        x = F.elu(self.g1(x, g["edge_index"]))
        return self.g2(F.dropout(x, self.p, self.training), g["edge_index"])


class PubmedGraphClassifier:
    def __init__(self):
        data = np.load(GRAPH_PATH, allow_pickle=False)
        self.ids = [str(i) for i in data["ids"]]
        self.index = {p: i for i, p in enumerate(self.ids)}
        self.y, self.vocab = data["y"], [str(v) for v in data["vocab"]]
        self.X = torch.tensor(data["X"].astype(np.float32))
        E = data["E"]
        n = len(self.ids)
        src = np.concatenate([E[0], E[1], np.arange(n)])
        dst = np.concatenate([E[1], E[0], np.arange(n)])
        self.edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long)
        ckpt = torch.load(MODEL_PATH, map_location="cpu")
        self.test_idx = set(ckpt["test_idx"].tolist())
        self.model = GAT()
        self.model.load_state_dict(ckpt["state"])
        self.model.eval()
        with torch.no_grad():
            self.prob = F.softmax(self.model(self.X, {"edge_index": self.edge_index}), 1).numpy()
        # trọng số chú ý của tầng 1 (trung bình 8 đầu) cho từng cạnh u → v
        self.alpha = self.model.g1.alpha.mean(1).numpy()
        self.src, self.dst = src, dst

    def paper(self, pid: Optional[str] = None, seed: Optional[int] = None) -> Dict[str, Any]:
        if pid is None:
            rng = np.random.default_rng(seed)
            i = int(rng.choice(sorted(self.test_idx)))
        else:
            if pid not in self.index:
                raise KeyError(pid)
            i = self.index[pid]
        mask = (self.dst == i) & (self.src != i)
        nbrs = sorted(zip(self.src[mask], self.alpha[mask]), key=lambda t: -t[1])[:8]
        terms = [self.vocab[j] for j in np.argsort(-self.X[i].numpy())[:8] if self.X[i, j] > 0]
        return {
            "paper_id": self.ids[i], "pubmed_url": f"https://pubmed.ncbi.nlm.nih.gov/{self.ids[i]}/",
            "in_test_set": i in self.test_idx, "true_label": CLASSES[int(self.y[i])],
            "predicted_label": CLASSES[int(self.prob[i].argmax())],
            "probabilities": {c: round(float(p), 4) for c, p in zip(CLASSES, self.prob[i])},
            "top_terms": terms, "degree": int(mask.sum()),
            "neighbors": [{"paper_id": self.ids[j], "attention": round(float(a), 4),
                           "label": CLASSES[int(self.y[j])]} for j, a in nbrs],
        }


@lru_cache(maxsize=1)
def get_pubmed_classifier() -> Optional[PubmedGraphClassifier]:
    return PubmedGraphClassifier() if MODEL_PATH.exists() and GRAPH_PATH.exists() else None
