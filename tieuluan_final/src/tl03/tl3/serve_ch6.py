"""Suy luận Chương 6 cho Web App: phân loại bài báo Pubmed-Diabetes và Cora bằng GCN / GraphSAGE / GAT / MLP."""
import time

import torch

from .ch6_gnn import CLASSES, GNN, graph_input, load_planetoid
from .common import MODELS

LABELS_VI = {"Case_Based": "Lập luận theo tình huống", "Genetic_Algorithms": "Thuật toán di truyền",
             "Neural_Networks": "Mạng nơ-ron", "Probabilistic_Methods": "Phương pháp xác suất",
             "Reinforcement_Learning": "Học tăng cường", "Rule_Learning": "Học luật", "Theory": "Lý thuyết",
             "ĐTĐ thực nghiệm": "ĐTĐ thực nghiệm", "ĐTĐ típ 1": "ĐTĐ típ 1", "ĐTĐ típ 2": "ĐTĐ típ 2"}
MODEL_IDS = ["gcn", "sage", "gat", "mlp"]
DATASETS = {"ch6_pubmed": ("PubMed", "pubmed"), "ch6_cora": ("Cora", "cora")}

SPECS = {
    "ch6_pubmed": {
        "title": "Phân loại y văn đái tháo đường trên đồ thị trích dẫn PubMed (GNN)",
        "kind": "graph",
        "inputs": [{"name": "node_id", "label": "Mã nút (bài báo) 0–19716", "type": "number", "min": 0, "max": 19716,
                    "default": 100},
                   {"name": "model", "label": "Mô hình", "type": "select", "options": MODEL_IDS, "default": "gcn"}],
        "models": MODEL_IDS,
        "example": {"node_id": 100, "model": "gcn"},
    },
    "ch6_cora": {
        "title": "Phân loại chủ đề bài báo trên đồ thị trích dẫn Cora (GNN)",
        "kind": "graph",
        "inputs": [{"name": "node_id", "label": "Mã nút (bài báo) 0–2707", "type": "number", "min": 0, "max": 2707,
                    "default": 1701},
                   {"name": "model", "label": "Mô hình", "type": "select", "options": MODEL_IDS, "default": "gcn"}],
        "models": MODEL_IDS,
        "example": {"node_id": 1701, "model": "gcn"},
    },
}

_STATE = {}


def _load(ds_name, prefix, model_id):
    """Nạp đồ thị một lần và mô hình theo yêu cầu; chạy CPU."""
    if ds_name not in _STATE:
        d = load_planetoid(ds_name)
        _STATE[ds_name] = {"data": d, "deg": torch.bincount(d.edge_index[1], minlength=d.num_nodes), "models": {}}
    st_ds = _STATE[ds_name]
    if model_id not in st_ds["models"]:
        ck = torch.load(MODELS / "ch6" / f"{prefix}_{model_id}.pt", map_location="cpu", weights_only=False)
        d = st_ds["data"]
        m = GNN(model_id, d.num_features, ck["hidden"], len(CLASSES[ds_name]))
        m.load_state_dict(ck["state"])
        m.eval()
        with torch.no_grad():
            probs = torch.softmax(m(d.x, graph_input(model_id, d, "cpu")), 1)
        st_ds["models"][model_id] = {"probs": probs, "test": ck["test"]}
    return st_ds["data"], st_ds["deg"], st_ds["models"][model_id]


def predict(exp_id, payload, model_id=None):
    """payload {"node_id": int, "model": gcn|sage|gat|mlp} -> nhãn, xác suất các lớp, láng giềng và nhãn thật."""
    if exp_id not in DATASETS:
        raise KeyError(exp_id)
    ds_name, prefix = DATASETS[exp_id]
    t0 = time.perf_counter()
    model_id = model_id or payload.get("model", "gcn")
    if model_id not in MODEL_IDS:
        raise ValueError(f"mô hình không hợp lệ: {model_id}")
    data, deg, st = _load(ds_name, prefix, model_id)
    v = int(payload.get("node_id", 0))
    if not 0 <= v < data.num_nodes:
        raise ValueError(f"node_id phải trong 0..{data.num_nodes - 1}")
    names = CLASSES[ds_name]
    p = st["probs"][v]
    k = int(p.argmax())
    nbrs = data.edge_index[0, data.edge_index[1] == v].tolist()
    split = "train" if data.train_mask[v] else "validation" if data.val_mask[v] else "test" if data.test_mask[v] else "không gán nhãn"
    return {"label": LABELS_VI[names[k]],
            "proba": {LABELS_VI[n]: round(float(p[i]), 4) for i, n in enumerate(names)},
            "details": {"dataset": ds_name, "node_id": v, "true_label": LABELS_VI[names[int(data.y[v])]],
                        "correct": k == int(data.y[v]), "degree": int(deg[v]), "split": split,
                        "model_test_acc": round(float(st["test"]), 4),
                        "neighbors": [{"id": int(u), "label": LABELS_VI[names[int(data.y[u])]]} for u in nbrs[:20]]},
            "model": model_id, "latency_ms": round(1000 * (time.perf_counter() - t0), 2)}
