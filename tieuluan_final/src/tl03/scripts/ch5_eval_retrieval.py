"""Chương 5 — đánh giá các bộ truy xuất trên bộ câu hỏi tự sinh có đáp án (ba miền).

    python scripts/ch5_eval_retrieval.py <miền>    mỗi miền một tiến trình -> outputs/metrics/ch5_retrieval_<miền>.json
    python scripts/ch5_eval_retrieval.py merge     gộp ba miền -> outputs/metrics/ch5_retrieval.json
Ghi data/ch5/qa_<miền>.json. Độ đo: Recall@k, MRR, nDCG theo bộ truy xuất, loại câu hỏi, kiểu diễn đạt; độ trễ;
thử nghiệm trọng số nhánh đồ thị trong RRF và số ứng viên rerank.
"""
import gc
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tl3.common import data_dir, limit_threads, load_json, save_json, seed_all  # noqa: E402
from tl3.gpu import cap  # noqa: E402
from tl3.ch5_agent import MODELS, load_domain  # noqa: E402
from tl3.ch5_qa import make_questions  # noqa: E402
from tl3.ch5_retrieval import (DenseRetriever, Embedder, GraphRetriever, HybridRRF, Reranker, RerankPipeline,  # noqa: E402
                               load_index, rank_metrics)

DOMAINS = ("diabetes", "stock", "gold")
DEPTH = 100


class Cached:
    """Kết quả top-DEPTH của một bộ truy xuất gốc, tính trước cho mọi câu hỏi rồi giải phóng mô hình.
    search(q, k) trả k phần tử đầu; ms là độ trễ đo được ở lần truy vấn thật."""

    def __init__(self, name, retriever, questions):
        self.name, self.hits, self.ms = name, {}, {}
        for q in questions:
            t0 = time.perf_counter()
            self.hits[q] = retriever.search(q, DEPTH)
            self.ms[q] = 1000 * (time.perf_counter() - t0)

    def search(self, q, k=10):
        return self.hits[q][:k]


def free():
    """Nhả mô hình và bộ nhớ GPU giữa hai bộ truy xuất: máy chạy song song nhiều chương, RAM có hạn."""
    Embedder._cache.clear()
    gc.collect()
    import torch
    torch.cuda.empty_cache()


def build(D, questions):
    """Mười cấu hình: 6 bộ gốc (tính trước, tuần tự để chỉ một mô hình nằm trong bộ nhớ) và các tổ hợp RRF."""
    base = {}
    for name, path in (("TF-IDF", "tfidf"), ("BM25", "bm25")):
        r = load_index(MODELS / f"{D.name}_{path}.pkl")
        base[name] = Cached(name, r, questions)
        del r
        free()
    for key in ("minilm", "e5-small", "e5-base"):
        r = DenseRetriever(np.load(MODELS / f"{D.name}_emb_{key}.npy"), Embedder(key, "cpu"))
        base[f"Dense {key}"] = Cached(f"Dense {key}", r, questions)
        del r
        free()
    base["Graph"] = Cached("Graph", GraphRetriever(D.ids, D.link), questions)
    combo = {"RRF BM25+Dense": HybridRRF([base["BM25"], base["Dense e5-small"]]),
             "RRF BM25+Dense+Graph": HybridRRF([base["BM25"], base["Dense e5-small"], base["Graph"]],
                                               weights=[1.0, 1.0, 2.0])}
    return base, combo


def evaluate(name, retr, qs, ids, rows, cached=None, extra_ms=None):
    for q in qs:
        t0 = time.perf_counter()
        hits = retr.search(q["question"], 10)
        ms = 1000 * (time.perf_counter() - t0)
        if cached is not None:
            ms = cached.ms[q["question"]]
        if extra_ms is not None:            # tổ hợp: cộng độ trễ thật của các bộ gốc chạy bên trong
            ms += extra_ms(q["question"])
        m = rank_metrics([ids[i] for i, _ in hits], set(q["gold"]))
        rows.append({"retriever": name, "qid": q["qid"], "domain": q["domain"], "type": q["type"],
                     "style": q["style"], **m, "ms": ms})


def run_domain(dom):
    limit_threads(4)
    seed_all()
    cap(0.35)
    rows, sweep = [], []
    D = load_domain(dom)
    qs = make_questions(D)
    (data_dir("ch5") / f"qa_{dom}.json").write_text(json.dumps(qs, ensure_ascii=False, indent=1), encoding="utf-8")
    qa_count = {"total": len(qs), **{t: sum(q["type"] == t for q in qs) for t in
                                     ("record", "group", "month", "year", "indicator")},
                "para": sum(q["style"] == "para" for q in qs)}
    texts = [q["question"] for q in qs]
    base, combo = build(D, texts)
    for name, r in base.items():
        evaluate(name, r, qs, D.ids, rows, cached=r)
    lat = lambda names: (lambda q: sum(base[n].ms[q] for n in names))
    combo_ms = {"RRF BM25+Dense": lat(["BM25", "Dense e5-small"]),
                "RRF BM25+Dense+Graph": lat(["BM25", "Dense e5-small", "Graph"])}
    for name, r in combo.items():
        evaluate(name, r, qs, D.ids, rows, extra_ms=combo_ms[name])
    # trọng số nhánh đồ thị trong RRF
    for w in (0.0, 0.5, 1.0, 2.0, 4.0):
        h = HybridRRF([base["BM25"], base["Dense e5-small"], base["Graph"]], weights=[1.0, 1.0, w])
        tmp = []
        evaluate(f"w={w}", h, qs, D.ids, tmp)
        sweep.append({"domain": dom, "param": "graph_weight", "value": w,
                      "r@1": float(np.mean([t["r@1"] for t in tmp])), "mrr": float(np.mean([t["mrr"] for t in tmp]))})
    # cross-encoder bf16 (~1,2 GB) chấm vài nghìn cặp: việc GPU nhẹ trong giới hạn cap(0.35), không giữ khoá nặng
    rr = Reranker()
    evaluate("RRF+Graph+Rerank", RerankPipeline(combo["RRF BM25+Dense+Graph"], rr, D.texts, n=20), qs, D.ids, rows,
             extra_ms=combo_ms["RRF BM25+Dense+Graph"])
    evaluate("RRF+Rerank (không đồ thị)", RerankPipeline(combo["RRF BM25+Dense"], rr, D.texts, n=20), qs, D.ids,
             rows, extra_ms=combo_ms["RRF BM25+Dense"])
    for n in (5, 10, 20, 40):
        tmp = []
        evaluate(f"n={n}", RerankPipeline(combo["RRF BM25+Dense"], rr, D.texts, n=n), qs, D.ids, tmp,
                 extra_ms=combo_ms["RRF BM25+Dense"])
        sweep.append({"domain": dom, "param": "rerank_n", "value": n,
                      "r@1": float(np.mean([t["r@1"] for t in tmp])),
                      "mrr": float(np.mean([t["mrr"] for t in tmp])),
                      "ms": float(np.mean([t["ms"] for t in tmp]))})
    save_json(f"ch5_retrieval_{dom}", {"qa": qa_count, "rows": rows, "sweep_rows": sweep})
    print(dom, "xong", flush=True)


def agg(rows, keys):
    groups = defaultdict(list)
    for r in rows:
        groups[tuple(r[k] for k in keys)].append(r)
    out = {}
    for k, rs in groups.items():
        out["|".join(map(str, k))] = {m: float(np.mean([r[m] for r in rs])) for m in
                                      ("r@1", "r@3", "r@5", "r@10", "mrr", "ndcg", "ms")} | {"n": len(rs)}
    return out


def merge():
    parts = {d: load_json(f"ch5_retrieval_{d}") for d in DOMAINS}
    rows = sum((p["rows"] for p in parts.values()), [])
    sweep = sum((p["sweep_rows"] for p in parts.values()), [])
    # khoá gọn không dấu cách cho macro \SL{ch5_retrieval/sl/<miền|all>/<bộ truy xuất>/<độ đo>}
    slug = {"TF-IDF": "tfidf", "BM25": "bm25", "Dense minilm": "minilm", "Dense e5-small": "esmall",
            "Dense e5-base": "ebase", "Graph": "graph", "RRF BM25+Dense": "rrf", "RRF BM25+Dense+Graph": "rrfg",
            "RRF+Rerank (không đồ thị)": "rrfrr", "RRF+Graph+Rerank": "rrfgrr"}
    sl = {}
    for scope, sub in [("all", rows)] + [(d, [r for r in rows if r["domain"] == d]) for d in DOMAINS] + \
            [(f"{d}{s}", [r for r in rows if r["domain"] == d and r["style"] == s]) for d in DOMAINS
             for s in ("lex", "para")]:
        if not sub:
            continue
        sl[scope] = {slug[n]: {m: float(np.mean([r[m] for r in sub if r["retriever"] == n]))
                               for m in ("r@1", "r@5", "r@10", "mrr", "ndcg", "ms")}
                     for n in slug if any(r["retriever"] == n for r in sub)}
    res = {"qa": {d: p["qa"] for d, p in parts.items()}, "sl": sl, "by_retriever": agg(rows, ["retriever"]),
           "by_domain": agg(rows, ["domain", "retriever"]), "by_style": agg(rows, ["domain", "style", "retriever"]),
           "by_type": agg(rows, ["type", "retriever"]), "sweep_rows": sweep, "rows": rows}
    save_json("ch5_retrieval", res)


if __name__ == "__main__":
    merge() if sys.argv[1] == "merge" else run_domain(sys.argv[1])
