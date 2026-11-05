"""Thực nghiệm truy xuất: 5 chiến lược chunking × 5 bộ truy xuất, so sánh mô hình embedding, quét kích thước chunk.

Một chunk được coi là liên quan nếu chứa trọn câu trả lời (span) của câu hỏi; nhờ vậy cùng một bộ 114 câu hỏi
chấm được mọi cách phân mảnh. Toàn bộ kho (y tế + tài chính) được lập chỉ mục chung, không lọc theo miền.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tl2 import chunking as C  # noqa: E402
from tl2.common import DATA, save_json, seed_all  # noqa: E402
from tl2.corpus import load_documents, nfc  # noqa: E402
from tl2.retrieval import (BM25Retriever, DenseRetriever, Embedder, HybridRRF, Reranker, RerankPipeline,  # noqa: E402
                           TfidfRetriever, is_relevant, rank_metrics)

K = 10


def load_qa():
    qa = json.loads((DATA / "qa" / "qa_eval.json").read_text(encoding="utf-8"))
    for q in qa:
        q["a"] = nfc(q["a"])
    return qa


class ParentChildRetriever:
    """Tìm trên chunk con (nhỏ, sắc nét) rồi trả chunk cha (đủ ngữ cảnh), bỏ cha trùng lặp."""

    def __init__(self, child_retriever, parents):
        self.child, self.parents = child_retriever, {p.chunk_id: p for p in parents}
        self.name = child_retriever.name
        self.chunks = list(self.parents.values())

    def search(self, query, k=5):
        out, seen = [], set()
        for c, s in self.child.search(query, k * 6):
            if c.parent_id not in seen:
                seen.add(c.parent_id)
                out.append((self.parents[c.parent_id], s))
            if len(out) == k:
                break
        return out


def evaluate(retriever, qa, corpus_chunks):
    rows, lat = [], []
    n_rel = {q["id"]: sum(is_relevant(c, q["a"]) for c in corpus_chunks) for q in qa}
    for q in qa:
        t0 = time.perf_counter()
        hits = retriever.search(q["q"], K)
        lat.append(1000 * (time.perf_counter() - t0))
        m = rank_metrics([c for c, _ in hits], q["a"], n_rel[q["id"]])
        rows.append({"id": q["id"], "domain": q["domain"], **m, "top1": hits[0][0].chunk_id})
    keys = [k for k in rows[0] if k not in ("id", "domain", "top1")]
    agg = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    for dom in ("health", "finance"):
        agg[f"{dom}_hit@1"] = float(np.mean([r["hit@1"] for r in rows if r["domain"] == dom]))
        agg[f"{dom}_hit@5"] = float(np.mean([r["hit@5"] for r in rows if r["domain"] == dom]))
    agg["latency_ms"] = float(np.median(lat))
    agg["answerable"] = float(np.mean([n_rel[q["id"]] > 0 for q in qa]))
    return agg, rows


def pipelines(chunks, emb, rr):
    tf, bm, de = TfidfRetriever(chunks), BM25Retriever(chunks), DenseRetriever(chunks, emb)
    hy = HybridRRF([bm, de])
    return {"tfidf": tf, "bm25": bm, "dense": de, "hybrid": hy, "hybrid_rerank": RerankPipeline(hy, rr)}


def build_strategies(docs, embed_fn):
    return {
        "fixed": C.chunk_corpus(docs, C.fixed_size, size=400, overlap=60),
        "recursive": C.chunk_corpus(docs, C.recursive, size=400, overlap=60),
        "structure": C.chunk_corpus(docs, C.structure_aware, size=900),
        "semantic": C.chunk_corpus(docs, C.semantic, embed=embed_fn, percentile=80),
    }


def stage_grid(docs, qa, emb):
    """Giai đoạn 1: chiến lược chunking × bộ truy xuất (cần reranker)."""
    rr = Reranker()
    out = {"n_docs": len(docs), "n_questions": len(qa), "n_chars": sum(len(d.text) for d in docs)}
    strategies = build_strategies(docs, lambda texts: emb.encode(texts))
    parents, children = [], []
    for d in docs:
        p, c = C.parent_child(d)
        parents += p
        children += c
    grid, per_q, stats = {}, {}, {}
    for name, chunks in list(strategies.items()) + [("parent_child", children)]:
        L = [len(c.text) for c in (parents if name == "parent_child" else chunks)]
        stats[name] = {"n_chunks": len(chunks), "n_parents": len(parents) if name == "parent_child" else None,
                       "mean_len": float(np.mean(L)), "max_len": int(np.max(L)), "min_len": int(np.min(L)),
                       "intact": float(np.mean([any(q["a"] in c.text for c in (parents if name == "parent_child"
                                                                                 else chunks)) for q in qa])),
                       "lengths": L}
        grid[name] = {}
        for pname, r in pipelines(chunks, emb, rr).items():
            if name == "parent_child":
                r = ParentChildRetriever(r, parents)
            agg, rows = evaluate(r, qa, parents if name == "parent_child" else chunks)
            grid[name][pname] = agg
            per_q[f"{name}/{pname}"] = rows
            print(f"{name:13s} {pname:14s} hit@1 {agg['hit@1']:.3f} hit@5 {agg['hit@5']:.3f} mrr {agg['mrr']:.3f} "
                  f"{agg['latency_ms']:.1f} ms", flush=True)
    out["grid"], out["chunk_stats"] = grid, stats
    save_json("retrieval_grid", out)
    save_json("retrieval_perq", per_q)


def stage_embed(docs, qa):
    """Giai đoạn 2: so sánh mô hình embedding trên chunk theo cấu trúc (mỗi mô hình nạp rồi giải phóng)."""
    import gc
    import torch
    st = C.chunk_corpus(docs, C.structure_aware, size=900)
    embs = {}
    for key in ("minilm", "e5-small", "e5-base"):
        t0 = time.perf_counter()
        r = DenseRetriever(st, Embedder(key))
        index_s = time.perf_counter() - t0
        agg, _ = evaluate(r, qa, st)
        agg["index_s"] = index_s
        agg["dim"] = int(r.E.shape[1])
        embs[key] = agg
        print("embed", key, agg["hit@1"], agg["hit@5"], flush=True)
        del r
        Embedder._cache.clear()
        gc.collect()
        torch.cuda.empty_cache()
    save_json("retrieval_embed", embs)


def stage_sweep(docs, qa, emb):
    """Giai đoạn 3: quét kích thước chunk (recursive, chồng lấn 15%)."""
    sweep = []
    for size in (120, 200, 300, 400, 600, 900, 1300):
        ch = C.chunk_corpus(docs, C.recursive, size=size, overlap=int(0.15 * size))
        row = {"size": size, "n_chunks": len(ch), "mean_len": float(np.mean([len(c.text) for c in ch])),
               "intact": float(np.mean([any(q["a"] in c.text for c in ch) for q in qa]))}
        for pname, r in (("bm25", BM25Retriever(ch)), ("dense", DenseRetriever(ch, emb))):
            agg, _ = evaluate(r, qa, ch)
            row[f"{pname}_hit@1"], row[f"{pname}_hit@5"], row[f"{pname}_mrr"] = agg["hit@1"], agg["hit@5"], agg["mrr"]
        # token ngữ cảnh đưa vào LLM khi lấy top-5 (ước lượng 1 token ≈ 4 ký tự)
        row["ctx_tokens_top5"] = 5 * row["mean_len"] / 4
        sweep.append(row)
        print("size", size, row, flush=True)
    save_json("retrieval_sweep", sweep)


def stage_viz(docs, qa, emb):
    """Giai đoạn 4: dữ liệu vẽ hình — khoảng cách ngữ nghĩa và không gian embedding."""
    embed_fn = lambda texts: emb.encode(texts)  # noqa: E731
    st = C.chunk_corpus(docs, C.structure_aware, size=900)
    out = {}
    doc = next(d for d in docs if d.doc_id == "h03-ruou-bia")
    _, dist, thr = C.semantic(doc, embed_fn, percentile=80)
    out["semantic_example"] = {"doc": doc.doc_id, "dist": dist.tolist(), "threshold": float(thr)}
    E = emb.encode([c.index_text() for c in st])
    Ec = E - E.mean(0)
    U, S, Vt = np.linalg.svd(Ec, full_matrices=False)
    P = Ec @ Vt[:2].T
    out["embedding_pca"] = {"x": P[:, 0].tolist(), "y": P[:, 1].tolist(), "doc": [c.doc_id for c in st],
                            "domain": [c.domain for c in st], "var_ratio": (S[:2] ** 2 / (S ** 2).sum()).tolist()}
    Q = emb.encode([q["q"] for q in qa], kind="query")
    sims = Q @ E.T
    out["sim_matrix_sample"] = {"q": [q["id"] for q in qa[::12]], "c": [c.chunk_id for c in st[::6]],
                                "s": sims[::12, ::6].round(3).tolist()}
    save_json("retrieval_viz", out)


def main():
    seed_all()
    docs, qa = load_documents(), load_qa()
    stages = sys.argv[1:] or ["grid", "embed", "sweep", "viz"]
    if "embed" in stages:
        stage_embed(docs, qa)
    emb = Embedder("e5-base")
    if "grid" in stages:
        stage_grid(docs, qa, emb)
    if "sweep" in stages:
        stage_sweep(docs, qa, emb)
    if "viz" in stages:
        stage_viz(docs, qa, emb)


if __name__ == "__main__":
    main()
