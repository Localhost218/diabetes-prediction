"""Chương 5 — dựng KB, KG và chỉ mục cho ba miền rồi lưu vào models/ch5/ (để đánh giá và Web App nạp nhanh).

    python scripts/ch5_build.py <miền>     (miền: diabetes | stock | gold; mỗi miền một tiến trình để tiết kiệm RAM)
Ghi outputs/metrics/ch5_kb_<miền>.json: số tài liệu, độ dài, số nút/cạnh, phân bố quan hệ, thời gian dựng.
"""
import os
import pickle
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np

os.environ.setdefault("USE_TF", "0")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tl3.common import limit_threads, model_dir, save_json, seed_all  # noqa: E402
from tl3.gpu import cap, heavy  # noqa: E402
from tl3.ch5_domain import Domain  # noqa: E402
from tl3.ch5_kg import to_rdf  # noqa: E402
from tl3.ch5_retrieval import EMBEDDERS, BM25Retriever, Embedder, TfidfRetriever, save_index  # noqa: E402

OUT = model_dir("ch5")


def build_sparse(D):
    """Chỉ mục thưa: TF-IDF (âm tiết + bigram) và BM25 (k1 = 1,5; b = 0,75)."""
    t = {}
    t0 = time.perf_counter()
    tfidf = TfidfRetriever(D.texts)
    t["tfidf_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    bm25 = BM25Retriever(D.texts)
    t["bm25_s"] = time.perf_counter() - t0
    save_index(OUT / f"{D.name}_tfidf.pkl", tfidf)
    save_index(OUT / f"{D.name}_bm25.pkl", bm25)
    return t, int((tfidf.X.getnnz(axis=0) > 0).sum())


def build_dense(texts, name, key, batch_size=256):
    """Mã hoá toàn bộ tài liệu bằng bi-encoder, lưu float16 (~0,7 KB/tài liệu với e5-small)."""
    t0 = time.perf_counter()
    E = Embedder(key).encode(texts, kind="passage", batch_size=batch_size)
    np.save(OUT / f"{name}_emb_{key}.npy", E)
    Embedder._cache.clear()          # nhả mô hình trước khi nạp mô hình kế tiếp
    import torch
    torch.cuda.empty_cache()
    return time.perf_counter() - t0


def rdf_demo(D):
    """Xuất một phần KG chứng khoán sang RDF Turtle và chạy truy vấn SPARQL minh hoạ."""
    sub = D.df[(D.df.Date >= "2020-03-01") & (D.df.Date <= "2020-03-31")]
    from tl3.ch5_kg import market_triples
    rdf = to_rdf(market_triples(sub.reset_index(drop=True), "TradingDay", "did"))
    ttl = rdf.serialize(format="turtle")
    query = """PREFIX kg: <http://tl3.vn/kg/>
SELECT ?day ?close ?ret WHERE {
  ?day kg:in_month kg:Month_2020-03 ;
       kg:has_close ?close ;
       kg:has_return_pct ?ret .
  FILTER (?ret < -4.0)
} ORDER BY ?ret"""
    rows = [[str(r.day).rsplit("/", 1)[-1], float(r.close), float(r.ret)] for r in rdf.query(query)]
    return {"n_rdf_triples": len(rdf), "turtle_head": "\n".join(ttl.splitlines()[:16]), "sparql": query,
            "sparql_rows": rows}


def main(name):
    limit_threads(4)
    seed_all()
    t0 = time.perf_counter()
    D = Domain(name, build_graph_now=False)
    t_corpus = time.perf_counter() - t0
    times, vocab = build_sparse(D)          # chỉ mục thưa dựng trước đồ thị để đỉnh bộ nhớ thấp hơn
    t0 = time.perf_counter()
    D.build_graph()
    t_graph = time.perf_counter() - t0
    st = D.stats()
    lens = np.array([len(t.split()) for t in D.texts])
    report = {"n_docs": len(D.docs), "n_records": int(sum(d["kind"] == "record" for d in D.docs)),
              "n_summaries": int(sum(d["kind"] == "summary" for d in D.docs)), "n_rows": len(D.df),
              "n_triples": len(D.triples()), "graph": st, "vocab": vocab,
              "doc_len_mean": float(lens.mean()), "doc_len_max": int(lens.max()),
              "corpus_s": t_corpus, "build_kg_s": t_graph, **times,
              "doc_len_hist": np.bincount(np.minimum(lens, 120)).tolist()}
    with open(OUT / f"{name}_domain.pkl", "wb") as f:
        pickle.dump(D, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(name, report["n_docs"], st["nodes"], st["edges"], flush=True)
    if name == "stock":
        report["rdf_demo"] = rdf_demo(D)
    save_json(f"ch5_kb_{name}", report)    # lưu trước bước GPU: nếu mã hoá lỗi, chạy lại riêng "emb"
    texts = D.texts
    del D
    embed(name, texts, report)


def embed(name, texts, report):
    """Mã hoá kho bằng ba bi-encoder; bỏ qua mô hình đã có tệp .npy (chạy tiếp được sau khi lỗi bộ nhớ)."""
    big = len(texts) > 20000
    # kho nhỏ (~2.000 tài liệu) mã hoá trong vài giây, chạy như việc GPU nhẹ; kho 96.000 tài liệu giữ khoá GPU nặng
    ctx = heavy(f"ch5 embedding {name}") if big else nullcontext(cap(0.3))
    with ctx:
        for key in EMBEDDERS:
            if (OUT / f"{name}_emb_{key}.npy").exists() and f"emb_{key}_s" in report:
                continue
            report[f"emb_{key}_s"] = build_dense(texts, name, key, 256 if big else 32)
            print(name, key, round(report[f"emb_{key}_s"], 1), flush=True)
            save_json(f"ch5_kb_{name}", report)


def embed_only(name):
    from tl3.common import load_json
    with open(OUT / f"{name}_domain.pkl", "rb") as f:
        texts = pickle.load(f).texts
    embed(name, texts, load_json(f"ch5_kb_{name}"))


if __name__ == "__main__":
    embed_only(sys.argv[1]) if sys.argv[2:] == ["emb"] else main(sys.argv[1])
