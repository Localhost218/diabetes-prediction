"""Chương 5 — tác tử hỏi đáp: chọn công cụ theo câu hỏi rồi trả lời có căn cứ.

Công cụ:
  kg_lookup     duyệt đồ thị tri thức tới nút ngày/hồ sơ, đọc thuộc tính (chính xác tuyệt đối)
  kg_aggregate  giao các nút khái niệm (giới, nhóm tuổi, tăng huyết áp…) rồi tính tỷ lệ trên tập hồ sơ
  pandas_calc   tính thống kê theo tháng/năm trên bảng gốc
  rag_search    truy xuất lai BM25 + dense + đồ thị (RRF), tái xếp hạng cross-encoder, LLM cục bộ tổng hợp
Bộ định tuyến là luật tất định (có thực thể cụ thể -> công cụ chính xác; câu mở -> RAG) để vết suy luận tái lập
được; LLM chỉ dùng ở bước tổng hợp ngôn ngữ, đúng như kiến trúc "agent chọn công cụ" của lucnvm.
"""
import pickle
import time

import numpy as np

from .common import model_dir
from .ch5_retrieval import (BM25Retriever, DenseRetriever, Embedder, GraphRetriever, HybridRRF, Reranker,
                            RerankPipeline, load_index)
from .llm import SYSTEM_GROUNDED, LocalLLM, grounded_prompt

MODELS = model_dir("ch5")
SYSTEM_PLAIN = ("Bạn là trợ lý tiếng Việt. Trả lời ngắn gọn câu hỏi của người dùng bằng hiểu biết của bạn; "
                "nếu không biết số liệu chính xác thì nói rõ.")


def load_domain(name):
    with open(MODELS / f"{name}_domain.pkl", "rb") as f:
        return pickle.load(f)


class Retrievers:
    """Nạp các chỉ mục đã dựng sẵn của một miền và lắp các bộ truy xuất dùng trong đánh giá / tác tử."""

    def __init__(self, D, dense_key="e5-small", rerank=True):
        self.D = D
        self.bm25: BM25Retriever = load_index(MODELS / f"{D.name}_bm25.pkl")
        self.dense = DenseRetriever(np.load(MODELS / f"{D.name}_emb_{dense_key}.npy"), Embedder(dense_key, "cpu"))
        self.graph = GraphRetriever(D.ids, D.link)
        self.hybrid = HybridRRF([self.bm25, self.dense])
        self.hybrid_graph = HybridRRF([self.bm25, self.dense, self.graph], weights=[1.0, 1.0, 2.0])
        self.rerank = RerankPipeline(self.hybrid_graph, Reranker(), D.texts, n=20) if rerank else None


class Ch5Agent:
    def __init__(self, D, R, llm=None):
        self.D, self.R = D, R
        self._llm = llm

    @property
    def llm(self):
        if self._llm is None:
            self._llm = LocalLLM(max_new_tokens=220)
        return self._llm

    # ------------------------------------------------------------ các hệ so sánh
    def llm_only(self, q):
        t0 = time.perf_counter()
        ans, ntok, _ = self.llm.chat(q, system=SYSTEM_PLAIN)
        return {"answer": ans, "sources": [], "tools": ["llm"], "trace": [], "latency_ms": 1000 * (time.perf_counter() - t0)}

    def rag(self, q, retriever, k=5, trace=None):
        """Truy xuất k tài liệu, đánh số [1..k] làm bằng chứng, LLM sinh câu trả lời có trích dẫn."""
        t0 = time.perf_counter()
        hits = retriever.search(q, k)
        t_ret = 1000 * (time.perf_counter() - t0)
        ev = [(self.D.ids[i], self.D.texts[i]) for i, _ in hits]
        if trace is not None:
            trace.append(("action", f"rag_search(\"{q[:60]}…\", k={k}) — {retriever.name}", round(t_ret)))
            trace.append(("obs", "; ".join(f"[{j}] {d}" for j, (d, _) in enumerate(ev, 1))))
        ans, ntok, t_gen = self.llm.chat(grounded_prompt(q, ev), system=SYSTEM_GROUNDED)
        return {"answer": ans, "sources": [d for d, _ in ev], "scores": [round(s, 4) for _, s in hits],
                "tools": ["rag_search", "llm"],
                "trace": trace or [], "latency_ms": 1000 * (time.perf_counter() - t0), "retrieval_ms": t_ret,
                "gen_tokens": ntok}

    def naive_rag(self, q):
        return self.rag(q, self.R.dense)

    def advanced_rag(self, q):
        return self.rag(q, self.R.rerank or self.R.hybrid_graph)

    # ------------------------------------------------------------ tác tử định tuyến
    def plan(self, q):
        """Thought: nhận diện thực thể -> chọn công cụ. Trả (tên công cụ, lý do)."""
        ids = self.D.link(q)
        if ids:
            kinds = sorted({("tổng hợp" if i.startswith(("S-", "M-")) else "bản ghi") for i in ids})
            return "exact", f"Câu hỏi nêu thực thể cụ thể ({', '.join(ids[:3])}{'…' if len(ids) > 3 else ''}; " \
                            f"loại {', '.join(kinds)}) → dùng công cụ tra cứu chính xác trên KG/bảng."
        return "rag", "Không thấy ngày, giai đoạn, nhóm hay hồ sơ cụ thể → câu hỏi mở, dùng RAG lai + LLM tổng hợp."

    def run(self, q, mode="auto"):
        t0 = time.perf_counter()
        trace = [("question", q)]
        route, why = ("rag", "Người dùng chọn chế độ RAG.") if mode == "rag" else self.plan(q)
        if mode == "graph":
            route, why = "exact", "Người dùng chọn chế độ đồ thị."
        trace.append(("thought", why))
        if route == "exact":
            t1 = time.perf_counter()
            f = self.D.facts(q)
            if f is not None:
                trace.append(("action", f"{f['tool']}(\"{q[:60]}…\")", round(1000 * (time.perf_counter() - t1))))
                trace.append(("obs", " | ".join(f["evidence"][:8]) or "(không có bộ ba)"))
                trace.append(("answer", f["answer"]))
                return {"answer": f["answer"], "sources": self.D.link(q)[:5], "tools": [f["tool"]], "trace": trace,
                        "values": f["values"], "latency_ms": 1000 * (time.perf_counter() - t0)}
            trace.append(("thought", "Công cụ chính xác không trả lời được → chuyển sang RAG."))
        out = self.rag(q, self.R.rerank or self.R.hybrid_graph, trace=trace)
        trace.append(("answer", out["answer"]))
        out["latency_ms"] = 1000 * (time.perf_counter() - t0)
        return out
