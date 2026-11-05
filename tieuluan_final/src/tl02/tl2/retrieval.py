"""Các bộ truy xuất của RAG: thưa (TF-IDF, BM25), dày (bi-encoder), lai (RRF) và tái xếp hạng (cross-encoder)."""
import re

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer

from .common import EMBEDDERS, RERANKER, device, hf_path

TOKEN = re.compile(r"\w+", re.U)


def tokenize(text):
    """Tách âm tiết tiếng Việt viết thường; giữ số và ký hiệu đơn vị như mg, dl."""
    return TOKEN.findall(text.lower())


class TfidfRetriever:
    name = "TF-IDF"

    def __init__(self, chunks):
        self.chunks = chunks
        # âm tiết + cặp âm tiết liền kề: bigram xấp xỉ từ ghép tiếng Việt ("đường huyết", "tỷ giá")
        self.vec = TfidfVectorizer(tokenizer=tokenize, lowercase=False, ngram_range=(1, 2), sublinear_tf=True,
                                   token_pattern=None)
        self.X = self.vec.fit_transform([c.index_text() for c in chunks])

    def scores(self, query):
        q = self.vec.transform([query])
        return (self.X @ q.T).toarray().ravel()           # vector đã chuẩn hóa L2 nên tích vô hướng = cosin

    def search(self, query, k=5):
        return _topk(self.chunks, self.scores(query), k)


class BM25Retriever:
    name = "BM25"

    def __init__(self, chunks, k1=1.5, b=0.75):
        self.chunks = chunks
        self.bm25 = BM25Okapi([tokenize(c.index_text()) for c in chunks], k1=k1, b=b)

    def scores(self, query):
        return self.bm25.get_scores(tokenize(query))

    def search(self, query, k=5):
        return _topk(self.chunks, self.scores(query), k)


class Embedder:
    """Bọc SentenceTransformer; mô hình e5 cần tiền tố "query: " / "passage: "."""

    _cache = {}

    def __init__(self, key="e5-base", half=False):
        """half=True: trọng số fp16, dùng khi chung tiến trình với LLM (GPU qua WDDM tính cả vào commit của Windows)."""
        from sentence_transformers import SentenceTransformer
        self.key = key
        if (key, half) not in Embedder._cache:
            model = SentenceTransformer(hf_path(EMBEDDERS[key]), device=device())
            Embedder._cache[(key, half)] = model.half() if half else model
        self.model = Embedder._cache[(key, half)]
        self.is_e5 = key.startswith("e5")

    def encode(self, texts, kind="passage", batch_size=64):
        if self.is_e5:
            texts = [f"{kind}: {t}" for t in texts]
        return self.model.encode(texts, batch_size=batch_size, normalize_embeddings=True, convert_to_numpy=True,
                                 show_progress_bar=False)


class DenseRetriever:
    def __init__(self, chunks, embedder):
        self.chunks, self.emb = chunks, embedder
        self.name = f"Dense ({embedder.key})"
        self.E = embedder.encode([c.index_text() for c in chunks])

    def scores(self, query):
        return self.E @ self.emb.encode([query], kind="query")[0]

    def search(self, query, k=5):
        return _topk(self.chunks, self.scores(query), k)


class HybridRRF:
    """Reciprocal Rank Fusion: s(d) = Σ_r 1 / (k + rank_r(d)), không cần chuẩn hóa thang điểm."""

    def __init__(self, retrievers, k=60):
        self.retrievers, self.k = retrievers, k
        self.chunks = retrievers[0].chunks
        self.name = "Hybrid RRF (" + " + ".join(r.name for r in retrievers) + ")"

    def scores(self, query):
        fused = np.zeros(len(self.chunks))
        for r in self.retrievers:
            rank = np.empty(len(self.chunks), dtype=int)
            rank[np.argsort(-r.scores(query))] = np.arange(1, len(self.chunks) + 1)
            fused += 1.0 / (self.k + rank)
        return fused

    def search(self, query, k=5):
        return _topk(self.chunks, self.scores(query), k)


class Reranker:
    """Cross-encoder đọc đồng thời (câu hỏi, đoạn văn) và chấm mức liên quan; chậm nhưng chính xác."""

    _model = None

    def __init__(self):
        from sentence_transformers import CrossEncoder
        if Reranker._model is None:
            Reranker._model = CrossEncoder(hf_path(RERANKER), device=device(), max_length=512,
                                           model_kwargs={"torch_dtype": "bfloat16"})
        self.model = Reranker._model

    def rerank(self, query, hits, k=5):
        if not hits:
            return hits
        s = self.model.predict([(query, c.index_text()) for c, _ in hits], batch_size=32, show_progress_bar=False)
        order = np.argsort(-s)[:k]
        return [(hits[i][0], float(s[i])) for i in order]


class RerankPipeline:
    """Truy xuất ứng viên rộng (top-n) bằng retriever nhanh rồi tái xếp hạng bằng cross-encoder."""

    def __init__(self, first_stage, reranker, n_candidates=30):
        self.first, self.rr, self.n = first_stage, reranker, n_candidates
        self.chunks = first_stage.chunks
        self.name = f"{first_stage.name} + Rerank"

    def search(self, query, k=5):
        return self.rr.rerank(query, self.first.search(query, self.n), k)


def _topk(chunks, scores, k):
    idx = np.argsort(-scores)[:k]
    return [(chunks[i], float(scores[i])) for i in idx]


# ---------------------------------------------------------------- độ đo xếp hạng
def is_relevant(chunk, answer):
    return answer in chunk.text


def rank_metrics(ranked, answer, n_relevant, ks=(1, 3, 5, 10)):
    """ranked: list[Chunk] theo thứ tự; n_relevant: số chunk trong kho chứa trọn câu trả lời.

    Trả hit@k, MRR và nDCG@10 nhị phân (nhiều chunk chồng lấn có thể cùng liên quan).
    """
    rel = [is_relevant(c, answer) for c in ranked]
    out = {f"hit@{k}": float(any(rel[:k])) for k in ks}
    first = next((i for i, r in enumerate(rel) if r), None)
    out["mrr"] = 0.0 if first is None else 1.0 / (first + 1)
    dcg = sum(r / np.log2(i + 2) for i, r in enumerate(rel[:10]))
    n_rel = max(1, min(10, n_relevant))
    idcg = sum(1 / np.log2(i + 2) for i in range(n_rel))
    out["ndcg@10"] = float(dcg / idcg) if any(rel[:10]) else 0.0
    return out
