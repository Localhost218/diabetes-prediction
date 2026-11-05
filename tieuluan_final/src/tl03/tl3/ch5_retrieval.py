"""Chương 5 — các bộ truy xuất của RAG trên kho tài liệu {id, text}: thưa (TF-IDF, BM25), dày (bi-encoder e5),
đồ thị (liên kết thực thể -> nút KG -> tài liệu), lai (RRF) và tái xếp hạng (cross-encoder bge-reranker).

Mọi bộ truy xuất có cùng giao diện scores(q) -> np.ndarray độ dài N và search(q, k) -> [(chỉ số, điểm)].
"""
import os
import pickle
import re
import time

# sentence-transformers/transformers mặc định dò cả TensorFlow: nạp TF tốn ~1 GB RAM mà chương này không dùng.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer

from .common import device, hf_path

TOKEN = re.compile(r"\w+", re.U)
EMBEDDERS = {"minilm": "paraphrase-multilingual-MiniLM-L12-v2", "e5-small": "multilingual-e5-small",
             "e5-base": "multilingual-e5-base"}
RERANKER = "bge-reranker-v2-m3-bf16"


def tokenize(text):
    """Tách âm tiết tiếng Việt viết thường; số "25,19" thành hai token "25", "19" như nhau ở tài liệu và câu hỏi."""
    return TOKEN.findall(text.lower())


def topk(scores, k):
    """k chỉ số điểm cao nhất (argpartition để nhanh trên 100.000 tài liệu)."""
    k = min(k, len(scores))
    idx = np.argpartition(-scores, k - 1)[:k]
    idx = idx[np.argsort(-scores[idx], kind="stable")]
    return [(int(i), float(scores[i])) for i in idx]


class TfidfRetriever:
    name = "TF-IDF"

    def __init__(self, texts, n_features=2 ** 20):
        # âm tiết + cặp âm tiết liền kề: bigram xấp xỉ từ ghép tiếng Việt ("đường huyết", "đóng cửa").
        # Băm đặc trưng (hashing trick) thay cho từ điển: không giữ bảng từ vựng nên đủ bộ nhớ cho 96.000 tài liệu.
        self.hasher = HashingVectorizer(tokenizer=tokenize, lowercase=False, ngram_range=(1, 2), token_pattern=None,
                                        n_features=n_features, alternate_sign=False, norm=None, dtype=np.float32)
        self.tfidf = TfidfTransformer(sublinear_tf=True)
        counts = self.hasher.transform(texts)
        self.X = self.tfidf.fit_transform(counts).astype(np.float32)

    def scores(self, q):
        qv = self.tfidf.transform(self.hasher.transform([q]))
        return (self.X @ qv.T).toarray().ravel()   # vector chuẩn hoá L2 -> tích vô hướng = cosin

    def search(self, q, k=10):
        return topk(self.scores(q), k)


class BM25Retriever:
    name = "BM25"

    def __init__(self, texts, k1=1.5, b=0.75):
        self.bm25 = BM25Okapi([tokenize(t) for t in texts], k1=k1, b=b)

    def scores(self, q):
        return np.asarray(self.bm25.get_scores(tokenize(q)), dtype=np.float32)

    def search(self, q, k=10):
        return topk(self.scores(q), k)


class Embedder:
    """Bọc SentenceTransformer; họ e5 cần tiền tố "query: " / "passage: " khi mã hoá."""
    _cache = {}

    def __init__(self, key="e5-small", on=None):
        from sentence_transformers import SentenceTransformer
        self.key = key
        on = on or device()
        if (key, on) not in Embedder._cache:
            Embedder._cache[(key, on)] = SentenceTransformer(hf_path(EMBEDDERS[key]), device=on)
        self.model = Embedder._cache[(key, on)]
        self.is_e5 = key.startswith("e5")

    def encode(self, texts, kind="passage", batch_size=256):
        if self.is_e5:
            texts = [f"{kind}: {t}" for t in texts]
        return self.model.encode(texts, batch_size=batch_size, normalize_embeddings=True, convert_to_numpy=True,
                                 show_progress_bar=False).astype(np.float16)


class DenseRetriever:
    """Bi-encoder: tài liệu mã hoá trước (lưu .npy float16), câu hỏi mã hoá lúc truy vấn; điểm = cosin."""

    def __init__(self, E, embedder):
        self.emb = embedder
        self.name = f"Dense ({embedder.key})"
        # đổi sang float32 một lần lúc nạp, không cấp phát lại bản ~300 MB mỗi truy vấn
        self.E = E.astype(np.float32)

    def scores(self, q):
        return self.E @ self.emb.encode([q], kind="query")[0].astype(np.float32)

    def search(self, q, k=10):
        return topk(self.scores(q), k)


class GraphRetriever:
    """Truy xuất đồ thị: hàm link(q) của miền trả danh sách id tài liệu ứng với nút KG khớp thực thể trong câu hỏi.
    Điểm = 1 cho tài liệu khớp (giảm dần theo thứ tự khớp), 0 cho phần còn lại."""
    name = "Graph"

    def __init__(self, ids, link):
        self.pos = {d: i for i, d in enumerate(ids)}
        self.n, self.link = len(ids), link

    def scores(self, q):
        s = np.zeros(self.n, dtype=np.float32)
        for r, d in enumerate(self.link(q)):
            if d in self.pos:
                s[self.pos[d]] = max(s[self.pos[d]], 1.0 - 1e-3 * r)
        return s

    def search(self, q, k=10):
        return [(i, v) for i, v in topk(self.scores(q), k) if v > 0]


class HybridRRF:
    """Reciprocal Rank Fusion: s(d) = Σ_r w_r / (k + rank_r(d)); chỉ cộng các tài liệu trong top-depth mỗi nguồn."""

    def __init__(self, retrievers, k=60, depth=100, weights=None):
        self.rs, self.k, self.depth = retrievers, k, depth
        self.w = weights or [1.0] * len(retrievers)
        self.name = "RRF(" + " + ".join(r.name for r in retrievers) + ")"

    def search(self, q, k=10):
        fused = {}
        for r, w in zip(self.rs, self.w):
            for rank, (i, _) in enumerate(r.search(q, self.depth), 1):
                fused[i] = fused.get(i, 0.0) + w / (self.k + rank)
        return sorted(fused.items(), key=lambda x: -x[1])[:k]


class Reranker:
    """Cross-encoder đọc đồng thời (câu hỏi, tài liệu) và chấm mức liên quan; chậm nhưng chính xác hơn bi-encoder."""
    _model = None

    def __init__(self):
        from sentence_transformers import CrossEncoder
        if Reranker._model is None:
            Reranker._model = CrossEncoder(hf_path(RERANKER), device=device(), max_length=384,
                                           model_kwargs={"torch_dtype": "bfloat16"})
        self.model = Reranker._model

    def rerank(self, q, hits, texts, k=10):
        if not hits:
            return hits
        s = self.model.predict([(q, texts[i]) for i, _ in hits], batch_size=32, show_progress_bar=False)
        order = np.argsort(-np.asarray(s, dtype=np.float32))[:k]
        return [(hits[j][0], float(s[j])) for j in order]


class RerankPipeline:
    """Giai đoạn 1 lấy rộng n ứng viên bằng bộ truy xuất nhanh; giai đoạn 2 cross-encoder xếp lại."""

    def __init__(self, first, reranker, texts, n=20):
        self.first, self.rr, self.texts, self.n = first, reranker, texts, n
        self.name = f"{first.name} + Rerank"

    def search(self, q, k=10):
        return self.rr.rerank(q, self.first.search(q, self.n), self.texts, k)


# ---------------------------------------------------------------- độ đo xếp hạng
def rank_metrics(ranked_ids, gold, ks=(1, 3, 5, 10)):
    """ranked_ids: id tài liệu theo thứ tự; gold: tập id đúng. Trả Recall@k, MRR@10, nDCG@10 (liên quan nhị phân)."""
    rel = [d in gold for d in ranked_ids[:10]]
    out = {f"r@{k}": float(sum(rel[:k]) / len(gold)) for k in ks}
    first = next((i for i, r in enumerate(rel) if r), None)
    out["mrr"] = 0.0 if first is None else 1.0 / (first + 1)
    dcg = sum(r / np.log2(i + 2) for i, r in enumerate(rel))
    idcg = sum(1 / np.log2(i + 2) for i in range(min(10, len(gold))))
    out["ndcg"] = float(dcg / idcg)
    return out


def timed_search(retriever, q, k=10):
    t0 = time.perf_counter()
    hits = retriever.search(q, k)
    return hits, 1000 * (time.perf_counter() - t0)


# ---------------------------------------------------------------- lưu / nạp chỉ mục
def save_index(path, obj):
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_index(path):
    with open(path, "rb") as f:
        return pickle.load(f)
