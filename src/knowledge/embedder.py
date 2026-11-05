"""
Embedding văn bản cho RAG.

- FastEmbedEmbedder: mô hình đa ngôn ngữ chạy ONNX trên CPU (không cần PyTorch/GPU).
- HashingEmbedder: dự phòng offline (không tải được mô hình) — vector băm n-gram, chỉ khớp từ vựng.

Cả hai cho ra vector cùng số chiều (EMBEDDING_DIM) để dùng chung collection Qdrant (hoặc cột pgvector);
`name` được lưu cùng mỗi chunk và ingest sẽ embed lại khi đổi backend.
"""

import logging
from functools import lru_cache
from typing import List

import numpy as np

from src.config import settings

log = logging.getLogger(__name__)


class FastEmbedEmbedder:
    def __init__(self, model_name: str, dim: int):
        from fastembed import TextEmbedding
        self.model = TextEmbedding(model_name=model_name)
        self.name = model_name
        self.dim = dim

    def embed(self, texts: List[str]) -> List[List[float]]:
        vectors = [v.tolist() for v in self.model.embed(texts)]
        if vectors and len(vectors[0]) != self.dim:
            raise ValueError(f"Mô hình {self.name} trả về {len(vectors[0])} chiều, cấu hình EMBEDDING_DIM={self.dim}")
        return vectors


class HashingEmbedder:
    def __init__(self, dim: int):
        from sklearn.feature_extraction.text import HashingVectorizer
        self.vectorizer = HashingVectorizer(n_features=dim, analyzer="char_wb", ngram_range=(2, 4),
                                            norm="l2", alternate_sign=False, strip_accents="unicode")
        self.name = f"hashing-char-2-4-{dim}"
        self.dim = dim

    def embed(self, texts: List[str]) -> List[List[float]]:
        matrix = self.vectorizer.transform([t.lower() for t in texts])
        return np.asarray(matrix.todense(), dtype=np.float32).tolist()


@lru_cache(maxsize=1)
def get_embedder():
    if settings.embedding_backend == "fastembed":
        try:
            return FastEmbedEmbedder(settings.embedding_model, settings.embedding_dim)
        except Exception as exc:  # noqa: BLE001
            log.warning("Không tải được mô hình embedding (%s), dùng HashingEmbedder", exc)
    return HashingEmbedder(settings.embedding_dim)
