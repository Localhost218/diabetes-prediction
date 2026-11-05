"""
Vector store cho RAG: hybrid search (vector + full-text) gộp bằng Reciprocal Rank Fusion.

Full-text giúp bắt đúng từ khóa chuyên môn (HbA1c, SGLT2, eGFR...) mà embedding câu
đôi khi bỏ lỡ; vector giúp khớp theo nghĩa ("khát nước nhiều" ~ "uống nhiều").

- QdrantVectorStore (mặc định): vector ở Qdrant (point id = kb_chunks.id), full-text bằng FTS5
  của SQLite, nội dung đoạn văn đọc từ bảng kb_chunks; RRF gộp ở phía Python.
- PgVectorStore: toàn bộ trong PostgreSQL (pgvector + tsvector), RRF trong một câu SQL.
Chọn bằng VECTOR_BACKEND=qdrant|pgvector.
"""

import logging
from functools import lru_cache
from typing import Any, Dict, List, Protocol, Sequence

from sqlalchemy import bindparam, text
from sqlalchemy.orm import Session

from src.config import settings

log = logging.getLogger(__name__)

RRF_K = 60

CHUNK_COLUMNS = "c.id, c.document_id, c.chunk_index, c.heading, c.content, d.title, d.source, d.url"


class VectorStore(Protocol):
    def index_chunks(self, chunks: Sequence[Any], vectors: List[List[float]]) -> None: ...
    def delete_document(self, document_id: str) -> None: ...
    def has_document(self, document_id: str) -> bool: ...
    def vector_search(self, query_vec: List[float], k: int = 5) -> List[Dict[str, Any]]: ...
    def fulltext_search(self, query_text: str, k: int = 5) -> List[Dict[str, Any]]: ...
    def hybrid_search(self, query_text: str, query_vec: List[float], k: int = 5, pool: int = 30) -> List[Dict[str, Any]]: ...
    def chunks_by_ids(self, ids: List[int]) -> List[Dict[str, Any]]: ...


def _clean(q: str) -> str:
    keep = [ch if ch.isalnum() or ch.isspace() else " " for ch in q]
    return " ".join("".join(keep).split())


# Từ hư / từ hỏi phổ biến: ở truy vấn OR chúng khớp gần như mọi đoạn và làm nhiễu bm25
VI_STOPWORDS = {
    "bao", "nhiêu", "thì", "là", "gì", "nào", "sao", "có", "không", "của", "và", "các", "những", "một",
    "cho", "với", "được", "bị", "khi", "nếu", "này", "đó", "thế", "như", "tôi", "mình", "em", "anh", "chị",
    "nên", "phải", "cần", "hay", "hoặc", "để", "vậy", "ạ", "à", "nhé", "lắm", "rất", "ở", "trong", "về",
}


def _words(q: str) -> List[str]:
    return [w for w in _clean(q).split() if len(w) > 1 and w.lower() not in VI_STOPWORDS]


def _chunks_by_ids(session: Session, ids: List[int]) -> Dict[int, Dict[str, Any]]:
    if not ids:
        return {}
    rows = session.execute(
        text(f"SELECT {CHUNK_COLUMNS} FROM kb_chunks c JOIN kb_documents d ON d.id = c.document_id "
             "WHERE c.id IN :ids").bindparams(bindparam("ids", expanding=True)),
        {"ids": list(ids)}).mappings().all()
    return {r["id"]: dict(r) for r in rows}


def _rrf(vec: List[tuple], fts: List[tuple], k: int) -> List[Dict[str, Any]]:
    """vec: [(id, cosine)], fts: [(id, score)] — đã sắp theo hạng. Trả [{id, score, cosine, vec_rank, fts_rank}]."""
    fused: Dict[int, Dict[str, Any]] = {}
    for rank, (cid, cos) in enumerate(vec, 1):
        fused[cid] = {"id": cid, "score": 1.0 / (RRF_K + rank), "cosine": cos, "vec_rank": rank, "fts_rank": None}
    for rank, (cid, _) in enumerate(fts, 1):
        f = fused.setdefault(cid, {"id": cid, "score": 0.0, "cosine": None, "vec_rank": None, "fts_rank": None})
        f["score"] += 1.0 / (RRF_K + rank)
        f["fts_rank"] = rank
    return sorted(fused.values(), key=lambda f: f["score"], reverse=True)[:k]


# ======================================================================= Qdrant + SQLite FTS5
@lru_cache(maxsize=1)
def qdrant_client():
    from qdrant_client import QdrantClient
    if settings.qdrant_url:
        return QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key or None, timeout=30)
    settings.qdrant_path.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(settings.qdrant_path))  # chế độ nhúng, không cần server


class QdrantVectorStore:
    def __init__(self, session: Session):
        from qdrant_client import models
        self.session = session
        self.client = qdrant_client()
        self.collection = settings.qdrant_collection
        self.m = models
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                self.collection,
                vectors_config=models.VectorParams(size=settings.embedding_dim, distance=models.Distance.COSINE))
            self.client.create_payload_index(self.collection, "document_id", models.PayloadSchemaType.KEYWORD)

    def _doc_filter(self, document_id: str):
        return self.m.Filter(must=[self.m.FieldCondition(key="document_id", match=self.m.MatchValue(value=document_id))])

    def index_chunks(self, chunks: Sequence[Any], vectors: List[List[float]]) -> None:
        points = [self.m.PointStruct(id=int(c.id), vector=list(v),
                                     payload={"document_id": c.document_id, "heading": c.heading,
                                              "embedding_model": c.embedding_model})
                  for c, v in zip(chunks, vectors)]
        if points:
            self.client.upsert(self.collection, points=points, wait=True)

    def delete_document(self, document_id: str) -> None:
        self.client.delete(self.collection, points_selector=self.m.FilterSelector(filter=self._doc_filter(document_id)),
                           wait=True)

    def has_document(self, document_id: str) -> bool:
        return self.client.count(self.collection, count_filter=self._doc_filter(document_id), exact=True).count > 0

    def _vector_ranked(self, query_vec: List[float], limit: int) -> List[tuple]:
        hits = self.client.query_points(self.collection, query=list(query_vec), limit=limit, with_payload=False).points
        return [(int(h.id), float(h.score)) for h in hits]

    def _fts_ranked(self, query_text: str, limit: int) -> List[tuple]:
        words = _words(query_text)
        if not words:
            return []
        # Câu hỏi tự nhiên: nối các từ bằng OR (AND quá chặt); bm25() càng nhỏ càng khớp
        match = " OR ".join('"' + w.replace('"', "") + '"' for w in words)
        # trọng số cột: tiêu đề mục ×3 so với nội dung
        rows = self.session.execute(text(
            "SELECT rowid, -bm25(kb_chunks_fts, 3.0, 1.0) AS score FROM kb_chunks_fts WHERE kb_chunks_fts MATCH :q "
            "ORDER BY bm25(kb_chunks_fts, 3.0, 1.0) LIMIT :n"), {"q": match, "n": limit}).all()
        return [(int(r[0]), float(r[1])) for r in rows]

    def _attach(self, ranked: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        rows = _chunks_by_ids(self.session, [r["id"] for r in ranked])
        # bỏ point mồ côi (chunk đã xóa khỏi SQLite nhưng còn trong Qdrant)
        return [{**rows[r["id"]], **r} for r in ranked if r["id"] in rows]

    def vector_search(self, query_vec: List[float], k: int = 5) -> List[Dict[str, Any]]:
        return self._attach([{"id": i, "score": s} for i, s in self._vector_ranked(query_vec, k)])

    def fulltext_search(self, query_text: str, k: int = 5) -> List[Dict[str, Any]]:
        return self._attach([{"id": i, "score": s} for i, s in self._fts_ranked(query_text, k)])

    def hybrid_search(self, query_text: str, query_vec: List[float], k: int = 5, pool: int = 30) -> List[Dict[str, Any]]:
        fused = _rrf(self._vector_ranked(query_vec, pool), self._fts_ranked(query_text, pool), k + 5)
        return self._attach(fused)[:k]

    def chunks_by_ids(self, ids: List[int]) -> List[Dict[str, Any]]:
        rows = _chunks_by_ids(self.session, ids)
        return [rows[i] for i in ids if i in rows]


# ======================================================================= PostgreSQL + pgvector
HYBRID_SQL = text("""
WITH vec AS (
    SELECT id, row_number() OVER (ORDER BY embedding <=> CAST(:qvec AS vector)) AS rank,
           1 - (embedding <=> CAST(:qvec AS vector)) AS cosine
    FROM kb_chunks
    ORDER BY embedding <=> CAST(:qvec AS vector)
    LIMIT :pool
),
fts AS (
    SELECT id, row_number() OVER (ORDER BY ts_rank_cd(tsv, q) DESC) AS rank
    FROM kb_chunks, websearch_to_tsquery('simple', immutable_unaccent(:qtext)) AS q
    WHERE tsv @@ q
    ORDER BY ts_rank_cd(tsv, q) DESC
    LIMIT :pool
),
fused AS (
    SELECT coalesce(vec.id, fts.id) AS id,
           coalesce(1.0 / (:rrf_k + vec.rank), 0) + coalesce(1.0 / (:rrf_k + fts.rank), 0) AS score,
           vec.cosine, vec.rank AS vec_rank, fts.rank AS fts_rank
    FROM vec FULL OUTER JOIN fts ON vec.id = fts.id
)
SELECT c.id, c.document_id, c.chunk_index, c.heading, c.content,
       d.title, d.source, d.url,
       f.score, f.cosine, f.vec_rank, f.fts_rank
FROM fused f
JOIN kb_chunks c ON c.id = f.id
JOIN kb_documents d ON d.id = c.document_id
ORDER BY f.score DESC
LIMIT :k
""")

PG_VECTOR_SQL = text(f"""SELECT {CHUNK_COLUMNS}, 1 - (c.embedding <=> CAST(:qvec AS vector)) AS score
                     FROM kb_chunks c JOIN kb_documents d ON d.id = c.document_id
                     ORDER BY c.embedding <=> CAST(:qvec AS vector) LIMIT :k""")
PG_FTS_SQL = text(f"""SELECT {CHUNK_COLUMNS}, ts_rank_cd(c.tsv, q) AS score
                  FROM kb_chunks c JOIN kb_documents d ON d.id = c.document_id,
                       websearch_to_tsquery('simple', immutable_unaccent(:q)) AS q
                  WHERE c.tsv @@ q ORDER BY score DESC LIMIT :k""")


class PgVectorStore:
    def __init__(self, session: Session):
        self.session = session

    def index_chunks(self, chunks: Sequence[Any], vectors: List[List[float]]) -> None:
        for c, v in zip(chunks, vectors):
            c.embedding = v

    def delete_document(self, document_id: str) -> None:
        pass  # vector nằm cùng hàng kb_chunks, bị xóa theo

    def has_document(self, document_id: str) -> bool:
        return True

    def vector_search(self, query_vec: List[float], k: int = 5) -> List[Dict[str, Any]]:
        return [dict(r) for r in self.session.execute(PG_VECTOR_SQL, {"qvec": str(list(query_vec)), "k": k}).mappings()]

    def fulltext_search(self, query_text: str, k: int = 5) -> List[Dict[str, Any]]:
        return [dict(r) for r in self.session.execute(PG_FTS_SQL, {"q": " OR ".join(_words(query_text)), "k": k}).mappings()]

    def hybrid_search(self, query_text: str, query_vec: List[float], k: int = 5, pool: int = 30) -> List[Dict[str, Any]]:
        # websearch_to_tsquery dùng AND giữa các từ — với câu hỏi tự nhiên thì quá chặt, nên đổi sang OR
        rows = self.session.execute(HYBRID_SQL, {
            "qvec": str(list(query_vec)), "qtext": " OR ".join(_words(query_text)), "pool": pool, "k": k,
            "rrf_k": RRF_K,
        }).mappings().all()
        return [dict(r) for r in rows]

    def chunks_by_ids(self, ids: List[int]) -> List[Dict[str, Any]]:
        rows = _chunks_by_ids(self.session, ids)
        return [rows[i] for i in ids if i in rows]


def get_vector_store(session: Session) -> VectorStore:
    return PgVectorStore(session) if settings.vector_backend == "pgvector" else QdrantVectorStore(session)
