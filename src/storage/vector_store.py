"""
Vector store trên pgvector: hybrid search (vector + full-text) gộp bằng Reciprocal Rank Fusion.

Full-text giúp bắt đúng từ khóa chuyên môn (HbA1c, SGLT2, eGFR...) mà embedding câu
đôi khi bỏ lỡ; vector giúp khớp theo nghĩa ("khát nước nhiều" ~ "uống nhiều").
"""

from typing import Any, Dict, List

from sqlalchemy import text
from sqlalchemy.orm import Session

RRF_K = 60

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

# websearch_to_tsquery dùng AND giữa các từ — với câu hỏi tự nhiên thì quá chặt, nên đổi sang OR
FTS_OR_SQL = text("SELECT string_agg(word, ' OR ') FROM unnest(string_to_array(:q, ' ')) AS word WHERE length(word) > 1")


class PgVectorStore:
    def __init__(self, session: Session):
        self.session = session

    def hybrid_search(self, query_text: str, query_vec: List[float], k: int = 5, pool: int = 30) -> List[Dict[str, Any]]:
        or_query = self.session.execute(FTS_OR_SQL, {"q": _clean(query_text)}).scalar() or ""
        rows = self.session.execute(HYBRID_SQL, {
            "qvec": str(list(query_vec)), "qtext": or_query, "pool": pool, "k": k, "rrf_k": RRF_K,
        }).mappings().all()
        return [dict(r) for r in rows]

    def chunks_by_ids(self, ids: List[int]) -> List[Dict[str, Any]]:
        if not ids:
            return []
        rows = self.session.execute(text("""
            SELECT c.id, c.document_id, c.heading, c.content, d.title, d.source, d.url
            FROM kb_chunks c JOIN kb_documents d ON d.id = c.document_id WHERE c.id = ANY(:ids)
        """), {"ids": ids}).mappings().all()
        return [dict(r) for r in rows]


def _clean(q: str) -> str:
    keep = [ch if ch.isalnum() or ch.isspace() else " " for ch in q]
    return " ".join("".join(keep).split())
