"""
Kết nối PostgreSQL (SQLAlchemy 2 + psycopg 3) và khởi tạo schema.

Postgres đảm nhận cả ba vai trò trên sơ đồ: DB quan hệ, Vector store (pgvector)
và Knowledge Graph (Apache AGE).
"""

from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import create_engine, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from src.config import settings

engine = create_engine(settings.database_url, pool_pre_ping=True, pool_size=10, max_overflow=10)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


@contextmanager
def session_scope() -> Iterator[Session]:
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


PRE_SCHEMA_SQL = [
    "CREATE EXTENSION IF NOT EXISTS vector",
    "CREATE EXTENSION IF NOT EXISTS unaccent",
    # unaccent() không IMMUTABLE nên không dùng được trong generated column; bọc lại để dùng cho full-text
    """
    CREATE OR REPLACE FUNCTION immutable_unaccent(text) RETURNS text
    LANGUAGE sql IMMUTABLE PARALLEL SAFE STRICT
    AS $$ SELECT public.unaccent('public.unaccent'::regdictionary, $1) $$
    """,
]

POST_SCHEMA_SQL = [
    "CREATE INDEX IF NOT EXISTS ix_kb_chunks_embedding ON kb_chunks USING hnsw (embedding vector_cosine_ops)",
    "CREATE INDEX IF NOT EXISTS ix_kb_chunks_tsv ON kb_chunks USING gin (tsv)",
]


def init_db() -> None:
    from src.storage import models  # noqa: F401 — đăng ký các bảng vào metadata

    with engine.begin() as conn:
        for stmt in PRE_SCHEMA_SQL:
            conn.execute(text(stmt))
    Base.metadata.create_all(engine)
    with engine.begin() as conn:
        for stmt in POST_SCHEMA_SQL:
            conn.execute(text(stmt))
