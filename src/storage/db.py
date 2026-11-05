"""
Kết nối DB quan hệ (SQLAlchemy 2) và khởi tạo schema.

- SQLite (mặc định): bảng hội thoại, hồ sơ, audit trail, nội dung KB; full-text bằng FTS5.
  Vector nằm ở Qdrant, Knowledge Graph nằm ở Neo4j.
- PostgreSQL (tùy chọn): DB quan hệ + pgvector + Apache AGE trong cùng một máy chủ.
"""

from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from src.config import settings


def _make_engine():
    if settings.is_postgres:
        return create_engine(settings.database_url, pool_pre_ping=True, pool_size=10, max_overflow=10)
    if settings.database_url.startswith("sqlite:///"):
        from pathlib import Path
        Path(settings.database_url.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
    eng = create_engine(settings.database_url, connect_args={"check_same_thread": False, "timeout": 30})

    @event.listens_for(eng, "connect")
    def _sqlite_pragmas(dbapi_conn, _):
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA foreign_keys = ON")    # để ON DELETE CASCADE có hiệu lực
        cur.execute("PRAGMA journal_mode = WAL")   # đọc song song khi đang ghi
        cur.execute("PRAGMA synchronous = NORMAL")
        cur.close()

    return eng


engine = _make_engine()
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


PG_PRE_SCHEMA_SQL = [
    "CREATE EXTENSION IF NOT EXISTS vector",
    "CREATE EXTENSION IF NOT EXISTS unaccent",
    # unaccent() không IMMUTABLE nên không dùng được trong generated column; bọc lại để dùng cho full-text
    """
    CREATE OR REPLACE FUNCTION immutable_unaccent(text) RETURNS text
    LANGUAGE sql IMMUTABLE PARALLEL SAFE STRICT
    AS $$ SELECT public.unaccent('public.unaccent'::regdictionary, $1) $$
    """,
]

PG_POST_SCHEMA_SQL = [
    "CREATE INDEX IF NOT EXISTS ix_kb_chunks_embedding ON kb_chunks USING hnsw (embedding vector_cosine_ops)",
    "CREATE INDEX IF NOT EXISTS ix_kb_chunks_tsv ON kb_chunks USING gin (tsv)",
]

# FTS5 dạng external-content: chỉ lưu chỉ mục, nội dung đọc từ kb_chunks; trigger giữ đồng bộ.
# remove_diacritics 2 bỏ dấu tiếng Việt khi so khớp ("đường huyết" ~ "duong huyet"), tương đương unaccent.
SQLITE_POST_SCHEMA_SQL = [
    """
    CREATE VIRTUAL TABLE IF NOT EXISTS kb_chunks_fts USING fts5(
        heading, content, content='kb_chunks', content_rowid='id',
        tokenize='unicode61 remove_diacritics 2')
    """,
    """
    CREATE TRIGGER IF NOT EXISTS kb_chunks_ai AFTER INSERT ON kb_chunks BEGIN
        INSERT INTO kb_chunks_fts(rowid, heading, content) VALUES (new.id, coalesce(new.heading, ''), new.content);
    END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS kb_chunks_ad AFTER DELETE ON kb_chunks BEGIN
        INSERT INTO kb_chunks_fts(kb_chunks_fts, rowid, heading, content)
        VALUES ('delete', old.id, coalesce(old.heading, ''), old.content);
    END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS kb_chunks_au AFTER UPDATE ON kb_chunks BEGIN
        INSERT INTO kb_chunks_fts(kb_chunks_fts, rowid, heading, content)
        VALUES ('delete', old.id, coalesce(old.heading, ''), old.content);
        INSERT INTO kb_chunks_fts(rowid, heading, content) VALUES (new.id, coalesce(new.heading, ''), new.content);
    END
    """,
]


def init_db() -> None:
    from src.storage import models  # noqa: F401 — đăng ký các bảng vào metadata

    if settings.is_postgres:
        with engine.begin() as conn:
            for stmt in PG_PRE_SCHEMA_SQL:
                conn.execute(text(stmt))
    Base.metadata.create_all(engine)
    with engine.begin() as conn:
        for stmt in (PG_POST_SCHEMA_SQL if settings.is_postgres else SQLITE_POST_SCHEMA_SQL):
            conn.execute(text(stmt))
