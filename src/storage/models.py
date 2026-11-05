"""
Mô hình dữ liệu quan hệ — chạy được trên SQLite (mặc định) và PostgreSQL.

JSON dùng kiểu JSONB khi ở Postgres. Cột `embedding` (pgvector) và `tsv` (tsvector) chỉ tồn tại
khi chạy Postgres; với SQLite, vector nằm ở Qdrant và full-text ở bảng ảo FTS5 `kb_chunks_fts`.
"""

import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import (JSON, Boolean, Computed, DateTime, Float, ForeignKey, Integer, String, Text,
                        UniqueConstraint, func)
from sqlalchemy.dialects.postgresql import JSONB as PG_JSONB, TSVECTOR
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.config import settings
from src.storage.db import Base

if settings.is_postgres:
    from pgvector.sqlalchemy import Vector

JSONB = JSON().with_variant(PG_JSONB(), "postgresql")  # JSON trên SQLite, JSONB trên Postgres


def _uuid() -> str:
    return uuid.uuid4().hex


def _now() -> datetime:
    # Gán từ Python (độ phân giải micro giây) để các tin nhắn trong cùng giao dịch vẫn sắp xếp đúng thứ tự
    return datetime.now(timezone.utc)


class ChatSession(Base):
    """Một cuộc hội thoại; `profile` là PatientProfile tích lũy qua các lượt chat."""
    __tablename__ = "chat_sessions"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    title: Mapped[str] = mapped_column(String(200), default="Cuộc trò chuyện mới")
    profile: Mapped[dict] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    messages: Mapped[list["ChatMessage"]] = relationship(
        back_populates="session", cascade="all, delete-orphan", order_by="ChatMessage.created_at")


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    session_id: Mapped[str] = mapped_column(ForeignKey("chat_sessions.id", ondelete="CASCADE"), index=True)
    sender: Mapped[str] = mapped_column(String(10))  # user | bot
    text: Mapped[str] = mapped_column(Text, default="")
    data: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)  # kết quả chi tiết của hội đồng
    attachments: Mapped[list] = mapped_column(JSONB, default=list)  # [{document_id, filename, mime_type}]
    is_helpful: Mapped[bool] = mapped_column(Boolean, default=False)
    is_useless: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    session: Mapped[ChatSession] = relationship(back_populates="messages")


class AgentMessageLog(Base):
    """Audit trail của MessageBus."""
    __tablename__ = "agent_messages"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    session_id: Mapped[str] = mapped_column(ForeignKey("chat_sessions.id", ondelete="CASCADE"), index=True)
    conversation_id: Mapped[str] = mapped_column(String(32), index=True)
    in_reply_to: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    sender: Mapped[str] = mapped_column(String(64))
    receiver: Mapped[str] = mapped_column(String(64))
    performative: Mapped[str] = mapped_column(String(16))
    msg_type: Mapped[str] = mapped_column(String(64))
    payload: Mapped[dict] = mapped_column(JSONB, default=dict)
    elapsed_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class Document(Base):
    """Bệnh án người dùng tải lên; file gốc nằm trên SeaweedFS (S3)."""
    __tablename__ = "documents"
    __table_args__ = (UniqueConstraint("session_id", "sha256"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    session_id: Mapped[str] = mapped_column(ForeignKey("chat_sessions.id", ondelete="CASCADE"), index=True)
    filename: Mapped[str] = mapped_column(String(255))
    mime_type: Mapped[str] = mapped_column(String(100))
    size_bytes: Mapped[int] = mapped_column(Integer)
    sha256: Mapped[str] = mapped_column(String(64))
    bucket: Mapped[str] = mapped_column(String(100))
    object_key: Mapped[str] = mapped_column(String(500))
    status: Mapped[str] = mapped_column(String(20), default="pending")  # pending | parsed | failed
    extracted: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Consultation(Base):
    """Kết quả một phiên hội chẩn đa tác tử."""
    __tablename__ = "consultations"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    session_id: Mapped[str] = mapped_column(ForeignKey("chat_sessions.id", ondelete="CASCADE"), index=True)
    message_id: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    profile: Mapped[dict] = mapped_column(JSONB)
    risk_score: Mapped[float] = mapped_column(Float)
    risk_level: Mapped[str] = mapped_column(String(40))
    debate_rounds: Mapped[int] = mapped_column(Integer, default=0)
    result: Mapped[dict] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class KbDocument(Base):
    """Tài liệu nguồn của Knowledge Base (hướng dẫn ADA, Bộ Y tế...)."""
    __tablename__ = "kb_documents"

    id: Mapped[str] = mapped_column(String(100), primary_key=True)
    title: Mapped[str] = mapped_column(String(300))
    source: Mapped[str] = mapped_column(String(300))
    url: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    bucket: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    object_key: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    sha256: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    chunks: Mapped[list["KbChunk"]] = relationship(back_populates="document", cascade="all, delete-orphan")


class KbChunk(Base):
    __tablename__ = "kb_chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    document_id: Mapped[str] = mapped_column(ForeignKey("kb_documents.id", ondelete="CASCADE"), index=True)
    chunk_index: Mapped[int] = mapped_column(Integer)
    heading: Mapped[Optional[str]] = mapped_column(String(300), nullable=True)
    content: Mapped[str] = mapped_column(Text)
    embedding_model: Mapped[str] = mapped_column(String(200))
    if settings.is_postgres:
        embedding = mapped_column(Vector(settings.embedding_dim))
        tsv = mapped_column(TSVECTOR, Computed(
            "to_tsvector('simple', immutable_unaccent(coalesce(heading, '') || ' ' || content))", persisted=True))

    document: Mapped[KbDocument] = relationship(back_populates="chunks")


class KgNode(Base):
    """Bảng dự phòng cho Knowledge Graph trên Postgres không có Apache AGE (KG_BACKEND=sql)."""
    __tablename__ = "kg_nodes"
    __table_args__ = (UniqueConstraint("label", "name"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    label: Mapped[str] = mapped_column(String(40))
    name: Mapped[str] = mapped_column(String(200))


class KgEdge(Base):
    __tablename__ = "kg_edges"
    __table_args__ = (UniqueConstraint("head_id", "relation", "tail_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    head_id: Mapped[int] = mapped_column(ForeignKey("kg_nodes.id", ondelete="CASCADE"), index=True)
    relation: Mapped[str] = mapped_column(String(40))
    tail_id: Mapped[int] = mapped_column(ForeignKey("kg_nodes.id", ondelete="CASCADE"), index=True)
    evidence: Mapped[str] = mapped_column(Text, default="")
    source_doc: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    chunk_ids: Mapped[list] = mapped_column(JSONB, default=list)
