"""
Pipeline nạp tri thức (khối PDF/Web → KB trên sơ đồ):

  file .md/.txt/.pdf/.html hoặc URL
    → lưu bản gốc lên SeaweedFS (bucket KB)
    → trích văn bản → chia đoạn theo mục
    → embedding → pgvector (kb_chunks)
    → bộ ba tri thức (tuyển chọn thủ công + trích bằng LLM nếu bật) → Apache AGE

Chạy lại an toàn: tài liệu không đổi (cùng sha256 và cùng mô hình embedding) sẽ được bỏ qua.
"""

import asyncio
import hashlib
import html
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from sqlalchemy import delete, select

from src.config import settings
from src.knowledge.embedder import get_embedder
from src.storage.db import session_scope
from src.storage.graph_store import get_graph_store
from src.storage.models import KbChunk, KbDocument
from src.storage.object_storage import get_storage

log = logging.getLogger(__name__)

CURATED_TRIPLES = settings.kb_sources_dir / "curated_triples.json"
MAX_CHUNK_CHARS = 1200


@dataclass
class SourceDoc:
    id: str
    title: str
    source: str
    url: Optional[str]
    raw: bytes
    ext: str
    sections: List[Tuple[str, str]] = field(default_factory=list)  # (heading, content)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.raw).hexdigest()


# ------------------------------------------------------------------ đọc nguồn
def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:90] or "doc"


def _split_markdown(body: str) -> List[Tuple[str, str]]:
    sections, heading, buf = [], None, []
    for line in body.splitlines():
        if line.startswith("## "):
            if heading and "".join(buf).strip():
                sections.append((heading, "\n".join(buf).strip()))
            heading, buf = line[3:].strip(), []
        elif line.startswith("# "):
            continue
        else:
            buf.append(line)
    if heading and "".join(buf).strip():
        sections.append((heading, "\n".join(buf).strip()))
    return sections


def _split_plain(text: str, title: str) -> List[Tuple[str, str]]:
    """Chia văn bản không có tiêu đề mục thành các đoạn ~MAX_CHUNK_CHARS theo ranh giới đoạn văn."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks, buf = [], ""
    for p in paragraphs:
        if buf and len(buf) + len(p) > MAX_CHUNK_CHARS:
            chunks.append(buf)
            buf = buf[-200:]  # chồng lấp để không cắt ngang ý
        buf = f"{buf}\n\n{p}".strip()
    if buf:
        chunks.append(buf)
    return [(f"{title} — phần {i + 1}", c) for i, c in enumerate(chunks)]


def load_markdown(path: Path) -> SourceDoc:
    raw = path.read_bytes()
    text = raw.decode("utf-8")
    meta, body = {}, text
    m = re.match(r"^---\n(.*?)\n---\n", text, re.S)
    if m:
        for line in m.group(1).splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                meta[k.strip()] = v.strip()
        body = text[m.end():]
    doc = SourceDoc(id=meta.get("id", _slug(path.stem)), title=meta.get("title", path.stem),
                    source=meta.get("source", path.name), url=meta.get("url"), raw=raw, ext=".md")
    doc.sections = _split_markdown(body) or _split_plain(body, doc.title)
    return doc


def load_pdf(path_or_bytes, name: str) -> SourceDoc:
    import pymupdf as fitz
    raw = path_or_bytes.read_bytes() if isinstance(path_or_bytes, Path) else path_or_bytes
    with fitz.open(stream=raw, filetype="pdf") as pdf:
        title = (pdf.metadata or {}).get("title") or name
        text = "\n\n".join(page.get_text() for page in pdf)
    doc = SourceDoc(id=_slug(Path(name).stem), title=title, source=name, url=None, raw=raw, ext=".pdf")
    doc.sections = _split_plain(text, title)
    return doc


def _html_to_text(raw_html: str) -> Tuple[str, str]:
    title_m = re.search(r"<title[^>]*>(.*?)</title>", raw_html, re.S | re.I)
    body = re.sub(r"<(script|style|nav|footer|header)[^>]*>.*?</\1>", " ", raw_html, flags=re.S | re.I)
    body = re.sub(r"<(br|p|div|li|h[1-6]|tr)[^>]*>", "\n\n", body, flags=re.I)
    body = html.unescape(re.sub(r"<[^>]+>", " ", body))
    body = re.sub(r"[ \t]+", " ", body)
    return (html.unescape(title_m.group(1).strip()) if title_m else ""), body


def load_html(raw: bytes, name: str, url: Optional[str] = None) -> SourceDoc:
    title, text = _html_to_text(raw.decode("utf-8", errors="ignore"))
    doc = SourceDoc(id=_slug(Path(name).stem if not url else url.split("//")[-1]), title=title or name,
                    source=url or name, url=url, raw=raw, ext=".html")
    doc.sections = _split_plain(text, doc.title)
    return doc


def load_url(url: str) -> SourceDoc:
    import httpx
    response = httpx.get(url, follow_redirects=True, timeout=30, headers={"User-Agent": "DiabetesPrediction-KB-Ingest/1.0"})
    response.raise_for_status()
    if "pdf" in response.headers.get("content-type", ""):
        doc = load_pdf(response.content, url.rsplit("/", 1)[-1] or "document.pdf")
        doc.url, doc.source = url, url
        return doc
    return load_html(response.content, url, url)


def discover_sources(directory: Path) -> List[SourceDoc]:
    docs = []
    for path in sorted(directory.iterdir()):
        suffix = path.suffix.lower()
        if suffix in (".md", ".txt"):
            docs.append(load_markdown(path))
        elif suffix == ".pdf":
            docs.append(load_pdf(path, path.name))
        elif suffix in (".html", ".htm"):
            docs.append(load_html(path.read_bytes(), path.name))
    return docs


# ------------------------------------------------------------------ nạp vào KB
CONTENT_TYPES = {".md": "text/markdown", ".pdf": "application/pdf", ".html": "text/html"}


def ingest_documents(docs: List[SourceDoc], force: bool = False) -> Dict[str, int]:
    embedder = get_embedder()
    storage = get_storage()
    stats = {"documents": 0, "skipped": 0, "chunks": 0}
    for doc in docs:
        with session_scope() as s:
            existing = s.get(KbDocument, doc.id)
            same_model = existing is not None and s.scalar(
                select(KbChunk.embedding_model).where(KbChunk.document_id == doc.id).limit(1)) == embedder.name
            if existing and existing.sha256 == doc.sha256 and same_model and not force:
                stats["skipped"] += 1
                continue

            key = f"kb-sources/{doc.id}{doc.ext}"
            try:
                storage.put(settings.s3_bucket_kb, key, doc.raw, CONTENT_TYPES.get(doc.ext, "application/octet-stream"))
            except Exception as exc:  # noqa: BLE001 — KB vẫn dùng được khi object storage tạm lỗi
                log.warning("Không lưu được bản gốc %s lên object storage: %s", doc.id, exc)
                key = None

            if existing:
                s.execute(delete(KbChunk).where(KbChunk.document_id == doc.id))
                existing.title, existing.source, existing.url, existing.sha256 = doc.title, doc.source, doc.url, doc.sha256
                existing.bucket, existing.object_key = settings.s3_bucket_kb, key
            else:
                s.add(KbDocument(id=doc.id, title=doc.title, source=doc.source, url=doc.url, sha256=doc.sha256,
                                 bucket=settings.s3_bucket_kb, object_key=key))
            s.flush()

            vectors = embedder.embed([f"{doc.title}. {h}. {c}" for h, c in doc.sections])
            for i, ((heading, content), vec) in enumerate(zip(doc.sections, vectors)):
                s.add(KbChunk(document_id=doc.id, chunk_index=i, heading=heading, content=content,
                              embedding=vec, embedding_model=embedder.name))
            stats["documents"] += 1
            stats["chunks"] += len(doc.sections)
    return stats


def _chunk_lookup() -> Dict[Tuple[str, str], int]:
    with session_scope() as s:
        return {(d, h): cid for cid, d, h in s.execute(select(KbChunk.id, KbChunk.document_id, KbChunk.heading))}


def load_curated_triples() -> List[dict]:
    if not CURATED_TRIPLES.exists():
        return []
    lookup = _chunk_lookup()
    triples = json.loads(CURATED_TRIPLES.read_text(encoding="utf-8"))
    for t in triples:
        chunk_id = lookup.get((t["source_doc"], t.pop("section", None)))
        t["chunk_ids"] = [chunk_id] if chunk_id else []
    return triples


async def extract_llm_triples() -> List[dict]:
    """Trích thêm bộ ba từ từng chunk bằng LLM (khi có API key)."""
    from src.llm.client import llm
    from src.llm.prompts import TRIPLE_SYSTEM, TripleList

    with session_scope() as s:
        chunks = s.execute(select(KbChunk.id, KbChunk.document_id, KbChunk.heading, KbChunk.content)).all()
    sem = asyncio.Semaphore(4)

    async def one(chunk):
        async with sem:
            result = await llm.parse(TRIPLE_SYSTEM, [f"Mục: {chunk.heading}\n\n{chunk.content}"], TripleList)
        return [{**t.model_dump(), "source_doc": chunk.document_id, "chunk_ids": [chunk.id]}
                for t in (result.triples if result else [])]

    results = await asyncio.gather(*(one(c) for c in chunks))
    return [t for batch in results for t in batch]


def rebuild_graph(llm_triples: bool = False) -> Dict[str, int]:
    triples = load_curated_triples()
    if llm_triples:
        triples += asyncio.run(extract_llm_triples())
    with session_scope() as s:
        graph = get_graph_store(s)
        graph.clear()
        graph.upsert_triples(triples)
        return graph.stats()
