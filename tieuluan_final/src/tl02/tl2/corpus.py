"""Nạp kho tri thức Markdown và tách thành các mục theo tiêu đề (giữ vị trí ký tự để chấm điểm chunking)."""
import re
import unicodedata
from dataclasses import dataclass, field

from .common import CORPUS

HEADING = re.compile(r"^(#{1,3})\s+(.+)$", re.M)


@dataclass
class Document:
    doc_id: str          # ví dụ "h03-ruou-bia"
    domain: str          # "health" | "finance"
    title: str
    text: str


@dataclass
class Section:
    doc_id: str
    domain: str
    path: list            # chuỗi tiêu đề từ gốc đến mục, ví dụ [tiêu đề tài liệu, tiêu đề mục]
    start: int            # vị trí ký tự trong Document.text
    end: int
    text: str = field(repr=False)


def nfc(s):
    return unicodedata.normalize("NFC", s)


def load_documents(domains=("health", "finance")):
    docs = []
    for domain in domains:
        for p in sorted((CORPUS / domain).glob("*.md")):
            text = nfc(p.read_text(encoding="utf-8"))
            title = HEADING.search(text).group(2).strip()
            docs.append(Document(p.stem, domain, title, text))
    return docs


def sections(doc):
    """Cắt tài liệu theo tiêu đề #, ##, ###; mỗi mục kéo dài tới tiêu đề kế tiếp."""
    heads = list(HEADING.finditer(doc.text))
    out, stack = [], []
    for i, m in enumerate(heads):
        level, name = len(m.group(1)), m.group(2).strip()
        stack = stack[: level - 1] + [name]
        end = heads[i + 1].start() if i + 1 < len(heads) else len(doc.text)
        out.append(Section(doc.doc_id, doc.domain, list(stack), m.start(), end, doc.text[m.start():end]))
    return out
