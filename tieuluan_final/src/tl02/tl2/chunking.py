"""Năm chiến lược phân mảnh văn bản cho RAG.

Mọi chiến lược trả về danh sách Chunk có (start, end) là vị trí ký tự trong tài liệu gốc, nhờ đó
có thể chấm điểm truy xuất bằng việc kiểm tra một chunk có chứa trọn câu trả lời hay không.
"""
import re
from dataclasses import dataclass, field

import numpy as np

from .corpus import sections


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    domain: str
    start: int
    end: int
    text: str
    heading: str = ""            # tiêu đề ngữ cảnh gắn vào metadata (structure-aware)
    parent_id: str = None        # small-to-big: chunk con trỏ về chunk cha
    meta: dict = field(default_factory=dict)

    def index_text(self):
        """Văn bản đem đi lập chỉ mục: gắn tiêu đề ngữ cảnh nếu có."""
        return f"{self.heading}\n{self.text}" if self.heading else self.text


def _mk(doc, i, s, e, prefix, **kw):
    return Chunk(f"{doc.doc_id}:{prefix}{i}", doc.doc_id, doc.domain, s, e, doc.text[s:e], **kw)


# ---------------------------------------------------------------- 1. kích thước cố định
def fixed_size(doc, size=400, overlap=60):
    """Cửa sổ trượt theo ký tự, bước S = size - overlap, không quan tâm ranh giới câu."""
    step, out, s = size - overlap, [], 0
    while s < len(doc.text):
        e = min(s + size, len(doc.text))
        out.append(_mk(doc, len(out), s, e, "fx"))
        if e == len(doc.text):
            break
        s += step
    return out


# ---------------------------------------------------------------- 2. đệ quy theo dấu phân cách
SEPARATORS = ["\n\n", "\n", ". ", "; ", ", ", " "]


def _recursive_spans(text, s, e, size, seps):
    """Trả về các khoảng [s, e) dài ≤ size, ưu tiên cắt ở dấu phân cách cấp cao nhất."""
    if e - s <= size:
        return [(s, e)]
    for k, sep in enumerate(seps):
        cuts = [s + m.end() for m in re.finditer(re.escape(sep), text[s:e])]
        cuts = [c for c in cuts if c < e]
        if not cuts:
            continue
        pieces = list(zip([s] + cuts, cuts + [e]))
        # gộp các mảnh nhỏ liên tiếp cho tới khi chạm giới hạn size
        merged = []
        for a, b in pieces:
            if merged and b - merged[-1][0] <= size:
                merged[-1] = (merged[-1][0], b)
            else:
                merged.append((a, b))
        out = []
        for a, b in merged:
            out += _recursive_spans(text, a, b, size, seps[k + 1:]) if b - a > size else [(a, b)]
        return out
    return [(i, min(i + size, e)) for i in range(s, e, size)]


def recursive(doc, size=400, overlap=60):
    """Recursive Character Chunking: tách theo đoạn, rồi dòng, câu, mệnh đề, từ; thêm chồng lấn."""
    spans = _recursive_spans(doc.text, 0, len(doc.text), size, SEPARATORS)
    out = []
    for i, (s, e) in enumerate(spans):
        s2 = max(0, s - overlap) if i else s
        # lùi điểm đầu về đầu từ để phần chồng lấn không cắt ngang chữ
        while s2 > 0 and not doc.text[s2 - 1].isspace() and s2 > s - overlap - 20:
            s2 -= 1
        out.append(_mk(doc, i, s2, e, "rc"))
    return [c for c in out if c.text.strip()]


# ---------------------------------------------------------------- 3. theo cấu trúc tài liệu
def structure_aware(doc, size=900):
    """Mỗi mục Markdown là một chunk, gắn chuỗi tiêu đề làm ngữ cảnh; mục quá dài tách đệ quy bên trong."""
    out = []
    for sec in sections(doc):
        body = sec.text
        if len(body.strip().splitlines()) <= 1:          # tiêu đề tài liệu không có nội dung riêng
            continue
        heading = " > ".join(sec.path)
        spans = _recursive_spans(doc.text, sec.start, sec.end, size, SEPARATORS)
        for s, e in spans:
            out.append(_mk(doc, len(out), s, e, "st", heading=heading))
    return out


# ---------------------------------------------------------------- 4. ngữ nghĩa
SENT = re.compile(r"[^.!?\n]+(?:[.!?]+|\n|$)")


def sentence_spans(text):
    return [(m.start(), m.end()) for m in SENT.finditer(text) if text[m.start():m.end()].strip()]


def semantic(doc, embed, percentile=80, max_size=900):
    """Cắt khi khoảng cách cosin giữa hai câu liên tiếp vượt phân vị p của chuỗi khoảng cách.

    embed: hàm nhận list[str] trả về ma trận embedding đã chuẩn hóa L2.
    """
    sents = sentence_spans(doc.text)
    if len(sents) < 3:
        return [_mk(doc, 0, 0, len(doc.text), "se")]
    E = embed([doc.text[s:e] for s, e in sents])
    d = 1.0 - np.sum(E[:-1] * E[1:], axis=1)              # d_i = 1 - cos(e_i, e_{i+1})
    thr = np.percentile(d, percentile)
    out, cur = [], sents[0][0]
    for i in range(len(sents) - 1):
        nxt = sents[i + 1][0]
        if d[i] > thr or nxt - cur > max_size:
            out.append(_mk(doc, len(out), cur, sents[i][1], "se", meta={"dist": float(d[i])}))
            cur = nxt
    out.append(_mk(doc, len(out), cur, sents[-1][1], "se"))
    return out, d, thr


# ---------------------------------------------------------------- 5. cha - con (small-to-big)
def parent_child(doc, parent_size=900, child_size=220):
    """Cha là mục theo cấu trúc; con là mảnh đệ quy nhỏ bên trong cha. Lập chỉ mục con, trả về cha."""
    parents = structure_aware(doc, size=parent_size)
    children = []
    for p in parents:
        for j, (s, e) in enumerate(_recursive_spans(doc.text, p.start, p.end, child_size, SEPARATORS)):
            if doc.text[s:e].strip():
                children.append(Chunk(f"{p.chunk_id}.c{j}", doc.doc_id, doc.domain, s, e, doc.text[s:e],
                                      heading=p.heading, parent_id=p.chunk_id))
    return parents, children


def chunk_corpus(docs, strategy, **kw):
    """Áp một chiến lược cho cả kho; semantic trả thêm khoảng cách để vẽ hình."""
    out = []
    for doc in docs:
        r = strategy(doc, **kw)
        out += r[0] if isinstance(r, tuple) and strategy is semantic else r
    return out
