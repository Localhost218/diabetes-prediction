"""
Kiểm bản dịch tiếng Việt của tri thức nguồn nước ngoài (ADA…) so với đoạn trích nguyên văn.

    python scripts/verify_kb.py            # in báo cáo, mã thoát 1 nếu có bộ ba cần người đọc lại

Với mỗi bộ ba trong kb_sources/curated_triples.json có "quotes":
  1. Trích dẫn có thật: đoạn trích phải xuất hiện nguyên văn trong tài liệu gốc (PDF/HTML/TXT).
  2. Khớp con số: mọi con số trong câu tiếng Việt (evidence) phải có trong đoạn trích gốc.
  3. Khớp nghĩa: cosine giữa câu Việt và đoạn trích (embedding đa ngữ MiniLM) ≥ MIN_SIMILARITY.
Bộ ba không có "quotes" được liệt kê riêng: chưa kiểm chứng được với bản gốc.
"""

import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import settings  # noqa: E402
from src.knowledge.embedder import get_embedder  # noqa: E402
from src.knowledge.ingest import discover_sources  # noqa: E402

# shortcut: ngưỡng chọn theo kinh nghiệm trên 18 bộ ba hiện có, hiệu chỉnh lại khi KB lớn hơn
MIN_SIMILARITY = 0.5
NUMBER = re.compile(r"\d+(?:[.,]\d+)?")


def numbers(text: str) -> set:
    """Các con số trong câu, chuẩn hóa dấu thập phân (7,0 ≡ 7.0 ≡ 7) để so được giữa tiếng Việt và tiếng Anh."""
    return {float(n.replace(",", ".")) for n in NUMBER.findall(text)}


def missing_numbers(vi: str, sources: list) -> set:
    """Số có trong câu tiếng Việt nhưng không có trong bất kỳ đoạn trích gốc nào."""
    found = set().union(*(numbers(s) for s in sources)) if sources else set()
    return numbers(vi) - found


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def main() -> int:
    triples = json.loads((settings.kb_sources_dir / "curated_triples.json").read_text(encoding="utf-8"))
    full_text = {d.id: _norm("\n".join(c for _, c in d.sections)) for d in discover_sources(settings.kb_sources_dir)}
    embedder = get_embedder()

    checked, flagged, unverified = 0, [], []
    for t in triples:
        name = f"{t['head']} —{t['relation']}→ {t['tail']}"
        quotes = t.get("quotes") or []
        if not quotes:
            unverified.append(name)
            continue
        checked += 1
        issues = []
        for q in quotes:
            if _norm(q["text"]) not in full_text.get(q["doc"], ""):
                issues.append(f"trích dẫn không có trong {q['doc']}: {q['text'][:60]}")
        lost = missing_numbers(t["evidence"], [q["text"] for q in quotes])
        if lost:
            issues.append(f"số không có trong bản gốc: {sorted(lost)}")
        vecs = np.array(embedder.embed([t["evidence"]] + [q["text"] for q in quotes]), dtype=float)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        sim = float((vecs[1:] @ vecs[0]).max())
        if sim < MIN_SIMILARITY:
            issues.append(f"độ tương đồng nghĩa thấp ({sim:.2f} < {MIN_SIMILARITY})")
        print(f"{'⚠' if issues else '✓'} {sim:.2f}  {name}")
        for i in issues:
            print(f"      - {i}")
        if issues:
            flagged.append(name)

    print(f"\nĐã kiểm {checked} bộ ba có trích dẫn gốc: {checked - len(flagged)} đạt, {len(flagged)} cần người đọc lại.")
    print(f"{len(unverified)} bộ ba chưa có trích dẫn gốc (chưa kiểm chứng được với tài liệu nguồn).")
    return 1 if flagged else 0


if __name__ == "__main__":
    sys.exit(main())
