"""
Đọc file bệnh án người dùng tải lên thành văn bản / ảnh / chuỗi số đo.
"""

import io
import re
from dataclasses import dataclass, field
from typing import List, Optional

from src.document_processing.lab_extractor import GLUCOSE_SERIES_HINT

SUPPORTED_TYPES = {
    "application/pdf": "pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    "text/csv": "csv",
    "application/vnd.ms-excel": "csv",
    "text/plain": "text",
    "image/jpeg": "image",
    "image/png": "image",
    "image/webp": "image",
}
EXTENSION_TYPES = {".pdf": "pdf", ".docx": "docx", ".csv": "csv", ".txt": "text",
                   ".jpg": "image", ".jpeg": "image", ".png": "image", ".webp": "image"}
IMAGE_MIME = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}
MAX_PDF_PAGES = 20
SCANNED_TEXT_THRESHOLD = 40  # số ký tự/trang dưới ngưỡng này coi như PDF scan


@dataclass
class LoadedDocument:
    kind: str                       # pdf_text | pdf_scanned | docx | csv | text | image
    text: str = ""
    images: List[bytes] = field(default_factory=list)   # ảnh PNG/JPEG để OCR hoặc gửi LLM vision
    image_mime: str = "image/png"
    glucose_series: Optional[List[float]] = None
    pages: int = 0
    notes: List[str] = field(default_factory=list)


def detect_kind(filename: str, mime_type: str) -> Optional[str]:
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    return EXTENSION_TYPES.get(ext) or SUPPORTED_TYPES.get(mime_type)


def load(data: bytes, filename: str, mime_type: str) -> LoadedDocument:
    kind = detect_kind(filename, mime_type)
    if kind == "pdf":
        return _load_pdf(data)
    if kind == "docx":
        return _load_docx(data)
    if kind == "csv":
        return _load_csv(data)
    if kind == "text":
        return LoadedDocument(kind="text", text=data.decode("utf-8", errors="ignore"))
    if kind == "image":
        ext = "." + filename.rsplit(".", 1)[-1].lower()
        return LoadedDocument(kind="image", images=[data], image_mime=IMAGE_MIME.get(ext, mime_type), pages=1)
    raise ValueError(f"Định dạng không hỗ trợ: {filename} ({mime_type})")


def _load_pdf(data: bytes) -> LoadedDocument:
    import pymupdf as fitz
    with fitz.open(stream=data, filetype="pdf") as pdf:
        pages = min(len(pdf), MAX_PDF_PAGES)
        texts = [pdf[i].get_text() for i in range(pages)]
        doc = LoadedDocument(kind="pdf_text", text="\n\n".join(texts), pages=pages)
        if len(pdf) > MAX_PDF_PAGES:
            doc.notes.append(f"Chỉ đọc {MAX_PDF_PAGES}/{len(pdf)} trang đầu")
        if sum(len(t.strip()) for t in texts) < SCANNED_TEXT_THRESHOLD * pages:
            doc.kind = "pdf_scanned"
            doc.images = [pdf[i].get_pixmap(dpi=150).tobytes("png") for i in range(pages)]
    return doc


def _load_docx(data: bytes) -> LoadedDocument:
    import docx
    document = docx.Document(io.BytesIO(data))
    lines = [p.text for p in document.paragraphs if p.text.strip()]
    for table in document.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells]
            lines.append(" : ".join(dict.fromkeys(c for c in cells if c)))  # bỏ ô gộp bị lặp
    return LoadedDocument(kind="docx", text="\n".join(lines), pages=1)


def _load_csv(data: bytes) -> LoadedDocument:
    import pandas as pd
    df = pd.read_csv(io.BytesIO(data), sep=None, engine="python")
    doc = LoadedDocument(kind="csv", text=df.head(30).to_string(index=False), pages=1)
    numeric = df.select_dtypes("number")
    column = next((c for c in numeric.columns if GLUCOSE_SERIES_HINT(str(c))), None)
    if column is None and numeric.shape[1] == 1:
        column = numeric.columns[0]
    if column is not None:
        series = [float(v) for v in numeric[column].dropna().tolist()]
        unit_hint = str(column).lower()
        if "mmol" in unit_hint or (series and max(series) < 35):
            series = [round(v * 18.016, 1) for v in series]
        doc.glucose_series = series
        doc.notes.append(f"Chuỗi glucose từ cột '{column}' ({len(series)} lần đo)")
    return doc


def tesseract_available() -> bool:
    import shutil
    return shutil.which("tesseract") is not None


def ocr_images(images: List[bytes]) -> str:
    """OCR tiếng Việt bằng Tesseract (nhánh dự phòng khi không có LLM vision)."""
    import pytesseract
    from PIL import Image
    texts = []
    for raw in images:
        texts.append(pytesseract.image_to_string(Image.open(io.BytesIO(raw)), lang="vie+eng"))
    return re.sub(r"[ \t]+", " ", "\n".join(texts))
