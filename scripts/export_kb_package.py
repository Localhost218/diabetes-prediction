"""
Script xuất toàn bộ tài liệu và tri thức Knowledge Base (KB) thành gói download hoàn chỉnh.
"""

import csv
import json
import os
import shutil
import sqlite3
import zipfile
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
KB_OUT_DIR = BASE_DIR / "data" / "output" / "kb"
DOWNLOADS_DIR = KB_OUT_DIR / "kb_downloads"
DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)

# 1. 01_raw_kb_sources
src_dir = DOWNLOADS_DIR / "01_raw_kb_sources"
src_dir.mkdir(exist_ok=True)
kb_sources_dir = KB_OUT_DIR / "kb_sources"
for f in kb_sources_dir.glob("*.md"):
    shutil.copy2(f, src_dir / f.name)
shutil.copy2(kb_sources_dir / "curated_triples.json", src_dir / "curated_triples.json")

# 2. 02_embedded_36_chunks
chunks_dir = DOWNLOADS_DIR / "02_embedded_36_chunks"
chunks_dir.mkdir(exist_ok=True)

con = sqlite3.connect(BASE_DIR / "data" / "output" / "storage" / "multiagent.db")
con.row_factory = sqlite3.Row
cur = con.cursor()

query = """
SELECT c.id, c.document_id, c.heading, c.content, c.embedding_model,
       d.title, d.source, d.url
FROM kb_chunks c
LEFT JOIN kb_documents d ON c.document_id = d.id
ORDER BY c.id
"""
chunks = [dict(r) for r in cur.execute(query).fetchall()]

with open(chunks_dir / "embedded_36_chunks.json", "w", encoding="utf-8") as f:
    json.dump(chunks, f, ensure_ascii=False, indent=2)

with open(chunks_dir / "embedded_36_chunks.csv", "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(
        f, fieldnames=["id", "document_id", "title", "heading", "source", "url", "content", "embedding_model"]
    )
    writer.writeheader()
    writer.writerows(chunks)

md_lines = ["# TỔNG HỢP TOÀN BỘ 36 ĐOẠN VĂN BẢN (CHUNKS) ĐÃ EMBEDDING VÀO KB & QDRANT\n"]
for c in chunks:
    md_lines.append(f"## Chunk #{c['id']}: {c['heading']}")
    md_lines.append(f"- **Tài liệu**: {c['title']} (`{c['document_id']}`)")
    md_lines.append(f"- **Nguồn trích dẫn**: {c['source']}")
    md_lines.append(f"- **Model Embedding**: `{c['embedding_model']}` | **Qdrant Vector Size**: `384` chiều")
    md_lines.append("")
    md_lines.append(c["content"])
    md_lines.append("\n---\n")

with open(chunks_dir / "embedded_36_chunks_all.md", "w", encoding="utf-8") as f:
    f.write("\n".join(md_lines))

# 3. 03_knowledge_graph
kg_dir = DOWNLOADS_DIR / "03_knowledge_graph"
kg_dir.mkdir(exist_ok=True)
if (KB_OUT_DIR / "diabetes_kb.cypher").exists():
    shutil.copy2(KB_OUT_DIR / "diabetes_kb.cypher", kg_dir / "diabetes_kb.cypher")
shutil.copy2(kb_sources_dir / "curated_triples.json", kg_dir / "triples.json")

# 4. 04_official_guidelines_summary
summary_dir = DOWNLOADS_DIR / "04_official_guidelines_summary"
summary_dir.mkdir(exist_ok=True)

byt_summary = """# TỔNG HỢP HƯỚNG DẪN BỘ Y TẾ: QUYẾT ĐỊNH 5481/QĐ-BYT (2020)
**Tên văn bản**: Quyết định số 5481/QĐ-BYT ngày 30/12/2020 của Bộ trưởng Bộ Y tế ban hành tài liệu chuyên môn "Hướng dẫn chẩn đoán và điều trị đái tháo đường típ 2".

## 1. Tiêu chuẩn chẩn đoán (áp dụng thống nhất tại Việt Nam)
Chẩn đoán xác định đái tháo đường khi có 1 trong 4 tiêu chí sau:
1. Glucose huyết tương lúc đói (FPG) >= 7.0 mmol/L (>= 126 mg/dL). Lúc đói là nhịn ăn ít nhất 8 giờ.
2. Glucose huyết tương 2 giờ sau nghiệm pháp dung nạp 75g glucose (OGTT) >= 11.1 mmol/L (>= 200 mg/dL).
3. HbA1c >= 6.5% (48 mmol/mol), thực hiện tại phòng xét nghiệm chuẩn hóa quốc tế theo tiêu chuẩn NGSP.
4. Bệnh nhân có triệu chứng kinh điển (tiểu nhiều, uống nhiều, sút cân không rõ nguyên nhân) kèm glucose huyết tương bất kỳ >= 11.1 mmol/L (>= 200 mg/dL).
*Lưu ý*: Nếu không có triệu chứng kinh điển, cần xét nghiệm lại lần 2 vào ngày khác hoặc xét nghiệm đồng thời 2 chỉ số khác nhau để xác nhận.

## 2. Tiêu chí sàng lọc người Việt Nam
- Người trưởng thành có BMI >= 23 kg/m2 (tiêu chuẩn thừa cân cho người châu Á của WHO) kèm ít nhất 1 yếu tố nguy cơ:
  + Tăng huyết áp (>= 130/80 mmHg hoặc đang uống thuốc hạ áp)
  + Rối loạn lipid máu (HDL-C < 0.9 mmol/L, Triglycerid > 2.8 mmol/L)
  + Tiền sử gia đình thế hệ thứ nhất (bố mẹ, anh chị em ruột) mắc ĐTĐ
  + Ít vận động thể lực
  + Tiền sử đái tháo đường thai kỳ hoặc sinh con >= 4.0 kg
  + Phụ nữ mắc hội chứng buồng trứng đa nang (PCOS)
- Tất cả người từ 35 tuổi trở lên (không phân biệt cân nặng).

## 3. Mục tiêu điều trị chung
- HbA1c < 7.0% (cá thể hóa mục tiêu: nới lỏng 7.5 - 8.0% ở người cao tuổi, nhiều bệnh đồng mắc, nguy cơ hạ đường huyết cao).
- Glucose huyết tương lúc đói / trước ăn: 4.4 - 7.2 mmol/L (80 - 130 mg/dL).
- Glucose huyết tương sau ăn 2 giờ: < 10.0 mmol/L (< 180 mg/dL).
- Huyết áp mục tiêu: < 130/80 mmHg.

## 4. Phác đồ điều trị bằng thuốc
- Thay đổi lối sống (dinh dưỡng giảm GI, vận động 150 phút/tuần) là nền tảng xuyên suốt.
- Metformin là thuốc lựa chọn đầu tay (khởi trị) nếu không có chống chỉ định.
- Chống chỉ định của Metformin: Suy thận nặng khi eGFR < 30 mL/phút/1.73m2; cẩn trọng/giảm liều khi eGFR 30 - 45.
- Phối hợp thuốc sớm: Khi HbA1c nền cao hơn mục tiêu >= 1.5% hoặc có bệnh tim mạch xơ vữa, suy tim, bệnh thận mạn (ưu tiên nhóm SGLT2i hoặc GLP-1 RA).
"""
with open(summary_dir / "Quyet_Dinh_5481_Bo_Y_Te_Chi_Tiet.md", "w", encoding="utf-8") as f:
    f.write(byt_summary)

# 5. README.md
readme = f"""# GÓI DỮ LIỆU TOÀN BỘ KNOWLEDGE BASE (KB) ĐÃ NẠP TRONG PROJECT

Thư mục này chứa đầy đủ toàn bộ tài liệu gốc, văn bản đã nạp, đồ thị tri thức và các đoạn văn bản (chunks) đã embedding vào Qdrant:

### Cấu trúc gói dữ liệu:
1. `01_raw_kb_sources/`: Các file Markdown nguồn chuẩn y khoa (Bộ Y tế QĐ 5481, ADA 2024, Long Châu, Vinmec) và file bộ ba tri thức `curated_triples.json`.
2. `02_embedded_36_chunks/`:
   - `embedded_36_chunks.json`: Xuất toàn bộ các chunks dưới dạng JSON cấu trúc.
   - `embedded_36_chunks.csv`: Bảng CSV các chunks kèm document_id, tiêu đề, nguồn và model embedding.
   - `embedded_36_chunks_all.md`: Tài liệu văn bản tổng hợp đọc trực tiếp toàn bộ đoạn đã embed.
3. `03_knowledge_graph/`:
   - `diabetes_kb.cypher`: Script Neo4j Cypher khởi tạo đồ thị tri thức.
   - `triples.json`: Danh sách các bộ ba thực thể (head, relation, tail, evidence, chunk_ids).
4. `04_official_guidelines_summary/`:
   - `Quyet_Dinh_5481_Bo_Y_Te_Chi_Tiet.md`: Tóm lược chi tiết Quyết định 5481/QĐ-BYT.

Tất cả đã được nén thành file ZIP tại: `data/output/kb/kb_export_full.zip`.
"""
with open(DOWNLOADS_DIR / "README.md", "w", encoding="utf-8") as f:
    f.write(readme)

# 6. Create zip file
zip_path = KB_OUT_DIR / "kb_export_full.zip"
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
    for root, dirs, files in os.walk(DOWNLOADS_DIR):
        for file in files:
            p = Path(root) / file
            arcname = p.relative_to(DOWNLOADS_DIR)
            zipf.write(p, arcname)

print(f"Export completed! Zip created at: {zip_path} ({zip_path.stat().st_size:,} bytes)")
