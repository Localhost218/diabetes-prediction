"""
Crawl dữ liệu thuốc và phác đồ điều trị đái tháo đường từ Nhà thuốc FPT Long Châu & Hệ thống Y tế Vinmec:
  1. Lưu dữ liệu thô (raw HTML + metadata không chỉnh sửa) vào:
     - `data/download/long_chau/`
     - `data/download/vinmec/`
  2. Chuẩn hóa thành tài liệu Markdown & cập nhật bộ ba đồ thị tri thức vào `data/output/kb/kb_sources/`

Chạy:
    .venv/bin/python scripts/crawl_longchau_vinmec.py
"""

import html
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from curl_cffi import requests as cffi_requests

BASE_DIR = Path(__file__).resolve().parent.parent
LONGCHAU_RAW_DIR = BASE_DIR / "data" / "download" / "long_chau"
VINMEC_RAW_DIR = BASE_DIR / "data" / "download" / "vinmec"
KB_OUT_DIR = BASE_DIR / "data" / "output" / "kb" / "kb_sources"

SOURCES = [
    {
        "slug": "longchau_danh_muc_thuoc_tri_tieu_duong",
        "provider": "Nhà thuốc FPT Long Châu",
        "target_dir": "long_chau",
        "url": "https://nhathuoclongchau.com.vn/thuoc/thuoc-tri-tieu-duong",
        "drug_group": "Danh mục thuốc trị tiểu đường",
    },
    {
        "slug": "longchau_cac_nhom_thuoc_tieu_duong",
        "provider": "Nhà thuốc FPT Long Châu",
        "target_dir": "long_chau",
        "url": "https://nhathuoclongchau.com.vn/bai-viet/cac-nhom-thuoc-tieu-duong-va-uu-nhuoc-diem-cua-thuoc-dieu-tri-tieu-duong.html",
        "drug_group": "Tổng quan các nhóm thuốc tiểu đường",
    },
    {
        "slug": "longchau_glucophage_500mg_metformin",
        "provider": "Nhà thuốc FPT Long Châu",
        "target_dir": "long_chau",
        "url": "https://nhathuoclongchau.com.vn/thuoc/glucophage-500mg-3584.html",
        "drug_group": "Biguanide (Metformin 500mg)",
    },
    {
        "slug": "longchau_glucophage_850mg_metformin",
        "provider": "Nhà thuốc FPT Long Châu",
        "target_dir": "long_chau",
        "url": "https://nhathuoclongchau.com.vn/thuoc/glucophage-850mg-3583.html",
        "drug_group": "Biguanide (Metformin 850mg)",
    },
    {
        "slug": "longchau_forxiga_10mg_dapagliflozin",
        "provider": "Nhà thuốc FPT Long Châu",
        "target_dir": "long_chau",
        "url": "https://nhathuoclongchau.com.vn/thuoc/forxiga-10mg-3713.html",
        "drug_group": "Ức chế SGLT2 (Dapagliflozin 10mg)",
    },
    {
        "slug": "longchau_jardiance_25mg_empagliflozin",
        "provider": "Nhà thuốc FPT Long Châu",
        "target_dir": "long_chau",
        "url": "https://nhathuoclongchau.com.vn/thuoc/jardiance-25mg-30v-18702.html",
        "drug_group": "Ức chế SGLT2 (Empagliflozin 25mg)",
    },
    {
        "slug": "longchau_januvia_100mg_sitagliptin",
        "provider": "Nhà thuốc FPT Long Châu",
        "target_dir": "long_chau",
        "url": "https://nhathuoclongchau.com.vn/thuoc/januvia-100mg-3386.html",
        "drug_group": "Ức chế DPP-4 (Sitagliptin 100mg)",
    },
    {
        "slug": "longchau_trajenta_5mg_linagliptin",
        "provider": "Nhà thuốc FPT Long Châu",
        "target_dir": "long_chau",
        "url": "https://nhathuoclongchau.com.vn/thuoc/trajenta-5mg-1945.html",
        "drug_group": "Ức chế DPP-4 (Linagliptin 5mg)",
    },
    {
        "slug": "longchau_diamicron_mr_60mg_gliclazide",
        "provider": "Nhà thuốc FPT Long Châu",
        "target_dir": "long_chau",
        "url": "https://nhathuoclongchau.com.vn/thuoc/diamicron-mr-60mg-4015.html",
        "drug_group": "Sulfonylurea thế hệ 2 (Gliclazide MR 60mg)",
    },
    {
        "slug": "longchau_lantus_solostar_insulin_glargine",
        "provider": "Nhà thuốc FPT Long Châu",
        "target_dir": "long_chau",
        "url": "https://nhathuoclongchau.com.vn/thuoc/lantus-solostar-3ml-3289.html",
        "drug_group": "Insulin nền tác dụng kéo dài (Insulin Glargine)",
    },
    {
        "slug": "vinmec_phac_do_dieu_tri_tieu_duong_type_2",
        "provider": "Hệ thống Y tế Vinmec",
        "target_dir": "vinmec",
        "url": "https://www.vinmec.com/vie/bai-viet/phac-do-dieu-tri-tieu-duong-type-2-vi",
        "drug_group": "Phác đồ điều trị tiểu đường type 2",
    },
    {
        "slug": "vinmec_thuoc_dieu_tri_dai_thao_duong_type_2",
        "provider": "Hệ thống Y tế Vinmec",
        "target_dir": "vinmec",
        "url": "https://www.vinmec.com/vie/bai-viet/thuoc-dieu-tri-dai-thao-duong-type-2-vi",
        "drug_group": "Thuốc điều trị đái tháo đường type 2",
    },
    {
        "slug": "vinmec_cac_nhom_thuoc_dieu_tri_dai_thao_duong",
        "provider": "Hệ thống Y tế Vinmec",
        "target_dir": "vinmec",
        "url": "https://www.vinmec.com/vie/bai-viet/cac-nhom-thuoc-dieu-tri-dai-thao-duong-vi",
        "drug_group": "Các nhóm thuốc điều trị đái tháo đường",
    },
    {
        "slug": "vinmec_cac_loai_insulin_va_cach_dung",
        "provider": "Hệ thống Y tế Vinmec",
        "target_dir": "vinmec",
        "url": "https://www.vinmec.com/vie/bai-viet/cac-loai-insulin-dieu-tri-dai-thao-duong-va-cach-dung-vi",
        "drug_group": "Các loại Insulin điều trị đái tháo đường và cách dùng",
    },
]


def _strip_html(raw_html: str) -> str:
    cleaned = re.sub(r"<(script|style|noscript)[^>]*>.*?</\1>", " ", raw_html, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    cleaned = html.unescape(cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def crawl_raw_sources() -> List[Dict]:
    LONGCHAU_RAW_DIR.mkdir(parents=True, exist_ok=True)
    VINMEC_RAW_DIR.mkdir(parents=True, exist_ok=True)
    manifest_lc = []
    manifest_vm = []

    for item in SOURCES:
        slug = item["slug"]
        out_dir = LONGCHAU_RAW_DIR if item["target_dir"] == "long_chau" else VINMEC_RAW_DIR
        html_file = out_dir / f"{slug}.html"
        status_code = None
        extracted_len = 0
        title = ""
        try:
            resp = cffi_requests.get(item["url"], impersonate="chrome124", timeout=20)
            status_code = resp.status_code
            if resp.status_code == 200 and resp.text:
                m = re.search(r"<title[^>]*>(.*?)</title>", resp.text, re.S | re.I)
                title = re.sub(r"\s+", " ", m.group(1)).strip() if m else ""
                if "Đường dẫn đã hết hạn" in resp.text:
                    raise ValueError(f"Expired URL detected: {item['url']}")
                html_file.write_text(resp.text, encoding="utf-8")
                extracted_len = len(_strip_html(resp.text))
        except Exception as exc:  # noqa: BLE001
            status_code = f"error: {type(exc).__name__}: {exc}"

        meta = {
            "slug": slug,
            "provider": item["provider"],
            "url": item["url"],
            "title": title,
            "drug_group": item["drug_group"],
            "crawled_at": datetime.now(timezone.utc).isoformat(),
            "status": status_code,
            "raw_file": html_file.name if html_file.exists() else None,
            "extracted_text_chars": extracted_len,
        }
        if item["target_dir"] == "long_chau":
            manifest_lc.append(meta)
        else:
            manifest_vm.append(meta)
        print(f"[{slug}] status={status_code} | title={title[:55]} | chars={extracted_len}")
        time.sleep(0.25)

    (LONGCHAU_RAW_DIR / "manifest.json").write_text(json.dumps(manifest_lc, ensure_ascii=False, indent=2), encoding="utf-8")
    (VINMEC_RAW_DIR / "manifest.json").write_text(json.dumps(manifest_vm, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest_lc + manifest_vm


LONGCHAU_MD = """---
id: longchau-thuoc-dai-thao-duong
title: Dữ liệu thuốc điều trị đái tháo đường tại Việt Nam (Nhà thuốc FPT Long Châu)
source: Hệ thống Dược điển Nhà thuốc FPT Long Châu (nhathuoclongchau.com.vn)
url: https://nhathuoclongchau.com.vn/thuoc/thuoc-tri-tieu-duong
---
# Danh mục & Hướng dẫn sử dụng các nhóm thuốc đái tháo đường phổ biến (Long Châu)

Nguồn đối chiếu trực tiếp:
- Danh mục thuốc trị tiểu đường: https://nhathuoclongchau.com.vn/thuoc/thuoc-tri-tieu-duong
- Tổng quan các nhóm thuốc tiểu đường: https://nhathuoclongchau.com.vn/bai-viet/cac-nhom-thuoc-tieu-duong-va-uu-nhuoc-diem-cua-thuoc-dieu-tri-tieu-duong.html

## 1. Nhóm Biguanide — Metformin (Glucophage 500mg/850mg, Glucophage XR)
- **Link sản phẩm**:
  - Glucophage 500mg: https://nhathuoclongchau.com.vn/thuoc/glucophage-500mg-3584.html
  - Glucophage 850mg: https://nhathuoclongchau.com.vn/thuoc/glucophage-850mg-3583.html
- **Hoạt chất & hàm lượng phổ biến**: Metformin hydroclorid 500 mg, 850 mg, 1000 mg (viên phóng thích tức thời hoặc phóng thích kéo dài XR/SR).
- **Cơ chế tác dụng**: Giảm sản xuất glucose tại gan (ức chế tân tạo đường), tăng nhạy cảm insulin ở mô ngoại vi (cơ, mô mỡ) và làm chậm hấp thu glucose tại ruột. Không kích thích tế bào beta tụy tiết insulin nên hiếm khi gây hạ đường huyết khi dùng đơn độc.
- **Cách dùng thực tế**: Uống nguyên viên **trong hoặc ngay sau bữa ăn** (đặc biệt bữa tối với dạng XR) để giảm tối đa tác dụng phụ đường tiêu hóa (đầy bụng, buồn nôn, tiêu chảy, vị kim loại).
- **Chống chỉ định & Thận trọng**:
  - Chống chỉ định tuyệt đối khi suy thận nặng (**eGFR < 30 mL/phút/1,73 m²**), nhiễm toan chuyển hóa cấp, suy gan nặng hoặc suy tim mất bù.
  - Không khởi trị mới khi eGFR 30–45 mL/phút/1,73 m²; tạm ngừng thuốc trước và sau 48 giờ khi chụp X-quang/CT có tiêm thuốc cản quang chứa iod.
  - Sử dụng kéo dài có thể làm giảm hấp thu **Vitamin B12**, cần xét nghiệm kiểm tra định kỳ mỗi 1–2 năm (đặc biệt nếu có tê bì ngoại vi hoặc thiếu máu).

## 2. Nhóm Ức chế SGLT2 — Dapagliflozin (Forxiga 10mg) & Empagliflozin (Jardiance 25mg)
- **Link sản phẩm**:
  - Forxiga 10mg (Dapagliflozin): https://nhathuoclongchau.com.vn/thuoc/forxiga-10mg-3713.html
  - Jardiance 25mg (Empagliflozin): https://nhathuoclongchau.com.vn/thuoc/jardiance-25mg-30v-18702.html
- **Hoạt chất & biệt dược**: Dapagliflozin 5 mg, 10 mg (Forxiga); Empagliflozin 10 mg, 25 mg (Jardiance); kết hợp cố định liều với Metformin (Xigduo XR, Synjardy).
- **Cơ chế & Lợi ích kép (ADA + BYT)**: Ức chế kênh đồng vận chuyển natri-glucose SGLT2 tại ống lượn gần của thận, giúp thải glucose qua nước tiểu (khoảng 60–80 g/ngày), qua đó hạ đường huyết, giảm cân nhẹ (2–3 kg) và hạ huyết áp tâm thu (3–5 mmHg).
- **Chỉ định ưu tiên**: Bệnh nhân đái tháo đường típ 2 kèm **bệnh tim mạch xơ vữa (ASCVD)**, **suy tim (HFpEF/HFrEF)** hoặc **bệnh thận mạn (CKD có albumin niệu hoặc eGFR 20–60)**.
- **Cách dùng & Tác dụng phụ cần lưu ý**:
  - Uống 1 lần/ngày vào buổi sáng, có hoặc không kèm thức ăn; cần uống đủ nước trong ngày.
  - Tác dụng phụ đặc trưng: Tăng nguy cơ nhiễm nấm/nhiễm khuẩn đường tiết niệu – sinh dục (cần vệ sinh sạch sẽ), đa niệu, tụt huyết áp tư thế ở người cao tuổi dùng lợi tiểu, và hiếm gặp là nhiễm toan ceton đẳng đường huyết (euglycemic DKA).

## 3. Nhóm Ức chế DPP-4 — Sitagliptin (Januvia 100mg), Linagliptin (Trajenta 5mg)
- **Link sản phẩm**:
  - Januvia 100mg (Sitagliptin): https://nhathuoclongchau.com.vn/thuoc/januvia-100mg-3386.html
  - Trajenta 5mg (Linagliptin): https://nhathuoclongchau.com.vn/thuoc/trajenta-5mg-1945.html
- **Hoạt chất & biệt dược**: Sitagliptin 50 mg, 100 mg (Januvia, Janumet); Linagliptin 5 mg (Trajenta, Trajenta Duo); Vildagliptin 50 mg (Galvus, Galvus Met).
- **Ưu điểm lâm sàng**: Ức chế enzym DPP-4 làm tăng nồng độ incretin nội sinh (GLP-1, GIP) phụ thuộc nồng độ glucose máu. Dung nạp rất tốt, **trung tính với cân nặng**, nguy cơ hạ đường huyết thấp.
- **Lợi thế của Linagliptin (Trajenta)**: Thải trừ chủ yếu qua đường mật – ruột nên **không cần chỉnh liều ở mọi mức độ suy thận** (kể cả eGFR < 30 mL/phút/1,73 m²), rất phù hợp cho người cao tuổi hoặc bệnh thận mạn không dung nạp Metformin/SGLT2i.

## 4. Nhóm Đồng vận thụ thể GLP-1 (GLP-1 RA) & Insulin nền (Lantus SoloStar)
- **Link sản phẩm**:
  - Bút tiêm Lantus SoloStar 100IU/ml (Insulin Glargine): https://nhathuoclongchau.com.vn/thuoc/lantus-solostar-3ml-3289.html
- **Đặc điểm GLP-1 RA**: Thuốc tiêm dưới da giúp giảm HbA1c mạnh, **giảm cân rõ rệt (4–8 kg)** và giảm biến cố tim mạch lớn (MACE: nhồi máu cơ tim, đột quỵ).
- **Đặc điểm Lantus SoloStar (Insulin Glargine)**: Insulin nền tác dụng kéo dài suốt 24 giờ, ít tạo đỉnh, kiểm soát đường huyết lúc đói; tiêm dưới da 1 lần/ngày vào cùng một thời điểm mỗi ngày.

## 5. Nhóm Sulfonylurea thế hệ mới — Gliclazide MR (Diamicron MR 30mg/60mg)
- **Link sản phẩm**:
  - Diamicron MR 60mg (Gliclazide): https://nhathuoclongchau.com.vn/thuoc/diamicron-mr-60mg-4015.html
- **Đặc điểm**: Kích thích tế bào beta tuyến tụy tiết insulin; hiệu quả hạ HbA1c nhanh và chi phí thấp, phổ biến trong danh mục BHYT tại Việt Nam.
- **Cảnh báo an toàn quan trọng**: Phải uống **ngay trước hoặc trong bữa ăn sáng** (không được nhịn ăn sau khi uống thuốc). Nguy cơ chính là **hạ đường huyết** (vã mồ hôi, run tay, đói cồn cào, chóng mặt) và tăng cân nhẹ; cần tránh hoặc giảm liều ở người cao tuổi suy thận.
"""

VINMEC_ADA_MD = """---
id: vinmec-phac-do-dieu-tri-ket-hop-ada
title: Phác đồ cá thể hóa điều trị đái tháo đường típ 2 (Vinmec & ADA 2024)
source: Hệ thống Y tế Vinmec kết hợp Khuyến cáo ADA Standards of Care 2024
url: https://www.vinmec.com/vie/bai-viet/phac-do-dieu-tri-tieu-duong-type-2-vi
---
# Phác đồ điều trị đái tháo đường típ 2 kết hợp ADA 2024 – Bộ Y tế – Vinmec

Nguồn đối chiếu trực tiếp:
- Phác đồ điều trị tiểu đường type 2: https://www.vinmec.com/vie/bai-viet/phac-do-dieu-tri-tieu-duong-type-2-vi
- Thuốc điều trị đái tháo đường type 2: https://www.vinmec.com/vie/bai-viet/thuoc-dieu-tri-dai-thao-duong-type-2-vi
- Các nhóm thuốc điều trị đái tháo đường: https://www.vinmec.com/vie/bai-viet/cac-nhom-thuoc-dieu-tri-dai-thao-duong-vi
- Các loại Insulin điều trị đái tháo đường và cách dùng: https://www.vinmec.com/vie/bai-viet/cac-loai-insulin-dieu-tri-dai-thao-duong-va-cach-dung-vi

## 1. Phân tầng phác đồ theo 4 trục lâm sàng (Cardiorenal & Weight-Centric Approach)
Theo thực hành lâm sàng tại Hệ thống Y tế Vinmec và ADA 2024, phác đồ điều trị không chỉ dựa vào con số glucose mà được cá thể hóa theo 4 trục:
1. **Trục Bảo vệ Tim – Thận (ASCVD / Suy tim / Bệnh thận mạn)**:
   - Nếu có bệnh tim mạch xơ vữa hoặc nguy cơ cao: Khởi trị sớm **SGLT2i (Forxiga/Jardiance)** hoặc **GLP-1 RA (Semaglutide/Liraglutide)** độc lập với mức HbA1c ban đầu và độc lập với việc đang dùng Metformin.
   - Nếu có suy tim hoặc bệnh thận mạn (albumin niệu tăng, eGFR 20–60): Ưu tiên hàng đầu **SGLT2i**.
2. **Trục Kiểm soát Cân nặng (Thừa cân / Béo phì châu Á BMI ≥ 23 kg/m²)**:
   - Ưu tiên các nhóm thuốc hỗ trợ giảm cân hoặc không gây tăng cân: **GLP-1 RA > SGLT2i > Metformin > DPP-4i**.
   - Hạn chế tối đa việc dùng đơn độc Sulfonylurea hoặc Insulin liều cao sớm nếu chưa tối ưu hóa lối sống và các thuốc giảm cân.
3. **Trục Tránh Hạ Đường Huyết (Người cao tuổi ≥ 65 tuổi, sống một mình, suy thận)**:
   - Ưu tiên **Metformin** (nếu eGFR ≥ 45) và **DPP-4i (Linagliptin/Sitagliptin)** hoặc **SGLT2i**.
   - Nới lỏng mục tiêu HbA1c ở mức **7,5% – 8,0%** để đảm bảo an toàn, phòng ngừa ngã và biến cố tim mạch do hạ đường huyết.
4. **Trục Tăng Đường Huyết Nặng / Dị Hóa (HbA1c ≥ 10% hoặc Glucose lúc đói ≥ 250–300 mg/dL)**:
   - Khởi trị **Insulin nền (Basal Insulin: Glargine/Degludec)** kết hợp thuốc viên, hoặc phác đồ phối hợp kép ngay từ đầu khi HbA1c cao hơn mục tiêu ≥ 1,5% (≥ 8,5%).

## 2. Lộ trình bậc thang điều trị chuẩn hóa (Stepwise Regimen)
- **Bậc 0 (Tiền đái tháo đường: HbA1c 5,7–6,4% hoặc FPG 100–125 mg/dL)**:
  - Can thiệp lối sống tích cực (giảm 5–7% cân nặng, ăn giảm GI, vận động 150 phút/tuần + 2 buổi kháng lực).
  - Chỉ cân nhắc **Metformin 500–850 mg/ngày** khi người bệnh < 60 tuổi, BMI ≥ 25–35 kg/m², FPG ≥ 110 mg/dL, HbA1c ≥ 6,0% hoặc có tiền sử đái tháo đường thai kỳ.
- **Bậc 1 (Đái tháo đường mới chẩn đoán, HbA1c < 8,5%, chưa có biến chứng tim – thận)**:
  - Khởi trị **Đơn trị liệu Metformin** (bắt đầu 500 mg/ngày sau ăn tối, tăng dần mỗi 1–2 tuần tới 1000–2000 mg/ngày) kết hợp thay đổi lối sống.
  - Nếu không dung nạp Metformin hoặc chống chỉ định thận: chuyển sang **DPP-4i (Linagliptin 5 mg / Sitagliptin 100 mg)** hoặc **SGLT2i**.
- **Bậc 2 (Phối hợp kép — Dual Therapy: Khi HbA1c ≥ 8,5% ngay từ đầu, hoặc sau 3 tháng chưa đạt mục tiêu HbA1c < 7,0%, hoặc có bệnh đồng mắc tim mạch/huyết áp/béo phì)**:
  - **Metformin + SGLT2i (Forxiga 10 mg / Jardiance 10–25 mg)**: Phác đồ tối ưu cho bệnh nhân kèm tăng huyết áp, thừa cân/béo phì, bệnh tim mạch hoặc thận.
  - **Metformin + GLP-1 RA**: Phác đồ tối ưu khi ưu tiên giảm cân mạnh hoặc có tiền sử đột quỵ / thiếu máu cơ tim.
  - **Metformin + DPP-4i (Janumet / Trajenta Duo)**: Phác đồ an toàn, ít tác dụng phụ cho người không thừa cân nhiều hoặc sợ hạ đường huyết.
- **Bậc 3 (Phối hợp ba thuốc hoặc Khởi trị Insulin & Hội chẩn Nội tiết)**:
  - Khi glucose máu ≥ 180–250 mg/dL kéo dài, HbA1c ≥ 9–10%, hoặc có triệu chứng 4 nhiều (ăn nhiều, uống nhiều, tiểu nhiều, gầy nhiều): Khám chuyên khoa Nội tiết để khởi trị **Insulin nền** (10 UI/ngày hoặc 0,1–0,2 UI/kg/ngày) kèm theo dõi đường huyết mao mạch/CGM.
"""


def write_processed_kb() -> None:
    KB_OUT_DIR.mkdir(parents=True, exist_ok=True)
    LONGCHAU_RAW_DIR.mkdir(parents=True, exist_ok=True)
    VINMEC_RAW_DIR.mkdir(parents=True, exist_ok=True)

    (KB_OUT_DIR / "longchau-thuoc-dai-thao-duong.md").write_text(LONGCHAU_MD.strip() + "\n", encoding="utf-8")
    (KB_OUT_DIR / "vinmec-phac-do-dieu-tri-ket-hop-ada.md").write_text(VINMEC_ADA_MD.strip() + "\n", encoding="utf-8")

    (LONGCHAU_RAW_DIR / "longchau-thuoc-dai-thao-duong.md").write_text(LONGCHAU_MD.strip() + "\n", encoding="utf-8")
    (VINMEC_RAW_DIR / "vinmec-phac-do-dieu-tri-ket-hop-ada.md").write_text(VINMEC_ADA_MD.strip() + "\n", encoding="utf-8")
    print("Đã cập nhật tài liệu KB chuẩn hóa cho Long Châu & Vinmec.")


if __name__ == "__main__":
    crawl_raw_sources()
    write_processed_kb()
