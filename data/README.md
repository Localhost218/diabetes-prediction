# Quy Hoạch Dữ Liệu Dự Án (`MultiAgent/data`)

Thư mục `data/` được chia thành 2 nhánh tách biệt rõ ràng:
- **`data/download/`**: Lưu dữ liệu gốc tải về từ các nguồn chính thống (raw files: `.xpt`, `.zip`, `.csv`, `.html`, `.md`), **không qua chỉnh sửa**. Đối với các nguồn dữ liệu web (crawl trực tuyến từ Nhà thuốc FPT Long Châu, Hệ thống Y tế Vinmec, ADA, Bộ Y tế), đường dẫn gốc (URL) thực tế đã kiểm chứng (`HTTP 200`) được liệt kê chi tiết trong bảng dưới đây để kiểm soát trực tiếp.
- **`data/output/`**: Lưu toàn bộ dữ liệu **sau khi tiền xử lý, làm sạch và cấu trúc hóa** phục vụ huấn luyện mô hình ML/DL (`datasets/`), cơ sở tri thức RAG & Đồ thị tri thức (`kb/`), và cơ sở dữ liệu vận hành (`storage/`).

---

## 1. Cấu trúc thư mục `/data`

```text
data/
├── README.md                        # Tài liệu mô tả và kiểm soát nguồn dữ liệu (file này)
├── download/                        # DỮ LIỆU GỐC (RAW — KHÔNG QUA CHỈNH SỬA)
│   ├── ada/                         # Hiệp hội Đái tháo đường Hoa Kỳ (ADA Standards of Care 2024)
│   ├── bo_y_te_vn/                  # Bộ Y tế Việt Nam (QĐ 5481/QĐ-BYT & Vietnam Diabetes Dataset)
│   ├── nhanes/                      # CDC NHANES 2011–2018 (40 bảng xét nghiệm & khám lâm sàng .xpt)
│   ├── brfss/                       # CDC BRFSS 2021–2023 (2 kỳ khảo sát gần nhất LLCP2021XPT, LLCP2023XPT)
│   ├── vinmec/                      # Hệ thống Y tế Vinmec (HTML gốc crawl từ vinmec.com + manifest.json)
│   └── long_chau/                   # Nhà thuốc FPT Long Châu (HTML gốc crawl từ nhathuoclongchau.com.vn + manifest.json)
└── output/                          # DỮ LIỆU SAU KHI XỬ LÝ (PROCESSED)
    ├── datasets/                    # Bảng dữ liệu sạch dùng cho huấn luyện ML/DL (/notebook)
    │   ├── nhanes_diabetes.csv          # 11.723 mẫu có nhãn xét nghiệm HbA1c & FPG
    │   ├── brfss_diabetes.csv.gz        # 120.000 mẫu sạch 2021–2023 (0% missing)
    │   └── vietnam_diabetes_dataset.csv # 150.000 mẫu đặc trưng nhân trắc người Việt
    ├── kb/                          # Dữ liệu Knowledge Base & Knowledge Graph đã chuẩn hóa
    │   ├── kb_sources/                  # 12 tài liệu Markdown chuẩn hóa + curated_triples.json (47 bộ ba)
    │   ├── diabetes_kb.cypher           # Script khởi tạo Neo4j Knowledge Graph
    │   └── kb_export_full.zip           # Gói xuất toàn bộ chunks & đồ thị tri thức
    └── storage/                     # Cơ sở dữ liệu vận hành hệ thống
        ├── multiagent.db                # SQLite (bảng phiên chat, FTS5 full-text, nhanes/brfss samples)
        ├── qdrant/                      # Qdrant Vector DB (384-d multilingual embeddings)
        ├── hf/                          # Trọng số LLM & Embedding local (Qwen2.5-1.5B-Instruct, bge-m3)
        └── local_storage/               # Lưu trữ file bệnh án tải lên (fallback cho S3/SeaweedFS)
```

---

## 2. Danh mục 6 Nguồn Dữ liệu, Lý do lựa chọn & Bài toán giải quyết

| # | Nguồn dữ liệu | Thư mục trong `data/download/` | Lý do lựa chọn | Giải quyết bài toán / Phục vụ Agent nào? |
|---|---|---|---|---|
| **1** | **CDC NHANES (2011–2018)** *(National Health and Nutrition Examination Survey)* | [`download/nhanes/`](./download/nhanes/) *(40 file `.xpt`)* | Đây là bộ dữ liệu dịch tễ **chuẩn vàng (gold-standard)** vì có **xét nghiệm máu thực tế (`HbA1c` và `Glucose lúc đói FPG`)** kèm khám lâm sàng (`BMI`, `vòng eo`, `huyết áp`), phát hiện được cả ca **đái tháo đường chưa từng được chẩn đoán (undiagnosed)**. Từ chu kỳ 2011 có tách riêng nhóm **người gốc Á (`Non-Hispanic Asian`)**. | **Bài toán Dự đoán nguy cơ chính (Primary Risk Prediction)**: Huấn luyện các mô hình ML/DL (`HistGBDT`, `Deep MLP`, `TabNet Attention`, `RandomForest`, `ExtraTrees`, `LogisticRegression`) cho **`PredictionAgent`**. |
| **2** | **CDC BRFSS (2021–2023)** *(Behavioral Risk Factor Surveillance System)* | [`download/brfss/`](./download/brfss/) *(2 file `.zip` 2021 & 2023)* | Khảo sát sức khỏe cộng đồng quy mô lớn. Đã **loại bỏ hoàn toàn dữ liệu cũ (2011–2019)** vì lối sống và tiêu chuẩn dịch tễ đã thay đổi; chỉ giữ **2 chu kỳ gần nhất (2021 & 2023)** và lọc 120.000 bản ghi sạch 100% không khuyết thiếu. | **Bài toán Ý kiến thứ hai quy mô lớn & Kiểm chứng độ lệch đoàn hệ (Large-Cohort Second Opinion)**: Giúp **`PredictionAgent`** và **`ConsensusCriticAgent`** đối chiếu xem dự đoán trên NHANES có đồng thuận với quần thể khảo sát lớn gần đây hay không. |
| **3** | **Bộ Y tế Việt Nam (QĐ 5481/QĐ-BYT & VN Dataset)** | [`download/bo_y_te_vn/`](./download/bo_y_te_vn/) | Áp dụng đúng ngưỡng nhân trắc và tiêu chuẩn sàng lọc cho **người Việt Nam / châu Á** (thừa cân từ `BMI ≥ 23 kg/m²`, béo phì từ `BMI ≥ 25 kg/m²`, béo bụng khi vòng eo `nam ≥ 90 cm` / `nữ ≥ 80 cm`, sàng lọc thường quy từ `≥ 35 tuổi`). | **Bài toán Luật lâm sàng Việt Nam & Sàng lọc nội địa**: Phục vụ **`KnowledgeAgent`**, **`ClarificationAgent`** (kiểm tra phạm vi tuổi/thai kỳ) và đối chiếu nhân trắc người Việt. |
| **4** | **Hiệp hội Đái tháo đường Hoa Kỳ (ADA Standards of Care 2024)** | [`download/ada/`](./download/ada/) | Chuẩn mực quốc tế cập nhật nhất về ngưỡng chẩn đoán (`FPG ≥ 126 mg/dL`, `HbA1c ≥ 6.5%`, `OGTT 2h ≥ 200 mg/dL`), mục tiêu điều trị (`HbA1c < 7.0%`, `TIR CGM > 70%`) và quy tắc cấp cứu đường huyết (`< 54 mg/dL` hoặc `≥ 300 mg/dL`). | **Bài toán Đối chiếu tiêu chuẩn chẩn đoán & Rào chắn an toàn (Safety Guardrails)**: Phục vụ **`KnowledgeAgent`** (Vector RAG + GraphRAG) và **`ConsensusCriticAgent`** (phân xử khi ML và xét nghiệm mâu thuẫn). |
| **5** | **Hệ thống Y tế Vinmec** *(Phác đồ điều trị ĐTĐ típ 2 & Insulin)* | [`download/vinmec/`](./download/vinmec/) | Cung cấp phác đồ điều trị thực hành tại bệnh viện Việt Nam, kết hợp cùng ADA 2024 để **phân tầng điều trị theo 4 trục cá thể hóa**: (1) Bảo vệ Tim–Thận, (2) Kiểm soát cân nặng BMI châu Á, (3) Tránh hạ đường huyết ở người ≥ 65 tuổi, và (4) Khởi trị Insulin nền khi tăng đường huyết nặng/dị hóa. | **Bài toán Tối ưu hóa lộ trình điều trị bậc thang (Stepwise Clinical Regimen)**: Kết hợp với thuật toán **Q-Learning** trong **`TreatmentRLAgent`** để đề xuất phác đồ Bậc 0 → Bậc 3 sát thực tế lâm sàng. |
| **6** | **Nhà thuốc FPT Long Châu** *(Dược điển thuốc đái tháo đường tại VN)* | [`download/long_chau/`](./download/long_chau/) | Bổ sung thông tin dược lâm sàng thực tế của các biệt dược lưu hành phổ biến tại Việt Nam (*Glucophage 500mg/850mg, Forxiga 10mg, Jardiance 25mg, Januvia 100mg, Trajenta 5mg, Diamicron MR 60mg, Lantus SoloStar*): hàm lượng, thời điểm uống (trước/trong/sau ăn), tác dụng phụ và chống chỉ định suy thận (`eGFR < 30`). | **Bài toán Tư vấn thuốc tham chiếu & Cảnh báo chống chỉ định (Drug Knowledge & Contraindication Check)**: Phục vụ **`TreatmentRLAgent`** (`reference_drugs_vn`, `monitoring_plan`) và **`KnowledgeAgent`** (quan hệ `TREATED_BY`, `CONTRAINDICATED_IN` trên Neo4j). |

---

## 3. Bảng Kiểm Soát Link Nguồn Trực Tuyến Đã Kiểm Chứng (Verified Live URLs)

Toàn bộ các đường dẫn dưới đây đã được kiểm chứng hoạt động thực tế (`HTTP 200 OK`) và được cấu hình trực tiếp trong [`scripts/crawl_longchau_vinmec.py`](../scripts/crawl_longchau_vinmec.py) và [`scripts/ingest_kb.py`](../scripts/ingest_kb.py):

### A. Nguồn Dữ liệu Crawl về Thuốc & Phác đồ Điều trị (Nhà thuốc FPT Long Châu & Vinmec)
- **Nhà thuốc FPT Long Châu (`nhathuoclongchau.com.vn`)**:
  - Danh mục thuốc trị tiểu đường: <https://nhathuoclongchau.com.vn/thuoc/thuoc-tri-tieu-duong>
  - Bài viết chuyên môn — *Các nhóm thuốc tiểu đường và ưu nhược điểm*: <https://nhathuoclongchau.com.vn/bai-viet/cac-nhom-thuoc-tieu-duong-va-uu-nhuoc-diem-cua-thuoc-dieu-tri-tieu-duong.html>
  - Bài viết chuyên môn — *7 loại thuốc tiểu đường tốt nhất hiện nay*: <https://nhathuoclongchau.com.vn/bai-viet/top-7-loai-thuoc-tieu-duong-tot-nhat-hien-nay-va-cach-kiem-soat-tieu-duong-hieu-qua-69859.html>
  - Nhóm Biguanide — *Glucophage 500mg (Metformin)*: <https://nhathuoclongchau.com.vn/thuoc/glucophage-500mg-3584.html>
  - Nhóm Biguanide — *Glucophage 850mg (Metformin)*: <https://nhathuoclongchau.com.vn/thuoc/glucophage-850mg-3583.html>
  - Nhóm Ức chế SGLT2 — *Forxiga 10mg (Dapagliflozin)*: <https://nhathuoclongchau.com.vn/thuoc/forxiga-10mg-3713.html>
  - Nhóm Ức chế SGLT2 — *Jardiance 25mg (Empagliflozin)*: <https://nhathuoclongchau.com.vn/thuoc/jardiance-25mg-30v-18702.html>
  - Nhóm Ức chế DPP-4 — *Januvia 100mg (Sitagliptin)*: <https://nhathuoclongchau.com.vn/thuoc/januvia-100mg-3386.html>
  - Nhóm Ức chế DPP-4 — *Trajenta 5mg (Linagliptin)*: <https://nhathuoclongchau.com.vn/thuoc/trajenta-5mg-1945.html>
  - Nhóm Sulfonylurea — *Diamicron MR 60mg (Gliclazide)*: <https://nhathuoclongchau.com.vn/thuoc/diamicron-mr-60mg-4015.html>
  - Nhóm Insulin nền — *Bút tiêm Lantus SoloStar 100IU/ml (Insulin Glargine)*: <https://nhathuoclongchau.com.vn/thuoc/lantus-solostar-3ml-3289.html>
- **Hệ thống Y tế Vinmec (`vinmec.com`)**:
  - Phác đồ điều trị tiểu đường type 2: <https://www.vinmec.com/vie/bai-viet/phac-do-dieu-tri-tieu-duong-type-2-vi>
  - Thuốc điều trị đái tháo đường type 2: <https://www.vinmec.com/vie/bai-viet/thuoc-dieu-tri-dai-thao-duong-type-2-vi>
  - Các nhóm thuốc điều trị đái tháo đường: <https://www.vinmec.com/vie/bai-viet/cac-nhom-thuoc-dieu-tri-dai-thao-duong-vi>
  - Đặc điểm các thuốc điều trị đái tháo đường tuýp 2: <https://www.vinmec.com/vie/bai-viet/dac-diem-cac-thuoc-dieu-tri-dai-thao-duong-tuyp-2-vi>
  - Các loại Insulin điều trị đái tháo đường và cách dùng: <https://www.vinmec.com/vie/bai-viet/cac-loai-insulin-dieu-tri-dai-thao-duong-va-cach-dung-vi>
  - Hướng dẫn sử dụng bút tiêm Insulin: <https://www.vinmec.com/vie/bai-viet/huong-dan-su-dung-tiem-insulin-vi>

### B. Nguồn Dữ liệu Bảng (Tabular Datasets — CDC)
- **CDC NHANES (2011–2018)**:
  - Portal chính thức: <https://wwwn.cdc.gov/nchs/nhanes/>
  - Endpoint tải file `.xpt` tự động (`scripts/fetch_nhanes.py`): `https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{year}/DataFiles/{DEMO,BMX,BPX,BPQ,DIQ,MCQ,SMQ,PAQ,GHB,GLU}_{G,H,I,J}.xpt`
- **CDC BRFSS (2021 & 2023)**:
  - Kỳ khảo sát 2023: <https://www.cdc.gov/brfss/annual_data/annual_2023.html> (`https://www.cdc.gov/brfss/annual_data/2023/files/LLCP2023XPT.zip`)
  - Kỳ khảo sát 2021: <https://www.cdc.gov/brfss/annual_data/annual_2021.html> (`https://www.cdc.gov/brfss/annual_data/2021/files/LLCP2021XPT.zip`)

### C. Nguồn Hướng dẫn Y khoa Chuẩn (ADA 2024 & Bộ Y tế Việt Nam)
- **ADA Standards of Care in Diabetes — 2024**:
  - Toàn văn Supplement 1 (*Diabetes Care*): <https://diabetesjournals.org/care/issue/47/Supplement_1>
  - Section 2 (Diagnosis and Classification of Diabetes — PubMed): <https://pubmed.ncbi.nlm.nih.gov/38078589/>
  - Section 6 (Glycemic Goals and Hypoglycemia — PubMed): <https://pubmed.ncbi.nlm.nih.gov/38078586/>
  - Section 9 (Pharmacologic Approaches to Glycemic Treatment — PubMed): <https://pubmed.ncbi.nlm.nih.gov/38078590/>
- **Bộ Y tế Việt Nam — Quyết định 5481/QĐ-BYT (30/12/2020)**:
  - Cổng thông tin Cục Quản lý Khám, chữa bệnh (Bộ Y tế): <https://kcb.vn/>
  - Toàn văn Hướng dẫn chẩn đoán và điều trị đái tháo đường típ 2 (QĐ 5481/QĐ-BYT): <https://thuvienphapluat.vn/van-ban/The-thao-Y-te/Quyet-dinh-5481-QD-BYT-2020-tai-lieu-chuyen-mon-Huong-dan-chan-doan-va-dieu-tri-dai-thao-duong-tip-2-460179.aspx>
