# Tiểu luận 03 — Phân tích các hệ thống thông minh

Mã nguồn của báo cáo [`Report/TieuLuan 03/tieuluan03_CT_07_nghiand.pdf`](../../Report/TieuLuan%2003/tieuluan03_CT_07_nghiand.pdf).

| Chương | Nội dung | Mã chính |
|---|---|---|
| 1. Lịch sử AI | Các giai đoạn phát triển, mùa đông AI, kỷ nguyên mô hình nền tảng | — |
| 2. Học máy | Tiểu đường (100.000 hồ sơ), giá nhà, nhận xét khách hàng; NumPy từ đầu, K-means, PCA | `tl3/ch2_*.py`, `scripts/ch2_*.py` |
| 3. CNN | MNIST, Fashion-MNIST, Rock/Paper/Scissors, CIFAR-10; NumPy, PyTorch, TensorFlow | `tl3/ch3_*.py`, `scripts/ch3_train.py` |
| 4. RNN | Doanh thu thương mại điện tử, cổ phiếu AAPL, quỹ vàng GLD; RNN/LSTM/GRU/BiLSTM | `tl3/ch4_*.py`, `scripts/ch4_train.py` |
| 5. KG và RAG | KB, KG, chỉ mục cho ba miền; 6 bộ truy xuất; LLM, Naive/Advanced RAG, tác tử định tuyến | `tl3/ch5_*.py`, `scripts/ch5_*.py` |
| 6. GNN | GCN, GraphSAGE, GAT trên Cora/CiteSeer; độ sâu, nhiễu cạnh, dự đoán liên kết | `tl3/ch6_gnn.py`, `scripts/ch6_run.py` |
| 7. Đa tác tử | Direct, CoT, PAL và hệ đa tác tử trên GSM8K với Qwen2.5-1.5B | `tl3/ch7_*.py`, `scripts/ch7_run.py` |

## Bố cục

```
tl3/             gói dùng chung
  common.py  gpu.py  plot_style.py  texfmt.py  llm.py     đường dẫn, khóa GPU, kiểu hình, bảng LaTeX, LLM cục bộ
  chN_*.py                                                 dữ liệu, mô hình, đánh giá của từng chương
  serve_chN.py                                             hợp đồng phục vụ Web App: SPECS + predict()
scripts/         các thí nghiệm (mỗi tệp một tiến trình, ghi outputs/metrics/*.json), hình và bảng
app/             Web App Flask (app.py, templates/, static/) + screenshots.py
data/            dữ liệu đã tải (CIFAR-10, ảnh .npz và Planetoid không đưa lên git, tải lại bằng lệnh bên dưới)
models/          mô hình đã huấn luyện; chỉ mục miền tiểu đường của Chương 5 dựng lại bằng ch5_build.py
outputs/         metrics/*.json (nguồn của mọi con số \SL{...} trong báo cáo), logs/
```

LLM và mô hình embedding dùng chung với Tiểu luận 02 (`src/TieuLuan 02/models/hf/`, tải bằng
`python "../TieuLuan 02/scripts/download_models.py"`).

## Dựng lại từ đầu

```bash
pip install -r requirements.txt
# Dữ liệu (kagglehub ẩn danh; Planetoid tự tải khi chạy ch6_run.py)
python scripts/ch3_download.py
python scripts/ch4_download.py
# Chương 2 (CPU)
python scripts/ch2_diabetes.py && python scripts/ch2_house.py && python scripts/ch2_comments.py
python scripts/ch2_scratch_run.py && python scripts/ch2_unsup.py && python scripts/ch2_tree_demo.py
python scripts/ch2_figs.py && python scripts/ch2_theory_figs.py && python scripts/ch2_tables.py
# Chương 3: NumPy và TF trên CPU, PyTorch trên GPU
python scripts/ch3_train.py --fw scratch
python scripts/ch3_train.py --fw torch
python scripts/ch3_train.py --fw tf --threads 8
python scripts/ch3_collect.py && python scripts/ch3_figures.py && python scripts/ch3_tables.py
# Chương 4
python scripts/ch4_eda.py && python scripts/ch4_scratch_run.py
python scripts/ch4_train.py --fw torch && python scripts/ch4_train.py --fw keras
python scripts/ch4_extras.py && python scripts/ch4_figures.py && python scripts/ch4_tables.py
# Chương 5 (embedding và LLM trên GPU)
for d in stock gold diabetes; do python scripts/ch5_build.py $d; done
for d in stock gold diabetes; do python scripts/ch5_eval_retrieval.py $d; done
python scripts/ch5_eval_retrieval.py merge
for d in stock gold diabetes; do python scripts/ch5_eval_agent.py $d; done
python scripts/ch5_figures.py && python scripts/ch5_tables.py
# Chương 6 và 7 (GPU)
python scripts/ch6_loop.py && python scripts/ch6_figs.py && python scripts/ch6_tables.py
python scripts/ch7_loop.py && python scripts/ch7_eval.py && python scripts/ch7_figs.py && python scripts/ch7_tables.py

# Web App và ảnh chụp giao diện (desktop và điện thoại)
python app/app.py                     # http://127.0.0.1:8093/
python app/screenshots.py
# Báo cáo
cd "../../Report/TieuLuan 03" && python build.py --inline    # --chapter N: dựng thử một chương
```

## Phân bổ phần cứng

GPU (RTX 4060 Laptop, 8 GB) chạy CNN và RNN bằng PyTorch, GNN, các mô hình embedding, reranker và Qwen2.5-1.5B bf16.
Các tác vụ nặng giữ khóa `outputs/gpu_heavy.lock` (`tl3.gpu.heavy()`) nên mỗi lúc chỉ một LLM nằm trên GPU; tác vụ nhẹ
chỉ giới hạn phần bộ nhớ GPU (`tl3.gpu.cap()`). CPU (32 luồng) chạy scikit-learn, XGBoost, LightGBM, cài đặt NumPy từ
đầu, TensorFlow (bản Windows không hỗ trợ GPU), đồ thị tri thức, BM25, vẽ hình và biên dịch LaTeX; mỗi tiến trình
gọi `limit_threads(n)` để nhiều thí nghiệm chạy song song mà không tranh luồng.
