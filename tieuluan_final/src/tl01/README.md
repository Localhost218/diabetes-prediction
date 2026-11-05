# Tiểu luận 01 — Học máy cơ bản, CNN và RNN trong hệ thống thông minh

Mã nguồn của báo cáo [`Report/TieuLuan 01/tieuluan01_CT_07_nghiand.600.pdf`](../../Report/TieuLuan%2001/tieuluan01_CT_07_nghiand.600.pdf).

| Chương | Mô hình | Dữ liệu |
|---|---|---|
| 2. Học máy cơ bản | Linear/Ridge, Logistic Regression, SVM/SVR, Random Forest; MLP trên NumPy, Keras, PyTorch | Diabetes Prediction (100.000), Vietnam Housing (30.229), Women's E-Commerce Reviews (22.641) |
| 3. CNN | Custom CNN, VGG-11, MobileNetV2, ResNet-18 (PyTorch), Custom CNN (Keras), 1D CNN; tích chập NumPy | SVHN, EuroSAT RGB, Diabetes |
| 4. RNN | Simple RNN, LSTM, GRU, BiLSTM trên PyTorch và Keras; LSTM NumPy | Giá VN30 (FPT, VCB, HPG, VIC) 2012–2026, KKBox Churn (100.000 người dùng × 31 ngày) |

## Bố cục

```
tl/              gói dùng chung: dữ liệu, mô hình, chỉ số
  data_tabular.py   data_vision.py   data_seq.py
  ml_classic.py     mlp_scratch.py   mlp_keras.py   mlp_torch.py
  cnn_scratch.py    cnn_keras.py     cnn_torch.py
  rnn_scratch.py    rnn_keras.py     rnn_torch.py   metrics.py   common.py
scripts/         các thí nghiệm (mỗi tệp một tiến trình, ghi outputs/metrics/*.json)
app/             Web App Flask sáu phân hệ + screenshots.py
data/            dữ liệu (tabular/ có sẵn; vision/ và seq/ tải bằng lệnh bên dưới)
models/          mô hình đã huấn luyện
outputs/         metrics/*.json, logs/
```

## Dựng lại từ đầu

```bash
pip install -r requirements.txt
python -m tl.data_vision          # tải SVHN + EuroSAT (~500 MB)
python -m tl.data_seq             # tải giá VN30 từ Yahoo Finance
# data/seq/kkbox_seq.npz dựng từ log KKBox ở Assignment 06 (a06/data_churn.py)

# Chương 2 — ba tiến trình MLP chạy song song: NumPy và Keras trên CPU, PyTorch trên GPU
python scripts/run_ch2_sklearn.py
python scripts/run_ch2_mlp.py --fw scratch & python scripts/run_ch2_mlp.py --fw keras & python scripts/run_ch2_mlp.py --fw torch
# Chương 3 — GPU và CPU song song
python scripts/run_ch3_torch.py   &   python scripts/run_ch3_keras.py
python scripts/run_ch3_scratch.py
# Chương 4
python scripts/run_ch4.py --fw torch   &   python scripts/run_ch4.py --fw keras
# Độ trễ đo lại khi máy rảnh, rồi hình
python scripts/bench.py
python scripts/make_eda.py && python scripts/make_figures.py

# Web App và ảnh chụp giao diện
python app/app.py                 # http://127.0.0.1:8090/
python app/screenshots.py
# Báo cáo
cd "../../Report/TieuLuan 01" && python build.py
```

Phân bổ phần cứng: PyTorch dùng GPU (RTX 4060 Laptop); TensorFlow ≥ 2.11 trên Windows không hỗ trợ GPU nên Keras chạy CPU; scikit-learn và NumPy chạy CPU, mỗi tiến trình giới hạn số luồng để chạy song song.
