# Đính chính báo cáo `docs/project01_CT_07_KienNT_v2.pdf` (đối chiếu code ngày 27/11/2026)

Mỗi mục ghi: **trang** (theo PDF), đoạn **hiện ghi**, đoạn **sửa thành** (tô CHỮ ĐỎ khi dán vào báo cáo) và **căn cứ** trong code.
Các con số đều đọc trực tiếp từ code / artifact, không ước lượng.

---

## A. Thay đổi mới của hệ thống (cần bổ sung vào báo cáo)

### A1. Dự đoán khi chưa có xét nghiệm — kết quả chỉ mang tính tương đối
**Trang 22, Mục quy trình, Bước 4.** Hiện ghi: *"Nếu dữ liệu không hợp lệ, thiếu chỉ số bắt buộc (chưa có cả đường huyết/HbA1c lẫn cặp tuổi + BMI) … Tác tử Điều phối dừng sớm luồng suy diễn…"*

Sửa thành:
> Bước 4: Nếu thiếu tuổi hoặc BMI, hoặc gặp điều kiện chặn luồng (blocking=True như đang mang thai, bệnh nhi), Tác tử Điều phối dừng sớm và hỏi bổ sung. **Khi chưa có xét nghiệm đường huyết/HbA1c, hệ thống vẫn chạy đủ hội đồng nhưng kết luận được gắn nhãn "ước tính từ yếu tố nguy cơ, chưa có xét nghiệm đường huyết"**; ConsensusCriticAgent ghi thêm một điểm (−) và ClarificationAgent hỏi kèm đường huyết lúc đói (không chặn kết quả).

Căn cứ: `PatientProfile.missing_required()` chỉ còn `age_years`, `bmi` (`src/schemas/patient_profile.py`); `NO_LABS_NOTE`, nhãn `status == "unknown"` trong `src/agents/consensus_critic_agent.py`.

**Trang 17, nguyên tắc 5 (Failure Awareness).** "thiếu chỉ số bắt buộc" → "thiếu tuổi hoặc BMI".

**Bổ sung (Chương an toàn, 10.1):**
> Mọi kết quả sàng lọc đều kèm ghi chú: *"Kết quả chỉ mang tính tương đối: mô hình ước tính nguy cơ so với quần thể có đặc điểm giống bạn, không phải chẩn đoán. Chẩn đoán đái tháo đường cần xét nghiệm và bác sĩ xác nhận."* (`RELATIVE_NOTE`, trường `relative_note` của kết luận hội đồng).

### A2. Lối sống có trong dữ liệu huấn luyện (rau, trái cây, giờ ngủ)
> Hồ sơ thu thêm 3 yếu tố lối sống **có trong dữ liệu huấn luyện**: ăn rau xanh mỗi ngày, ăn trái cây mỗi ngày (mô hình BRFSS, biến CDC `_VEGLT1A`, `_FRTLT1A`, chỉ có ở BRFSS 2021: 55.974/120.000 mẫu) và **giờ ngủ mỗi đêm** (mô hình NHANES, bảng SLQ: `SLD010H`/`SLD012`). Câu hỏi nằm ở bước 3 của form và được ClarificationAgent hỏi dần trong chat (tối đa 2 câu/lượt, sau câu hỏi đường huyết). Hội đồng ghi ít rau/trái cây, ngủ < 6 h hoặc ≥ 9 h là bằng chứng (+); ResponseAgent đưa lời khuyên tương ứng.
>
> **Đã thử và loại bỏ (ghi lại để bảo vệ khi vấn đáp):** (i) thói quen ăn uống của NHANES (đồ ăn nhanh – DBQ, đường/tinh bột/chất xơ – khẩu phần 24 h DR1TOT): mô hình học **ngược chiều** (ăn uống xấu → nguy cơ thấp hơn) vì người đã được chẩn đoán đã kiêng ăn theo lời bác sĩ (nhân quả ngược); che dữ liệu ăn uống của họ thì sự thiếu dữ liệu lại lộ nhãn (AUC 0,94 giả tạo). (ii) Giờ ngồi (PAQ `PAD680`): cũng ngược chiều. Vì vậy các câu hỏi nước ngọt, đồ ngọt, tinh bột, đồ chiên, ăn khuya đã bỏ khỏi hệ thống.

### A3. Chống LLM bịa số liệu
> Mọi con số do LLM bóc tách từ tin nhắn phải xuất hiện nguyên văn trong tin nhắn mới được nhận (`ExtractedClinicalData.to_profile(source_text=…)`). Lý do: khi kiểm thử, qwen2.5:3b tự tạo "chuỗi 10 lần đo glucose" từ câu trả lời "có", làm kết luận sai thành "Rất cao".
> Câu hỏi làm rõ do LLM tự sinh (loại OTHER) bị loại nếu trích nguyên cả tin nhắn làm bằng chứng hoặc hỏi lại chỉ số luật đã hỏi, và luôn xếp sau câu hỏi theo luật.

### A4. Hạ tầng LLM
> Ollama chạy trong container `ollama` của docker-compose, dùng GPU NVIDIA RTX 3050 4 GB qua nvidia-container-toolkit (model qwen2.5:3b, 100% GPU, ~2,4 GB VRAM). Backend gọi `http://ollama:11434`.

---

## B. Số liệu / mô tả sai so với code (sửa lại)

| Trang | Hiện ghi | Sửa thành | Căn cứ |
|---|---|---|---|
| 20 (công thức 4.4), 27, 29, 32 | GBDT "hiệu chỉnh Platt", "Platt Scaling và Temperature Calibration" | **Không có hiệu chỉnh xác suất.** `p_ens(x) = ½·(f_GBDT(x) + f_MLP(x))`, f_GBDT là `HistGradientBoostingClassifier` thuần | `src/models/tabular_ml.py` (`train`) |
| 20, 29, 32 | MLP "kết nối tắt (Residual)", "LayerNorm, SiLU", "Deep Residual MLP" | MLP 3 lớp ẩn **128–64–32**, mỗi lớp Linear → **BatchNorm1d → ReLU → Dropout(0,2)**; đầu vào 2d chiều (giá trị z-score + bit đánh dấu thiếu) | `src/models/deep_mlp.py` (`MLPNet`) |
| 14, 29, 32 | "Bi-LSTM" | **LSTM 2 lớp, một chiều**, hidden 64, nhìn lại 12 điểm, dự báo 4 bước (15–60 phút) | `notebook/model/rnn_glucose_lstm.pt` (cell=lstm, layers=2), `src/models/rnn_temporal.py` |
| 29, 32 | NHANES N = 23.005, tỷ lệ mắc 15,2% | **N = 20.646** người ≥ 20 tuổi, **tỷ lệ 19,8%** | `data/output/datasets/nhanes_diabetes.csv` |
| 29 | NHANES "d = 12 đặc trưng" | **d = 14**: age, sex, bmi, waist, systolic_bp, diastolic_bp, high_bp, high_chol, family_history, smoker, phys_activity, heart_disease, **stroke, sleep_hours** | `NHANES_FEATURES` (`src/models/datasets.py`) |
| 32 | NHANES GBDT 0,8167 / MLP 0,8250 | Sau khi thêm giờ ngủ: **GBDT ROC-AUC 0,8158, PR-AUC 0,5185, Brier 0,1249; MLP 0,8235, 0,5213, 0,1247; ensemble 0,8242** (trước 0,8252 — chênh trong sai số; kiểm định chéo 5-fold: 0,8295 → 0,8289) | `scripts/train_model.py --dataset nhanes` |
| 29 | BRFSS "7 đặc trưng", tỷ lệ 14,2%, "lọc sạch hoàn toàn khuyết thiếu" | **13 đặc trưng**: 11 cũ (age, sex, bmi, high_bp, high_chol, smoker, phys_activity, heart_disease, stroke, heavy_alcohol, gen_hlth) + **fruits_daily, vegetables_daily**; tỷ lệ **14,7%**; lọc sạch ở 11 đặc trưng cốt lõi, rau/trái cây được phép NaN | `BRFSS_FEATURES`, `scripts/fetch_brfss.py` |
| 29 | Masked-Feature Augmentation "nhân bản 3 tổ hợp che (vòng eo, huyết áp, mỡ máu)" | **Nhân đôi tập huấn luyện**: bản sao che ngẫu nhiên mỗi đặc trưng không cốt lõi với xác suất **p = 0,4**, giữ cố định age, bmi | `mask_augment` (`src/models/datasets.py`) |
| 32 | BRFSS GBDT ROC-AUC 0,8133 / Brier 0,1041; MLP 0,8128 / 0,1043 | Sau khi train lại có rau/trái cây: **GBDT ROC-AUC 0,8160, PR-AUC 0,4319, Brier 0,1020; MLP 0,8161, 0,4316, 0,1023; ensemble 0,8171** (trước đó 0,8154 / 0,8160) | `scripts/train_model.py --dataset brfss` |
| 32 | Random Forest 0,7941 / Extra Trees 0,7924 / Logistic 0,7888 | **RF 0,8185 (Brier 0,1243); ET 0,8104 (0,1271); LR 0,8099 (0,1280)** | `notebook/model/evaluation_leaderboard.csv` |
| 32 | TabNet ROC-AUC 0,7918 | Không có trong `evaluation_leaderboard.csv` → chạy lại notebook 06 để lấy số, hoặc ghi "chỉ thử nghiệm, chưa đánh giá trên tập kiểm thử chung". TabNet, RF, ET, LR **không được tác tử nào sử dụng** | `MODEL_REGISTRY` (`src/api/routes_system.py`) |
| 32 | ResNet-18: Accuracy 57,50%, QWK 0,6024 (RetinaMNIST) | **Accuracy 57,75%, macro-F1 0,4819, QWK 0,7003, AUC-OvR 0,8327** (tập test 400 ảnh) | `~/Projects/FinalEssay/lab/results/ch3.json` → `models.resnet18.test` |
| 32 | "Bi-LSTM", RMSE(+30p) 9,85 mg/dL, Clarke A+B 99,5%, "Chuỗi CGM 5 phút" | **LSTM 2 lớp** (bản triển khai `lstm2`), dữ liệu **ShanghaiT1DM + T2DM** (125 bản ghi, 112 bệnh nhân), bước 15 phút: **RMSE +15p 5,93 · +30p 13,67 · +60p 24,77 mg/dL; Clarke A+B (+30p) 99,75%** | `ch4.json` → `results.lstm2`, `deploy_key = "lstm2"` |
| 44 (kết luận) | "NHANES 2011–2014, BRFSS 2021", "MLP + Platt ROC-AUC 0.9469, Brier 0.0445", "LightGBM trên BRFSS (253.680 mẫu)", "QWK 0.8707", "BiLSTM + Attention RMSE 7.21" | NHANES **2011–2018**, BRFSS **2021–2023 (120.000 mẫu)**; ensemble GBDT + MLP **ROC-AUC 0,8252, Brier 0,1217**; **không có LightGBM**; ResNet-18 **QWK 0,7003**, LSTM **RMSE (+30p) 13,67 mg/dL** (hiện trang 32 và trang 44 ghi hai bộ số khác nhau, và cả hai đều không khớp file kết quả) | leaderboard, `src/models/` |

**Cần lưu ý khi vấn đáp:** mô hình NHANES/BRFSS **cố ý không dùng glucose/HbA1c** vì nhãn được gán bằng chính các xét nghiệm đó (`scripts/fetch_nhanes.py:95`: `diabetes = đã chẩn đoán | HbA1c ≥ 6,5 | FPG ≥ 126`) — đưa vào sẽ rò rỉ nhãn. Vì vậy hệ thống có hai nhánh độc lập: mô hình chấm điểm yếu tố nguy cơ và KnowledgeAgent đối chiếu xét nghiệm với ngưỡng ADA; khi lệch nhau (ví dụ mô hình 10%, FPG 165 mg/dL) hội đồng chất vấn và ưu tiên xét nghiệm.

