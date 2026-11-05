# Tiểu luận final — kế hoạch và tiến độ

Đề tài: AI cho bệnh đái tháo đường (mạch chính) + miền đối chiếu từ code nhóm (Nguyễn Duy Nghĩa, TL01–TL03).
Khuôn chương theo bài của Trung (tieuluan_final_CT_07_TrungNT.861.pdf): lý thuyết → dữ liệu → cài đặt 3 cách
(NumPy / Keras / PyTorch) → kết quả → mô hình 5 lớp dạng hợp các hàm → notebook + giải thích ô → demo localhost → tiểu kết.

Quy tắc: mọi con số trong báo cáo phải đến từ lần chạy trên máy này (RTX 3050 4 GB, i5-11400H, 16 GB).
Kết quả cũ của Nghĩa được dời sang `outputs/metrics_nghia/` (tl01) — không dùng làm số liệu báo cáo.

Môi trường: `../.venv-gpu` (PyTorch 2.11 CUDA, TensorFlow-CPU 2.21, Keras 3.15, PyG, sentence-transformers).
Web app: `app/app.py` (cổng 8095), giao diện mới "Diabetes AI Lab"; mô-đun serve `dm.serve_chN` (mạch tiểu đường) + `tl3.serve_chN`.

## Chương (cập nhật 07:45)
| Ch | Trạng thái |
|---|---|
| 2 | XONG số liệu (tl01 outputs/metrics) + report/chapters/02b_ml_three_ways.tex; notebook ch2 đang chạy lại; còn: mục notebook + demo localhost (đang train lại model web tl03 ch2) |
| 3 | retina: scratch/torch_basic/torch_improved/tf_basic/tf_improved XONG (acc 0,50–0,55, cần tính QWK); đối chiếu mnist/fashion torch XONG, mnist tf+scratch đang chạy, cifar10 đang tải; serve_ch3 đã thêm retina; CHƯA viết |
| 4 | CGM torch+numpy XONG, keras đang chạy lstm5; report 04b_rnn_three_ways.tex đã viết (chờ [[KERAS_LSTM5]], sửa số tham số Keras theo json, [[CH4_DOMAINS]]); tl3 ch4 aapl/gold/ecommerce torch→keras đang chạy; dm/serve_ch4.py XONG |
| 5 | chưa: build chỉ mục diabetes tl3 ch5_build + eval; deploy agent |
| 6 | PubMed/Cora/CiteSeer đang chạy (tl3 ch6_loop, có PubMed GCN 0,789) ; cần serve PubMed |
| 7 | dm/ch7_medqa.py + scripts/run_ch7_medqa.py XONG; launch_ch7_when_free.sh chờ VRAM rồi chạy 150 câu × 4 hệ; GSM8K đối chiếu chưa chạy |

## Việc chung còn lại
- notebooks/chN_*.ipynb (chạy thật, giải thích 2 ô mỗi chương)
- dm/serve_ch3,4,6,7 cho web app; ảnh chụp demo bằng Playwright + Chrome hệ thống
- report: thêm mục 3 cách cài đặt, đối chiếu, hợp hàm 5 lớp, notebook, demo localhost; ghi nguồn kế thừa code nhóm
