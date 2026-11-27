# Diabetes Prediction — giao diện web

Giao diện React + Vite cho hệ thống hội chẩn đa tác tử sàng lọc đái tháo đường.

## Không gian hội chẩn

- **Bên trái:** nhập tuổi, giới tính, chiều cao, cân nặng, BMI, vòng eo, đường huyết lúc đói và HbA1c. Mục mở rộng có huyết áp, cholesterol, eGFR và tiền sử. Bên dưới biểu mẫu là dự đoán, bằng chứng giải thích, kế hoạch can thiệp, thuốc tham khảo và lịch theo dõi do backend trả về.
- **Bên phải:** chatbot hỗ trợ câu hỏi tiếp nối, tải bệnh án, camera và nhập liệu giọng nói.
- Hai phần dùng chung phiên `/api/sessions` và luồng NDJSON `/api/chat/stream`. Kết quả hội chẩn gần nhất được giữ khi người dùng hỏi tiếp và được khôi phục khi chọn lại phiên.
- Trên màn hình dưới 960 px, hai phần xếp dọc. Mỗi phần có vùng cuộn riêng trên desktop.

Biểu mẫu yêu cầu tuổi, BMI (hoặc chiều cao + cân nặng) và ít nhất một xét nghiệm đường huyết. BMI được tự tính khi nhập chiều cao và cân nặng; BMI được người dùng cập nhật qua chat vẫn được giữ. Tiền sử có ba trạng thái **chưa cung cấp / có / không**, để thông tin chưa biết không bị coi là câu trả lời phủ định.

`src/services/predictionForm.js` đồng bộ nhãn, đơn vị và giới hạn với `src/schemas/patient_profile.py` của backend. Biểu mẫu chuyển các dữ liệu đã nhập thành tin nhắn có đơn vị rõ ràng; các câu khẳng định được đặt trước các câu phủ định để tương thích với bộ trích xuất ngữ cảnh của backend.

## Chạy phát triển

Khởi động backend tại cổng 8000 theo [README dự án](../README.md), sau đó chạy các lệnh sau trong `web/`:

```bash
npm ci
npm run dev
```

Mở `http://localhost:5173`. Vite proxy các yêu cầu `/api` đến `http://localhost:8000`.

## Kiểm tra và build

```bash
npm test        # Kiểm tra BMI, dữ liệu hồ sơ, validation và tin nhắn từ biểu mẫu
npm run lint
npm run build  # Xuất bản build vào web/dist/
```
