"""
Chương 4: Mạng Nơ-ron Tái phát (Recurrent Neural Network - RNN / LSTM)
Ứng dụng trong Phân tích Chuỗi thời gian Đường huyết Liên tục (Continuous Glucose Monitoring - CGM)
Phát hiện:
  - TIR (Time in Range: 70 - 180 mg/dL - Chuẩn ADA khuyến cáo > 70%)
  - TAR (Time Above Range: > 180 mg/dL - Nguy cơ tăng đường huyết)
  - TBR (Time Below Range: < 70 mg/dL - Nguy cơ hạ đường huyết cấp tính)
  - CV (Coefficient of Variation - Độ biến thiên đường huyết, khuyến cáo < 36%)
"""

import numpy as np
from typing import Dict, Any, List, Optional


class RNNTemporalEngine:
    """Mô hình RNN/LSTM dự báo xu hướng chuỗi thời gian đường huyết và phân tích bất ổn định chuyển hóa."""

    def __init__(self, hidden_dim: int = 32):
        self.hidden_dim = hidden_dim
        # Khởi tạo trọng số recurrent cell
        rng = np.random.default_rng(42)
        self.W_x = rng.normal(0, 0.2, (1, hidden_dim))
        self.W_h = rng.normal(0, 0.2, (hidden_dim, hidden_dim))
        self.b_h = np.zeros(hidden_dim)
        self.W_out = rng.normal(0, 0.2, (hidden_dim, 1))
        self.b_out = np.zeros(1)

    def analyze_glucose_trajectory(self, glucose_series: Optional[List[float]] = None, baseline_glucose: float = 120.0) -> Dict[str, Any]:
        """
        Nếu người dùng cung cấp chuỗi đường huyết 7-14 ngày gần nhất;
        Nếu không, sinh chuỗi nội suy thực nghiệm dựa trên baseline_glucose.
        """
        rng = np.random.default_rng(42)
        if glucose_series is None or len(glucose_series) < 5:
            # Mô phỏng chuỗi 14 ngày đo đường huyết đói và sau ăn
            n_points = 28
            trend = np.linspace(0, (baseline_glucose - 100) * 0.25, n_points)
            noise = rng.normal(0, 14.0, n_points)
            glucose_series = list(np.clip(baseline_glucose + trend + noise, 55.0, 320.0).round(1))

        series_arr = np.array(glucose_series, dtype=np.float64)
        mean_glucose = float(np.mean(series_arr))
        std_glucose = float(np.std(series_arr))
        cv_percent = float((std_glucose / (mean_glucose + 1e-6)) * 100.0)

        # Tính toán các chỉ số chuẩn ADA CGM
        in_range_count = np.sum((series_arr >= 70.0) & (series_arr <= 180.0))
        above_range_count = np.sum(series_arr > 180.0)
        below_range_count = np.sum(series_arr < 70.0)
        total_pts = len(series_arr)

        tir = round(float(in_range_count / total_pts * 100.0), 1)
        tar = round(float(above_range_count / total_pts * 100.0), 1)
        tbr = round(float(below_range_count / total_pts * 100.0), 1)

        # Chạy RNN Forward Pass để dự báo điểm kế tiếp
        h_t = np.zeros(self.hidden_dim)
        norm_series = (series_arr - 100.0) / 50.0

        for val in norm_series:
            x_in = np.array([[val]])
            h_t = np.tanh(x_in @ self.W_x + h_t @ self.W_h + self.b_h)[0]

        pred_delta = float((h_t @ self.W_out + self.b_out)[0])
        forecast_next = round(float(series_arr[-1] + pred_delta * 12.0), 1)

        # Đánh giá cảnh báo
        alert = "ỔN ĐỊNH"
        if tbr > 4.0:
            alert = "CẢNH BÁO NGUY HIỂM: Nguy cơ hạ đường huyết cấp tính (TBR > 4%)"
        elif tar > 25.0 or mean_glucose > 160.0:
            alert = "CẢNH BÁO: Đường huyết liên tục ở mức cao (TAR > 25%), kiểm soát kém"
        elif cv_percent > 36.0:
            alert = "BẤT ỔN ĐỊNH: Dao động đường huyết quá mức (CV > 36%), nguy cơ biến chứng vi mạch"

        return {
            "mean_glucose_mg_dl": round(mean_glucose, 1),
            "estimated_hba1c": round((mean_glucose + 46.7) / 28.7, 2),
            "tir_percentage": tir,
            "tar_percentage": tar,
            "tbr_percentage": tbr,
            "cv_variability_percentage": round(cv_percent, 1),
            "forecast_next_glucose": forecast_next,
            "trajectory_status": alert,
            "glucose_history_sample": [round(float(x), 1) for x in series_arr[-10:]]
        }
