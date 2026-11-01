"""
Chương 3: Mạng Nơ-ron Tích chập (Convolutional Neural Network - CNN)
Ứng dụng trong Tầm soát Biến chứng Võng mạc Đái tháo đường (Diabetic Retinopathy - DR) từ ảnh đáy mắt
Theo tiêu chuẩn Phân loại Lâm sàng Quốc tế (ICDR):
  - 0: Không tổn thương (No DR)
  - 1: Tổn thương nhẹ (Mild NPDR - Vi phình mạch)
  - 2: Tổn thương vừa (Moderate NPDR - Xuất huyết võng mạc, xuất tiết cứng)
  - 3: Tổn thương nặng (Severe NPDR - Tắc mạch diện rộng, xuất tiết mềm)
  - 4: Bệnh võng mạc tăng sinh (Proliferative DR - Tăng sinh tân mạch, nguy kịch)
"""

import numpy as np
from typing import Dict, Any, Optional

DR_CLASSES = [
    {"stage": 0, "code": "NO_DR", "name_vi": "Võng mạc bình thường", "severity": "None"},
    {"stage": 1, "code": "MILD_NPDR", "name_vi": "Bệnh võng mạc ĐTĐ giai đoạn nhẹ", "severity": "Low"},
    {"stage": 2, "code": "MODERATE_NPDR", "name_vi": "Bệnh võng mạc ĐTĐ giai đoạn trung bình", "severity": "Medium"},
    {"stage": 3, "code": "SEVERE_NPDR", "name_vi": "Bệnh võng mạc ĐTĐ giai đoạn nặng", "severity": "High"},
    {"stage": 4, "code": "PDR", "name_vi": "Bệnh võng mạc tăng sinh (Nguy kịch mù lòa)", "severity": "Critical"},
]


class CNNRetinopathyEngine:
    """Mô hình CNN mô phỏng phân tích ảnh tổn thương đáy mắt và đánh giá dấu hiệu thị giác."""

    def __init__(self):
        # Trọng số Conv filters đặc trưng (microaneurysms, hemorrhages, exudates, neovascularization)
        self.feature_weights = {
            "microaneurysms": 0.25,
            "hemorrhages": 0.35,
            "hard_exudates": 0.30,
            "cotton_wool_spots": 0.45,
            "neovascularization": 0.85,
        }

    def analyze_fundus(self, image_features: Optional[Dict[str, float]] = None, patient_profile: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Nhận diện tổn thương vi mạch qua lớp tích chập không gian:
        Nếu có ảnh hoặc chỉ số khám mắt, tính toán theo biểu hiện;
        Nếu chưa có ảnh chụp, dùng ước lượng rủi ro vi mạch dựa trên thời gian mắc bệnh, huyết áp và HbA1c.
        """
        if image_features is None:
            image_features = {}

        # Nếu người dùng nhập thông số khám đáy mắt cụ thể
        ma = float(image_features.get("microaneurysms", 0.0))
        hem = float(image_features.get("hemorrhages", 0.0))
        ex = float(image_features.get("hard_exudates", 0.0))
        cws = float(image_features.get("cotton_wool_spots", 0.0))
        nv = float(image_features.get("neovascularization", 0.0))

        # Nếu không có ảnh chụp trực tiếp, ước tính theo các yếu tố sinh lý tim mạch từ patient_profile
        if not any([ma, hem, ex, cws, nv]) and patient_profile:
            glucose_high = float(patient_profile.get("Glucose", 110.0)) >= 140
            hba1c_high = float(patient_profile.get("HbA1c", 5.6)) >= 7.0
            bp_high = float(patient_profile.get("HighBP", 0)) == 1
            age_high = float(patient_profile.get("Age", 5)) >= 8
            
            score = (0.35 if glucose_high else 0.0) + (0.30 if hba1c_high else 0.0) + (0.25 if bp_high else 0.0) + (0.15 if age_high else 0.0)
            ma = min(1.0, score * 0.8)
            hem = min(1.0, max(0.0, score - 0.3) * 1.2)
            ex = min(1.0, max(0.0, score - 0.5) * 1.5)

        # Mô phỏng Dense output logits từ Convolutional feature map
        logits = np.zeros(5)
        logits[0] = 2.0 - 2.5 * (ma + hem)
        logits[1] = 1.8 * ma - 1.2 * nv
        logits[2] = 2.2 * hem + 1.5 * ex
        logits[3] = 2.5 * cws + 1.8 * hem
        logits[4] = 4.0 * nv + 2.0 * cws

        # Softmax
        exp_l = np.exp(logits - np.max(logits))
        probs = exp_l / np.sum(exp_l)
        pred_stage = int(np.argmax(probs))

        stage_info = DR_CLASSES[pred_stage]

        return {
            "dr_stage": pred_stage,
            "stage_code": stage_info["code"],
            "diagnosis_vi": stage_info["name_vi"],
            "severity": stage_info["severity"],
            "stage_probabilities": [round(float(p), 4) for p in probs],
            "detected_biomarkers": {
                "microaneurysms": round(ma, 2),
                "hemorrhages": round(hem, 2),
                "hard_exudates": round(ex, 2),
                "neovascularization": round(nv, 2),
            },
            "clinical_advice": (
                "Chưa phát hiện tổn thương võng mạc, duy trì tái khám mắt định kỳ 1 năm/lần."
                if pred_stage == 0 else
                ("Có dấu hiệu vi phình mạch sớm, cần kiểm soát chặt chẽ đường huyết và huyết áp."
                 if pred_stage <= 2 else
                 "NGUY HIỂM: Tổn thương võng mạc tiến triển nặng, cần can thiệp laser quang đông hoặc tiêm anti-VEGF ngay.")
            )
        }
