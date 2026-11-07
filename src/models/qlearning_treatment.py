"""
Học Tăng Cường (Reinforcement Learning) - Thuật toán Q-Learning (Off-Policy TD Control)
Ứng dụng trong Tối ưu hóa Phác đồ Can thiệp và Điều trị Đái tháo đường Tuần tự
Kết hợp hướng dẫn lâm sàng ADA 2024, Quyết định 5481/QĐ-BYT, Phác đồ Vinmec và Dược điển Nhà thuốc Long Châu.
"""

import numpy as np
from typing import Dict, List, Any


ACTION_DESCRIPTIONS: List[Dict[str, Any]] = [
    {
        "action_id": 0,
        "name": "Can thiệp lối sống tích cực (Lifestyle Modification — ADA/Vinmec)",
        "details": (
            "Áp dụng chế độ ăn giảm chỉ số đường huyết (GI thấp, đĩa ăn lành mạnh 1/2 rau xanh - 1/4 đạm - 1/4 tinh bột nguyên hạt), "
            "vận động thể lực ≥ 150 phút/tuần (đi bộ nhanh, bơi, đạp xe) kết hợp 2 buổi tập kháng lực; mục tiêu giảm 5–7% cân nặng nếu BMI ≥ 23 kg/m²."
        ),
        "reference_drugs_vn": [],
        "monitoring_plan": "Tái kiểm tra Glucose máu lúc đói (FPG) và HbA1c sau 3–6 tháng; theo dõi cân nặng và vòng eo hàng tuần.",
        "invasiveness": "Thấp",
        "cost": "Thấp",
    },
    {
        "action_id": 1,
        "name": "Đơn trị liệu Thuốc Đầu tay (Metformin Monotherapy — ADA/BYT/Long Châu)",
        "details": (
            "Trao đổi với bác sĩ khởi trị Metformin (biệt dược phổ biến: Glucophage 500mg/850mg hoặc Glucophage XR uống trong/ngay sau bữa tối "
            "để giảm đầy bụng, tiêu chảy) kết hợp duy trì lối sống. Nếu không dung nạp Metformin hoặc suy thận (eGFR < 45), bác sĩ có thể thay bằng "
            "nhóm ức chế DPP-4 (Linagliptin/Trajenta 5mg hoặc Sitagliptin/Januvia 100mg)."
        ),
        "reference_drugs_vn": [
            "Metformin HCl (Glucophage 500mg/850mg, Glucophage XR 500mg/750mg/1000mg, Panfor SR)",
            "Thay thế khi không dung nạp: Linagliptin (Trajenta 5mg — không cần chỉnh liều ở bệnh thận) hoặc Sitagliptin (Januvia 100mg)",
        ],
        "monitoring_plan": "Kiểm tra chức năng thận (eGFR) trước khi khởi trị, đánh giá lại HbA1c sau 3 tháng (mục tiêu < 7,0%) và định lượng Vitamin B12 định kỳ mỗi năm.",
        "invasiveness": "Trung bình",
        "cost": "Vừa phải",
    },
    {
        "action_id": 2,
        "name": "Phác đồ Phối hợp Kép Bảo vệ Tim – Thận – Cân nặng (Metformin + SGLT2i / GLP-1 RA)",
        "details": (
            "Trao đổi với bác sĩ về phác đồ phối hợp sớm theo khuyến cáo ADA 2024 & Vinmec: Metformin kết hợp ức chế SGLT2 "
            "(Dapagliflozin/Forxiga 10mg hoặc Empagliflozin/Jardiance 10–25mg uống buổi sáng) giúp hạ đường huyết, giảm cân 2–3 kg, hạ huyết áp "
            "và bảo vệ tim – thận; hoặc phối hợp đồng vận GLP-1 (Semaglutide/Liraglutide) khi ưu tiên giảm cân mạnh."
        ),
        "reference_drugs_vn": [
            "Metformin + Dapagliflozin (Forxiga 10mg / viên phối hợp Xigduo XR)",
            "Metformin + Empagliflozin (Jardiance 10mg/25mg / viên phối hợp Synjardy)",
            "Metformin + DPP-4i (Janumet / Trajenta Duo — khi ưu tiên an toàn, tránh hạ đường huyết ở người lớn tuổi)",
            "Đồng vận thụ thể GLP-1 (Liraglutide/Victoza, Semaglutide/Ozempic — khi BMI ≥ 25 và ưu tiên giảm cân/bảo vệ mạch vành)",
        ],
        "monitoring_plan": "Theo dõi eGFR, huyết áp, cân nặng, HbA1c mỗi 3 tháng; uống đủ nước và giữ vệ sinh đường tiết niệu – sinh dục khi dùng nhóm SGLT2i.",
        "invasiveness": "Cao",
        "cost": "Cao",
    },
    {
        "action_id": 3,
        "name": "Hội chẩn Chuyên khoa Nội tiết Khẩn & Liệu pháp Kiểm soát Tích cực (Insulin / Đa trị liệu)",
        "details": (
            "Khám chuyên khoa Nội tiết sớm để đánh giá toàn diện (HbA1c, ceton máu/niệu, chức năng gan thận, đáy mắt). "
            "Khi đường huyết tăng rất cao (≥ 250–300 mg/dL, HbA1c ≥ 10%) hoặc có triệu chứng dị hóa (sút cân, tiểu nhiều, khát nhiều), "
            "bác sĩ sẽ cân nhắc khởi trị Insulin nền (Glargine/Lantus, Degludec/Tresiba) phối hợp thuốc viên theo phác đồ ADA/Vinmec."
        ),
        "reference_drugs_vn": [
            "Insulin nền tác dụng kéo dài (Insulin Glargine: Lantus Solostar / Toujeo; Insulin Degludec: Tresiba)",
            "Phối hợp Metformin + SGLT2i/DPP-4i khi tình trạng đường huyết cấp đã ổn định",
        ],
        "monitoring_plan": "Theo dõi đường huyết mao mạch tại nhà (SMBG) hoặc CGM nhiều lần/ngày; nhận biết và xử trí ngay dấu hiệu hạ đường huyết (quy tắc 15–15).",
        "invasiveness": "Rất cao",
        "cost": "Rất cao",
    },
]


class DiabetesQLearningEngine:
    """Mô hình Q-Learning tối ưu hóa chính sách can thiệp bệnh tiểu đường theo MDP."""

    def __init__(self, alpha: float = 0.1, gamma: float = 0.90, epsilon: float = 0.15):
        self.alpha = alpha      # Tốc độ học (Learning rate)
        self.gamma = gamma      # Hệ số chiết khấu (Discount factor)
        self.epsilon = epsilon  # Tỷ lệ khám phá (Exploration rate)

        # Không gian trạng thái: 4 (Glucose) x 3 (BMI) x 2 (Comorbidities) = 24 trạng thái
        self.n_glucose_bins = 4
        self.n_bmi_bins = 3
        self.n_comorb_bins = 2
        self.n_states = self.n_glucose_bins * self.n_bmi_bins * self.n_comorb_bins
        self.n_actions = 4

        self.q_table = np.zeros((self.n_states, self.n_actions), dtype=np.float64)
        self._pretrain_clinical_policy()

    def encode_state(self, glucose: float, bmi: float, has_comorbidity: int) -> int:
        """Chuyển đổi các chỉ số liên tục thành chỉ số trạng thái rời rạc."""
        # Glucose bin
        if glucose < 100.0:
            g_bin = 0  # Bình thường
        elif glucose < 126.0:
            g_bin = 1  # Tiền tiểu đường
        elif glucose < 180.0:
            g_bin = 2  # Tiểu đường vừa
        else:
            g_bin = 3  # Nguy cơ tăng đường huyết cấp

        # BMI bin (theo ngưỡng châu Á WHO/IDI & BYT: ≥ 23 thừa cân, ≥ 25 béo phì)
        if bmi < 23.0:
            b_bin = 0  # Bình thường
        elif bmi < 25.0:
            b_bin = 1  # Thừa cân châu Á
        else:
            b_bin = 2  # Béo phì châu Á

        c_bin = 1 if has_comorbidity else 0

        state_idx = (g_bin * self.n_bmi_bins + b_bin) * self.n_comorb_bins + c_bin
        return int(np.clip(state_idx, 0, self.n_states - 1))

    def _pretrain_clinical_policy(self, episodes: int = 1500):
        """Mô phỏng tiền huấn luyện Q-Table theo hướng dẫn lâm sàng ADA 2024, BYT và Vinmec."""
        rng = np.random.default_rng(42)

        for _ in range(episodes):
            g_bin = rng.integers(0, 4)
            b_bin = rng.integers(0, 3)
            c_bin = rng.integers(0, 2)
            s = (g_bin * self.n_bmi_bins + b_bin) * self.n_comorb_bins + c_bin

            # Chọn hành động e-greedy
            if rng.random() < self.epsilon:
                a = rng.integers(0, self.n_actions)
            else:
                a = int(np.argmax(self.q_table[s]))

            # Hàm phần thưởng y khoa (Reward Function):
            # Thưởng cao nếu hành động phù hợp với mức độ nghiêm trọng, BMI và bệnh đồng mắc
            reward = 0.0
            if g_bin == 0:  # Bình thường
                reward = 4.0 if a == 0 else (-2.0 * a)
            elif g_bin == 1:  # Tiền ĐTĐ
                reward = 3.5 if a == 0 else (1.8 if (a == 1 and b_bin == 2) else (1.2 if a == 1 else -2.5))
            elif g_bin == 2:  # ĐTĐ vừa
                if c_bin == 0 and b_bin < 2:
                    reward = 4.0 if a == 1 else (2.2 if a == 2 else -2.0)
                else:
                    # Có bệnh đồng mắc tim-mạch/huyết áp hoặc béo phì (BMI >= 25): ưu tiên phối hợp SGLT2i/GLP-1 RA
                    reward = 4.5 if a == 2 else (2.5 if a == 1 else -3.0)
            elif g_bin == 3:  # ĐTĐ nặng / Cấp tính
                reward = 5.0 if a == 3 else (2.0 if a == 2 else -4.0)

            # Mô phỏng trạng thái kế tiếp s' sau khi can thiệp
            next_g = max(0, g_bin - (1 if a >= 1 else 0))
            s_next = (next_g * self.n_bmi_bins + b_bin) * self.n_comorb_bins + c_bin

            # Cập nhật Bellman Q-Learning: Q(s,a) = Q(s,a) + alpha * [r + gamma * max Q(s',a') - Q(s,a)]
            td_target = reward + self.gamma * np.max(self.q_table[s_next])
            self.q_table[s, a] += self.alpha * (td_target - self.q_table[s, a])

    def recommend_treatment(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Đưa ra phác đồ can thiệp tối ưu dựa trên trạng thái hiện tại của bệnh nhân."""
        glucose = float(patient_data.get("Glucose", 110.0))
        bmi = float(patient_data.get("BMI", 22.0))
        bp = int(patient_data.get("HighBP", 0))
        has_comorb = 1 if (bp == 1 or patient_data.get("HeartDiseaseorAttack", 0) == 1) else 0

        state_idx = self.encode_state(glucose, bmi, has_comorb)
        q_values = self.q_table[state_idx]

        best_action_idx = int(np.argmax(q_values))
        best_action = ACTION_DESCRIPTIONS[best_action_idx]

        # Tính toán phân phối xác suất hành động qua hàm Softmax nhiệt độ
        exp_q = np.exp((q_values - np.max(q_values)) / 0.8)
        action_probs = exp_q / np.sum(exp_q)

        return {
            "state_id": state_idx,
            "optimal_action_id": best_action_idx,
            "recommended_action": best_action["name"],
            "intervention_details": best_action["details"],
            "reference_drugs_vn": best_action["reference_drugs_vn"],
            "monitoring_plan": best_action["monitoring_plan"],
            "invasiveness": best_action["invasiveness"],
            "estimated_q_value": round(float(q_values[best_action_idx]), 3),
            "action_q_distribution": [
                {
                    "action_id": i,
                    "action_name": ACTION_DESCRIPTIONS[i]["name"],
                    "q_value": round(float(q_values[i]), 3),
                    "confidence": round(float(action_probs[i] * 100), 1),
                }
                for i in range(self.n_actions)
            ],
            "rl_algorithm": "Tabular Q-Learning (ADA 2024 + QĐ 5481/BYT + Vinmec + Dược điển Long Châu)",
        }
