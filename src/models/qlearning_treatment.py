"""
Học Tăng Cường (Reinforcement Learning) - Thuật toán Q-Learning (Off-Policy TD Control)
Ứng dụng trong Tối ưu hóa Phác đồ Can thiệp và Điều trị Đái tháo đường Tuần tự
Kế thừa và mở rộng từ bài tập Agent 02 (Tabular Q-Learning).
"""

import numpy as np
from typing import Dict, List, Any, Tuple


ACTION_DESCRIPTIONS = [
    {
        "action_id": 0,
        "name": "Can thiệp lối sống (Lifestyle Modification)",
        "details": "Chế độ ăn giảm chỉ số GI, hạn chế carbohydrate tinh chế, tập thể dục 150 phút/tuần.",
        "invasiveness": "Thấp",
        "cost": "Thấp"
    },
    {
        "action_id": 1,
        "name": "Đơn trị liệu Metformin (First-Line Monotherapy)",
        "details": "Trao đổi với bác sĩ về việc khởi trị metformin (thuốc đầu tay theo ADA/Bộ Y tế) kèm duy trì thay đổi lối sống.",
        "invasiveness": "Trung bình",
        "cost": "Vừa phải"
    },
    {
        "action_id": 2,
        "name": "Liệu pháp Phối hợp Thuốc (Dual Therapy - Metformin + SGLT2i)",
        "details": "Trao đổi với bác sĩ về phác đồ phối hợp (metformin cùng nhóm thuốc có lợi ích tim – thận như ức chế SGLT2/GLP-1 RA) do có bệnh đồng mắc.",
        "invasiveness": "Cao",
        "cost": "Cao"
    },
    {
        "action_id": 3,
        "name": "Hội chẩn Chuyên khoa Khẩn & Liệu pháp Insulin",
        "details": "Khám chuyên khoa Nội tiết sớm để được đánh giá toàn diện; bác sĩ có thể cân nhắc insulin hoặc xét nghiệm cấp.",
        "invasiveness": "Rất cao",
        "cost": "Rất cao"
    }
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

        # BMI bin
        if bmi < 23.0:
            b_bin = 0  # Bình thường
        elif bmi < 25.0:
            b_bin = 1  # Thừa cân
        else:
            b_bin = 2  # Béo phì

        c_bin = 1 if has_comorbidity else 0

        state_idx = (g_bin * self.n_bmi_bins + b_bin) * self.n_comorb_bins + c_bin
        return int(np.clip(state_idx, 0, self.n_states - 1))

    def _pretrain_clinical_policy(self, episodes: int = 1500):
        """Mô phỏng tiền huấn luyện Q-Table theo hướng dẫn lâm sàng ADA và BYT."""
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
            # Thưởng cao nếu hành động phù hợp với mức độ nghiêm trọng; phạt can thiệp quá mức hoặc thiếu sót
            reward = 0.0
            if g_bin == 0:  # Bình thường
                reward = 4.0 if a == 0 else (-2.0 * a)
            elif g_bin == 1:  # Tiền ĐTĐ
                reward = 3.5 if a == 0 else (1.5 if a == 1 else -2.5)
            elif g_bin == 2:  # ĐTĐ vừa
                if c_bin == 0:
                    reward = 4.0 if a == 1 else (2.0 if a == 2 else -2.0)
                else:
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
            "invasiveness": best_action["invasiveness"],
            "estimated_q_value": round(float(q_values[best_action_idx]), 3),
            "action_q_distribution": [
                {
                    "action_id": i,
                    "action_name": ACTION_DESCRIPTIONS[i]["name"],
                    "q_value": round(float(q_values[i]), 3),
                    "confidence": round(float(action_probs[i] * 100), 1)
                }
                for i in range(self.n_actions)
            ],
            "rl_algorithm": "Tabular Q-Learning (Off-Policy TD Control với Discounted Bellman Expectation)"
        }
