"""
Chương 6: Học sâu trên Đồ thị (Deep Graph Learning / Graph Neural Networks - GNN)
Ứng dụng Graph Convolutional Network (GCN) và Graph Attention Network (GAT)
trong Bài toán Dự đoán Liên kết Bệnh lý Đồng mắc (Comorbidity Link Prediction).
Công thức lan truyền GCN chuẩn Kipf & Welling:
    H^(l+1) = ReLU( \tilde{D}^(-1/2) \tilde{A} \tilde{D}^(-1/2) H^(l) W^(l) )
"""

import numpy as np
from typing import Dict, List, Any


COMORBIDITY_NODES = [
    "Patient",
    "Type2_Diabetes",
    "Hypertension",
    "Diabetic_Nephropathy",
    "Diabetic_Retinopathy",
    "Dyslipidemia",
    "Coronary_Heart_Disease",
    "Metabolic_Syndrome"
]


class DeepGraphComorbidityGNN:
    """Mạng Nơ-ron Đồ thị (GCN) dự đoán mức độ rủi ro liên kết bệnh đồng mắc."""

    def __init__(self, node_dim: int = 8, hidden_dim: int = 16):
        self.nodes = COMORBIDITY_NODES
        self.num_nodes = len(self.nodes)
        self.node_to_idx = {name: i for i, name in enumerate(self.nodes)}

        # Khởi tạo ma trận kề gốc giữa các bệnh lý dựa trên dịch tễ y khoa
        self.adj = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float64)
        # Các liên kết y học giữa các bệnh lý
        edges = [
            ("Type2_Diabetes", "Hypertension", 0.85),
            ("Type2_Diabetes", "Diabetic_Nephropathy", 0.90),
            ("Type2_Diabetes", "Diabetic_Retinopathy", 0.92),
            ("Type2_Diabetes", "Dyslipidemia", 0.80),
            ("Type2_Diabetes", "Coronary_Heart_Disease", 0.78),
            ("Type2_Diabetes", "Metabolic_Syndrome", 0.95),
            ("Hypertension", "Coronary_Heart_Disease", 0.88),
            ("Hypertension", "Diabetic_Nephropathy", 0.82),
            ("Dyslipidemia", "Coronary_Heart_Disease", 0.85),
            ("Metabolic_Syndrome", "Hypertension", 0.90),
        ]
        for src, dst, w in edges:
            i, j = self.node_to_idx[src], self.node_to_idx[dst]
            self.adj[i, j] = w
            self.adj[j, i] = w

        # Khởi tạo trọng số các lớp GCN
        rng = np.random.default_rng(42)
        self.W0 = rng.normal(0, np.sqrt(2.0 / node_dim), (node_dim, hidden_dim))
        self.W1 = rng.normal(0, np.sqrt(2.0 / hidden_dim), (hidden_dim, 8))

    def _normalize_adjacency(self, A: np.ndarray) -> np.ndarray:
        """Chuẩn hóa ma trận kề thêm tự khuyên (Self-loops): D^(-1/2) * (A + I) * D^(-1/2)"""
        A_tilde = A + np.eye(A.shape[0])
        d = np.sum(A_tilde, axis=1)
        d_inv_sqrt = np.zeros_like(d)
        mask = d > 0
        d_inv_sqrt[mask] = 1.0 / np.sqrt(d[mask])
        D_inv_sqrt = np.diag(d_inv_sqrt)
        return D_inv_sqrt @ A_tilde @ D_inv_sqrt

    def forward(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Nạp đặc trưng bệnh nhân vào nút 'Patient', gắn cạnh tạm thời và chạy 2 lớp GCN
        """
        # Khởi tạo ma trận đặc trưng nút H (num_nodes x node_dim)
        H = np.zeros((self.num_nodes, 8), dtype=np.float64)

        # Trích xuất đặc trưng cho nút Patient (chỉ số BMI, Đường huyết, Huyết áp, Tuổi, ...)
        bmi = float(patient_data.get("BMI", 22.0))
        glucose = float(patient_data.get("Glucose", 110.0))
        hba1c = float(patient_data.get("HbA1c", 5.6))
        bp = float(patient_data.get("HighBP", 0))
        chol = float(patient_data.get("HighChol", 0))
        age = float(patient_data.get("Age", 5))

        H[0] = [
            (bmi - 18.0) / 15.0,
            (glucose - 70.0) / 150.0,
            (hba1c - 4.0) / 8.0,
            bp,
            chol,
            (age - 1) / 12.0,
            float(patient_data.get("FamilyHistory", 0)),
            float(patient_data.get("Smoker", 0))
        ]

        # Khởi tạo vector đặc trưng cho các nút bệnh lý khác
        for i in range(1, self.num_nodes):
            H[i, i % 8] = 1.0

        # Cập nhật cạnh động từ Patient đến các bệnh lý đã có biểu hiện
        A_dynamic = self.adj.copy()
        if glucose >= 126 or hba1c >= 6.5:
            A_dynamic[0, 1] = A_dynamic[1, 0] = 0.95
        elif glucose >= 100 or hba1c >= 5.7:
            A_dynamic[0, 1] = A_dynamic[1, 0] = 0.65
        else:
            A_dynamic[0, 1] = A_dynamic[1, 0] = 0.20

        if bp == 1:
            A_dynamic[0, 2] = A_dynamic[2, 0] = 0.90
        if chol == 1:
            A_dynamic[0, 5] = A_dynamic[5, 0] = 0.85
        if bmi >= 25.0:
            A_dynamic[0, 7] = A_dynamic[7, 0] = 0.88

        # Lan truyền GCN Lớp 1: H1 = ReLU(A_norm @ H @ W0)
        A_norm = self._normalize_adjacency(A_dynamic)
        Z1 = A_norm @ H @ self.W0
        H1 = np.maximum(0.0, Z1)

        # Lan truyền GCN Lớp 2: H2 = ReLU(A_norm @ H1 @ W1)
        Z2 = A_norm @ H1 @ self.W1
        H2 = np.maximum(0.0, Z2)

        # Tính toán xác suất liên kết Comorbidity Link Prediction giữa Patient (node 0) và các bệnh (node j)
        patient_emb = H2[0]
        comorbidity_risks = []

        for j in range(1, self.num_nodes):
            disease_name = self.nodes[j]
            disease_emb = H2[j]
            # Tính tương đồng Cosine giữa vector nhúng
            dot = np.dot(patient_emb, disease_emb)
            norm = (np.linalg.norm(patient_emb) * np.linalg.norm(disease_emb)) + 1e-8
            sim = float(dot / norm)
            # Sigmoid scaling
            link_prob = float(1.0 / (1.0 + np.exp(-3.5 * (sim - 0.25))))

            comorbidity_risks.append({
                "comorbidity": disease_name,
                "risk_score": round(link_prob, 3),
                "risk_level": "Rất cao" if link_prob >= 0.70 else ("Cao" if link_prob >= 0.45 else "Thấp"),
                "graph_degree_centrality": round(float(np.sum(A_dynamic[j]) / (self.num_nodes - 1)), 2)
            })

        comorbidity_risks.sort(key=lambda x: x["risk_score"], reverse=True)

        return {
            "model_type": "2-Layer Spectral Graph Convolutional Network (GCN)",
            "graph_nodes_count": self.num_nodes,
            "patient_embedding_norm": round(float(np.linalg.norm(patient_emb)), 3),
            "predicted_comorbidities": comorbidity_risks
        }
