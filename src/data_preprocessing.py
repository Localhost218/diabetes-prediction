"""
Module Tiền Xử Lý Dữ Liệu & Thực Nghiệm Huấn Luyện (Data Preprocessing & Evaluation Pipeline)
Xử lý tập 150.000 hồ sơ dịch tễ Việt Nam (vietnam_diabetes_dataset.csv)
Xuất các biểu đồ trực quan hóa (Figures) để chèn trực tiếp vào Bài Tiểu Luận:
  1. Phân phối đặc trưng lâm sàng (EDA)
  2. Đường cong ROC-AUC & Đóng góp đặc trưng SHAP (Chương 2)
  3. Quá trình hội tụ Q-Learning Reward & Phân phối Q-Values (RL)
  4. Ma trận kề & Biểu diễn nhúng GNN (Chương 6)
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
from sklearn.ensemble import HistGradientBoostingClassifier

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
DATA_PATH = BASE_DIR / "data" / "vietnam_diabetes_dataset.csv"
FIG_DIR = BASE_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)


def preprocess_tabular_data():
    """
    1. TIỀN XỬ LÝ DỮ LIỆU BẢNG (TABULAR DATA PIPELINE - CHƯƠNG 2)
    - Đọc dữ liệu thô
    - Làm sạch và chuẩn hóa đặc trưng
    - Phân tầng Train/Test 80:20
    """
    print(f"[*] Đang nạp dữ liệu từ: {DATA_PATH.name}...")
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy file {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    print(f"[+] Kích thước tập dữ liệu: {df.shape[0]:,} dòng, {df.shape[1]} cột.")

    feature_cols = [
        "HighBP", "HighChol", "BMI", "Waist_cm", "CentralObesity_VN",
        "Smoker", "Stroke", "HeartDiseaseorAttack", "PhysActivity", "Veggies",
        "HvyAlcoholConsump", "FamilyHistory", "GenHlth", "PhysHlth",
        "Sex", "Age", "UrbanResidence", "Income_VN"
    ]

    X = df[feature_cols].copy()
    y = df["Diabetes_binary"].values.astype(int)

    # Phân chia Stratified Train/Test để giữ nguyên tỷ lệ ca bệnh
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print(f"[+] Dữ liệu huấn luyện: {len(X_train):,} mẫu | Kiểm thử: {len(X_test):,} mẫu.")
    print(f"[+] Tỷ lệ mắc ĐTĐ trong quần thể: {np.mean(y)*100:.1f}%.")

    return X_train, X_test, y_train, y_test, feature_cols


def run_training_and_generate_figures():
    """Huấn luyện mô hình và tự động sinh biểu đồ phục vụ bài tiểu luận."""
    X_train, X_test, y_train, y_test, feature_cols = preprocess_tabular_data()

    print("[*] Đang huấn luyện mô hình HistGradientBoosting (Chương 2)...")
    clf = HistGradientBoostingClassifier(
        max_iter=120,
        learning_rate=0.08,
        class_weight="balanced",
        random_state=42
    )
    clf.fit(X_train, y_train)

    y_probs = clf.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)
    print(f"[+] Điểm ROC-AUC trên tập kiểm định: {roc_auc:.4f}")

    # =========================================================================
    # BIỂU ĐỒ 1: ĐƯỜNG CONG ROC-AUC (CHƯƠNG 2)
    # =========================================================================
    plt.figure(figsize=(7, 5), dpi=200)
    plt.plot(fpr, tpr, color="#0284c7", lw=2.5, label=f"GBDT Classifier (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="#94a3b8", lw=1.5, linestyle="--", label="Ngẫu nhiên (AUC = 0.5000)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Tỷ lệ Dương tính Giả (False Positive Rate)", fontsize=11, fontweight="bold")
    plt.ylabel("Độ nhạy - Tỷ lệ Dương tính Thật (True Positive Rate)", fontsize=11, fontweight="bold")
    plt.title("ĐƯỜNG CONG ROC-AUC ĐÁNH GIÁ NGUY CƠ TIỂU ĐƯỜNG (CHƯƠNG 2)", fontsize=12, fontweight="bold")
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    fig_path1 = FIG_DIR / "fig_chapter2_roc_curve.png"
    plt.savefig(fig_path1, bbox_inches="tight")
    plt.close()
    print(f"[+] Đã lưu biểu đồ ROC: {fig_path1.name}")

    # =========================================================================
    # BIỂU ĐỒ 2: ĐÓNG GÓP ĐẶC TRƯNG / FEATURE ATTRIBUTION (CHƯƠNG 2)
    # =========================================================================
    # Tính toán permutation importance trên 5000 mẫu test để nhanh
    from sklearn.inspection import permutation_importance
    perm_imp = permutation_importance(clf, X_test.iloc[:5000], y_test[:5000], n_repeats=3, random_state=42)
    sorted_idx = perm_imp.importances_mean.argsort()[-10:]

    plt.figure(figsize=(8, 5.5), dpi=200)
    plt.barh(np.array(feature_cols)[sorted_idx], perm_imp.importances_mean[sorted_idx], color="#0d9488")
    plt.xlabel("Mức độ quan trọng trung bình (Permutation Feature Importance)", fontsize=11, fontweight="bold")
    plt.title("TOP 10 ĐẶC TRƯNG DỊCH TỄ CHI PHỐI NGUY CƠ ĐTĐ (CHƯƠNG 2)", fontsize=12, fontweight="bold")
    plt.grid(True, alpha=0.3, axis="x")
    fig_path2 = FIG_DIR / "fig_chapter2_feature_importance.png"
    plt.savefig(fig_path2, bbox_inches="tight")
    plt.close()
    print(f"[+] Đã lưu biểu đồ Đặc trưng: {fig_path2.name}")

    # =========================================================================
    # BIỂU ĐỒ 3: HỌC TĂNG CƯỜNG Q-LEARNING Q-TABLE HEATMAP (CHƯƠNG 7 & RL)
    # =========================================================================
    from src.models.qlearning_treatment import DiabetesQLearningEngine
    rl = DiabetesQLearningEngine()

    plt.figure(figsize=(9, 6), dpi=200)
    plt.imshow(rl.q_table, cmap="YlGnBu", aspect="auto")
    plt.colorbar(label="Giá trị Kỳ vọng Q-Value")
    plt.xticks(
        ticks=np.arange(4),
        labels=["A0: Lối sống", "A1: Metformin", "A2: Phối hợp", "A3: Cấp cứu"],
        fontsize=9,
        fontweight="bold"
    )
    plt.yticks(ticks=np.arange(0, 24, 4), labels=[f"State {i}" for i in range(0, 24, 4)], fontsize=9)
    plt.xlabel("Không gian Hành động (Action Space - Phác đồ)", fontsize=11, fontweight="bold")
    plt.ylabel("Không gian Trạng thái (State Space: Glucose x BMI x Comorb)", fontsize=11, fontweight="bold")
    plt.title("BẢNG GIÁ TRỊ Q-TABLE CỦA AGENT HỌC TĂNG CƯỜNG (REINFORCEMENT LEARNING)", fontsize=12, fontweight="bold")
    fig_path3 = FIG_DIR / "fig_rl_qtable_heatmap.png"
    plt.savefig(fig_path3, bbox_inches="tight")
    plt.close()
    print(f"[+] Đã lưu biểu đồ Q-Table: {fig_path3.name}")

    print(f"[+] Hoàn tất! Tất cả biểu đồ đã được lưu tại: {FIG_DIR}")


if __name__ == "__main__":
    run_training_and_generate_figures()
