"""Cây quyết định độ sâu 2 trên Diabetes (đặc trưng gốc, chưa chuẩn hóa) để minh hoạ cách tách nút trong Hình cây.
Ghi outputs/metrics/ch2_tree.json: ngưỡng, Gini, tỷ lệ mẫu và xác suất lá."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sklearn.tree import DecisionTreeClassifier, export_text  # noqa: E402

from tl3 import ch2_data as cd  # noqa: E402
from tl3.common import save_json  # noqa: E402


def describe_tree(tree, names):
    """Duyệt cấu trúc cây scikit-learn: với mỗi nút trả đặc trưng, ngưỡng, Gini, tỷ lệ mẫu, P(lớp 1)."""
    t = tree.tree_
    n0 = t.weighted_n_node_samples[0]
    nodes = {}
    for i in range(t.node_count):
        val = t.value[i][0]
        nodes[f"n{i}"] = {"feature": names[t.feature[i]] if t.feature[i] >= 0 else "lá",
                          "thr": round(float(t.threshold[i]), 2), "gini": round(float(t.impurity[i]), 3),
                          "share": round(float(t.weighted_n_node_samples[i] / n0), 3),
                          "p1": round(float(val[1] / val.sum()), 2)}
    return nodes


def main():
    df = cd.clean_diabetes(cd.load_diabetes_raw())
    cols = ["age", "bmi", "HbA1c_level", "blood_glucose_level", "hypertension", "heart_disease"]
    tree = DecisionTreeClassifier(max_depth=2, random_state=42).fit(df[cols], df.diabetes)
    res = {"nodes": describe_tree(tree, cols), "text": export_text(tree, feature_names=cols)}
    save_json("ch2_tree", res)
    print(res["text"])


if __name__ == "__main__":
    main()
