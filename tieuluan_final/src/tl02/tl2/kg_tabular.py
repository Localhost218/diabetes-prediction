"""Xây KB và KG từ dữ liệu bảng: mỗi dòng thành (1) một tài liệu văn bản và (2) một tập bộ ba.

Đây là quy trình "bản ghi -> tài liệu + bộ ba" áp dụng cho hồ sơ bệnh nhân và phiên giao dịch. Giá trị số
được giữ ở thuộc tính của nút giá trị để truy vấn khoảng (HbA1c ≥ 6,5) — việc mà TF-IDF hay embedding
không làm chính xác được.
"""
import networkx as nx
import pandas as pd

DIAB_REL = {"gender": "has_gender", "age": "has_age", "hypertension": "has_hypertension",
            "heart_disease": "has_heart_disease", "smoking_history": "has_smoking_history", "bmi": "has_bmi",
            "HbA1c_level": "has_hba1c_level", "blood_glucose_level": "has_blood_glucose_level",
            "diabetes": "has_diabetes_label"}
DIAB_VI = {"gender": "giới tính", "age": "tuổi", "hypertension": "tăng huyết áp", "heart_disease": "bệnh tim",
           "smoking_history": "tiền sử hút thuốc", "bmi": "BMI", "HbA1c_level": "HbA1c",
           "blood_glucose_level": "glucose máu", "diabetes": "đái tháo đường"}


def diabetes_documents(df):
    """Hồ sơ -> câu mô tả có tên trường, phù hợp cho TF-IDF và embedding."""
    docs = []
    for i, r in df.iterrows():
        parts = [f"{DIAB_VI[c]} {_vi_value(c, r[c])}" for c in DIAB_REL]
        docs.append({"record_id": f"Person_{i}", "document": f"Hồ sơ Person_{i}: " + ", ".join(parts) + "."})
    return pd.DataFrame(docs)


def _vi_value(col, v):
    if col == "gender":
        return {"Female": "nữ", "Male": "nam"}.get(v, "khác")
    if col in ("hypertension", "heart_disease", "diabetes"):
        return "có" if int(v) == 1 else "không"
    if col == "smoking_history":
        return {"never": "chưa từng", "No Info": "không rõ", "current": "đang hút", "former": "đã bỏ",
                "ever": "từng hút", "not current": "hiện không hút"}.get(v, v)
    return f"{v:g}" if isinstance(v, float) else str(v)


def diabetes_triples(df):
    out = []
    for i, r in df.iterrows():
        s = f"Person_{i}"
        out.append((s, "rdf:type", "Patient"))
        for c, rel in DIAB_REL.items():
            out.append((s, rel, r[c]))
    return out


def stock_triples(df, symbol):
    out = []
    for i, (d, r) in enumerate(df.iterrows()):
        s = f"TradingDay_{symbol}_{i}"
        out += [(s, "rdf:type", "TradingDay"), (s, "of_stock", symbol), (s, "has_date", d.strftime("%Y-%m-%d"))]
        for c in ("Open", "High", "Low", "Close", "Volume"):
            out.append((s, f"has_{c.lower()}", float(r[c])))
    return out


def build_graph(triples):
    """Đồ thị có hướng: nút thực thể và nút giá trị (giá trị trùng nhau dùng chung một nút)."""
    G = nx.DiGraph()
    for s, p, o in triples:
        vnode = o if isinstance(o, str) and not p.startswith("has_") else f"{p}={o}"
        G.add_node(s, kind="entity")
        G.add_node(vnode, kind="value", value=o)
        G.add_edge(s, vnode, relation=p)
    return G


def retrieve_graph(triples, relation, predicate):
    """Truy xuất theo quan hệ và điều kiện trên giá trị: retrieve_graph(T, 'has_hba1c_level', lambda v: v >= 9)."""
    return [s for s, p, o in triples if p == relation and predicate(o)]


def record_facts(triples, subject):
    return {p: o for s, p, o in triples if s == subject}
