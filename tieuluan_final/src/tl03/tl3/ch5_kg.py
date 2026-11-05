"""Chương 5 — dựng đồ thị tri thức (KG) từ dữ liệu bảng: bộ ba (chủ thể, quan hệ, đối tượng) -> NetworkX DiGraph.

Hai tầng nút:
  * tầng dữ kiện: mỗi dòng bảng là một thực thể (Person_P000123, TradingDay_2020-03-16, MarketDay_2015-04-24),
    giá trị số giữ làm thuộc tính nút (literal) để tra cứu chính xác;
  * tầng khái niệm: nhóm tuổi, phân loại BMI/HbA1c theo chuẩn, tháng, năm, kiểu biến động… — các nút dùng chung
    nối nhiều thực thể, nhờ đó truy vấn theo nhóm ("nữ 60–79 tuổi có tăng huyết áp") đi được trên đồ thị.
"""
import re
from collections import Counter

import networkx as nx
import numpy as np

from .ch5_data import GENDER_VN, SMOKE_VN, vnum


# ---------------------------------------------------------------- bộ ba
def diabetes_triples(df):
    """Mỗi hồ sơ sinh 9 bộ ba dữ kiện (has_*) + 5 bộ ba khái niệm (in_age_group, bmi_class…)."""
    out = []
    for r in df.itertuples():
        s = f"Person_{r.pid}"
        out += [(s, "has_gender", f"Gender_{r.gender}"), (s, "has_age", float(r.age)),
                (s, "has_hypertension", int(r.hypertension)), (s, "has_heart_disease", int(r.heart_disease)),
                (s, "has_smoking_history", f"Smoking_{r.smoking_history}"), (s, "has_bmi", float(r.bmi)),
                (s, "has_hba1c_level", float(r.HbA1c_level)), (s, "has_blood_glucose_level", int(r.blood_glucose_level)),
                (s, "has_diabetes", int(r.diabetes)),
                (s, "in_age_group", f"AgeGroup_{r.age_group}"), (s, "bmi_class", f"BMI_{r.bmi_class}"),
                (s, "hba1c_class", f"HbA1c_{r.hba1c_class}"), (s, "glucose_class", f"Glucose_{r.glucose_class}"),
                (s, "diabetes_status", "Diabetes_yes" if r.diabetes else "Diabetes_no")]
        if r.hypertension:
            out.append((s, "has_condition", "Condition_Hypertension"))
        if r.heart_disease:
            out.append((s, "has_condition", "Condition_HeartDisease"))
    # tầng ontology: khái niệm -> yếu tố nguy cơ (tri thức nền theo ADA/WHO, không suy ra từ dữ liệu)
    out += [("HbA1c_ngưỡng chẩn đoán", "indicates", "Diabetes_yes"),
            ("HbA1c_tiền tiểu đường", "risk_factor_of", "Diabetes_yes"),
            ("Glucose_từ 200", "indicates", "Diabetes_yes"), ("BMI_béo phì", "risk_factor_of", "Diabetes_yes"),
            ("Condition_Hypertension", "risk_factor_of", "Diabetes_yes"),
            ("Condition_HeartDisease", "comorbid_with", "Diabetes_yes"),
            ("AgeGroup_60–79", "risk_factor_of", "Diabetes_yes"), ("AgeGroup_từ 80", "risk_factor_of", "Diabetes_yes")]
    return out


def market_triples(df, entity, idcol):
    """Ngày giao dịch -> quan hệ OHLCV (literal), thuộc tháng/năm, kiểu biến động, ngày kế tiếp."""
    out = []
    ids = [f"{entity}_{d:%Y-%m-%d}" for d in df.Date]
    for i, r in enumerate(df.itertuples()):
        s = ids[i]
        out += [(s, "has_date", f"{r.Date:%Y-%m-%d}"), (s, "has_open", float(r.Open)), (s, "has_high", float(r.High)),
                (s, "has_low", float(r.Low)), (s, "has_close", float(r.Close)), (s, "has_volume", int(r.Volume)),
                (s, "in_month", f"Month_{r.month}"), (s, "has_movement", f"Move_{r.move}")]
        if not np.isnan(r.ret):
            out.append((s, "has_return_pct", float(r.ret)))
        if i + 1 < len(ids):
            out.append((s, "next_day", ids[i + 1]))
    for m in sorted(df.month.unique()):
        out.append((f"Month_{m}", "in_year", f"Year_{m[:4]}"))
    return out


def gold_triples(df):
    """Như market_triples cho quỹ GLD, thêm các chỉ báo thị trường cùng ngày (has_sp500_close, has_brent…)."""
    out = market_triples(df, "MarketDay", "gid")
    extra = {"SP_close": "has_sp500_close", "DJ_close": "has_dowjones_close", "EU_Price": "has_eurusd",
             "OF_Price": "has_brent", "OS_Price": "has_wti", "USB_Price": "has_us_bond_yield",
             "USDI_Price": "has_usd_index", "GDX_Close": "has_gdx_close", "USO_Close": "has_uso_close"}
    for r in df.itertuples():
        s = f"MarketDay_{r.Date:%Y-%m-%d}"
        out += [(s, p, float(getattr(r, c))) for c, p in extra.items()]
    return out


def build_graph(triples):
    """Đối tượng là chuỗi -> cạnh tới nút thực thể/khái niệm; đối tượng là số -> thuộc tính của nút chủ thể."""
    g = nx.DiGraph()
    for s, p, o in triples:
        if s not in g:
            g.add_node(s, kind=s.split("_", 1)[0])
        if isinstance(o, str) and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", o):
            if o not in g:
                g.add_node(o, kind=o.split("_", 1)[0])
            g.add_edge(s, o, rel=p)
        else:
            g.nodes[s][p] = o
    return g


def graph_stats(g):
    """Số nút, số cạnh, phân bố loại nút và quan hệ, bậc vào lớn nhất (nút khái niệm trung tâm)."""
    kinds = Counter(d["kind"] for _, d in g.nodes(data=True))
    rels = Counter(d["rel"] for *_, d in g.edges(data=True))
    indeg = sorted(g.in_degree(), key=lambda x: -x[1])[:8]
    return {"nodes": g.number_of_nodes(), "edges": g.number_of_edges(), "kinds": dict(kinds.most_common()),
            "relations": dict(rels.most_common()), "top_in_degree": [[n, d] for n, d in indeg],
            "n_literals": int(sum(len(d) - 1 for _, d in g.nodes(data=True)))}


def to_rdf(triples, base="http://tl3.vn/kg/"):
    """Bộ ba -> đồ thị RDF (rdflib): thực thể/khái niệm thành URI, giá trị số thành literal có kiểu XSD."""
    from rdflib import Graph, Literal, Namespace, URIRef
    from rdflib.namespace import XSD
    ns = Namespace(base)
    rdf = Graph()
    rdf.bind("kg", ns)
    uri = lambda x: URIRef(base + re.sub(r"[^\w\-]", "_", x))
    for s, p, o in triples:
        if isinstance(o, str) and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", o):
            obj = uri(o)
        elif isinstance(o, str):
            obj = Literal(o, datatype=XSD.date)
        else:
            obj = Literal(o, datatype=XSD.integer if isinstance(o, int) else XSD.double)
        rdf.add((uri(s), ns[p], obj))
    return rdf


# ---------------------------------------------------------------- liên kết thực thể từ câu hỏi
DATE_PATTERNS = [
    (r"(\d{1,2})/(\d{1,2})/(\d{4})", lambda m: (m[3], m[2], m[1])),
    (r"ngày (\d{1,2}) tháng (\d{1,2}) năm (\d{4})", lambda m: (m[3], m[2], m[1])),
    (r"(\d{4})-(\d{2})-(\d{2})", lambda m: (m[1], m[2], m[3])),
]


def link_dates(q):
    """Trích mọi ngày trong câu hỏi -> danh sách 'YYYY-MM-DD' (không phân biệt hoa thường: 'Ngày 5 tháng 3…')."""
    q = q.lower()
    out = []
    for pat, f in DATE_PATTERNS:
        for m in re.finditer(pat, q):
            y, mo, d = f(m)
            out.append(f"{int(y):04d}-{int(mo):02d}-{int(d):02d}")
    return sorted(set(out))


def link_periods(q):
    """Tháng ('tháng 3/2020', 'Tháng 03 năm 2020') và năm ('năm 2019') -> ['2020-03', '2019']."""
    q = q.lower()
    months = [f"{int(y):04d}-{int(m):02d}" for m, y in re.findall(r"tháng (\d{1,2})(?:/| năm )(\d{4})", q)]
    years = [y for y in re.findall(r"năm (\d{4})", q) if not any(m.startswith(y) for m in months)]
    return months, years


AGE_RE = {"dưới 20": r"dưới 20", "20–39": r"20[–-]39", "40–59": r"40[–-]59", "60–79": r"60[–-]79",
          "từ 80": r"từ 80|trên 80"}


def link_diabetes(q):
    """Nhận diện giới, nhóm tuổi, tăng huyết áp, hút thuốc, phân loại chỉ số và mã hồ sơ trong câu hỏi."""
    ql = q.lower()
    ent = {}
    if re.search(r"\bnữ\b", ql):
        ent["gender"] = "Female"
    elif re.search(r"\bnam\b", ql):
        ent["gender"] = "Male"
    for k, pat in AGE_RE.items():
        if re.search(pat, ql):
            ent["age_group"] = k
    if re.search(r"không (?:bị )?(?:tăng|cao) huyết áp", ql):
        ent["hypertension"] = 0
    elif re.search(r"(?:tăng|cao) huyết áp", ql):
        ent["hypertension"] = 1
    for k, v in SMOKE_VN.items():
        if v in ql:
            ent["smoking_history"] = k
    for c, vals in (("hba1c_class", ("tiền tiểu đường", "ngưỡng chẩn đoán")),
                    ("bmi_class", ("thiếu cân", "thừa cân", "béo phì"))):
        for v in vals:
            if v in ql:
                ent[c] = v
    pid = re.search(r"\bP\d{6}\b", q)
    if pid:
        ent["pid"] = pid.group(0)
    return ent


# ---------------------------------------------------------------- truy xuất trên đồ thị
def verbalize(g, node, max_edges=12):
    """Nút -> danh sách câu bộ ba dễ đọc (thuộc tính literal + cạnh ra) làm bằng chứng cho LLM."""
    d = g.nodes[node]
    lines = [f"({node}, {k}, {vnum(v) if isinstance(v, float) else v})" for k, v in d.items() if k != "kind"]
    for _, o, e in list(g.out_edges(node, data=True))[:max_edges]:
        lines.append(f"({node}, {e['rel']}, {o})")
    return lines


def persons_matching(g, ent):
    """Giao các tập láng giềng vào của nút khái niệm (Gender_, AgeGroup_, Condition_…) -> tập Person thoả điều kiện."""
    sets = []
    if "gender" in ent:
        sets.append(set(g.predecessors(f"Gender_{ent['gender']}")))
    if "age_group" in ent:
        sets.append(set(g.predecessors(f"AgeGroup_{ent['age_group']}")))
    if "smoking_history" in ent:
        smoke = ent["smoking_history"]
        sets.append(set(g.predecessors(f"Smoking_{smoke}")))
    for c, pre in (("hba1c_class", "HbA1c_"), ("bmi_class", "BMI_")):
        if c in ent:
            sets.append(set(g.predecessors(pre + ent[c])))
    if not sets and "hypertension" not in ent:
        return set()
    base = set.intersection(*sets) if sets else {n for n in g if n.startswith("Person_")}
    if "hypertension" in ent:
        hyp = set(g.predecessors("Condition_Hypertension"))
        base = base & hyp if ent["hypertension"] else base - hyp
    return base
