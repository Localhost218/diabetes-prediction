"""Chương 5 — một "miền tri thức" gom dữ liệu bảng, kho tài liệu, đồ thị tri thức và các công cụ tra cứu chính xác.

Domain.link(q)  : liên kết thực thể trong câu hỏi -> id tài liệu (bộ truy xuất đồ thị dùng).
Domain.facts(q) : công cụ của tác tử — trả lời chính xác bằng duyệt đồ thị hoặc phép tính pandas, kèm bộ ba bằng chứng;
                  trả None khi câu hỏi không nêu thực thể cụ thể (để tác tử chuyển sang RAG + LLM).
"""
import re

import numpy as np
import pandas as pd

from .ch5_data import GENDER_VN, GOLD_COLS, SMOKE_VN, build_corpus, vnum
from .ch5_kg import (build_graph, diabetes_triples, gold_triples, graph_stats, link_dates, link_diabetes, link_periods,
                     market_triples, persons_matching, verbalize)

NUM = r"(\d+(?:[.,]\d+)?)"


def fnum(s):
    return float(s.replace(".", "").replace(",", ".")) if re.search(r",\d", s) else float(s.replace(",", "."))


# trường giá được hỏi -> (cột DataFrame, tên đọc, đơn vị)
MARKET_FIELDS = [
    (r"mở cửa", "Open", "giá mở cửa"), (r"cao nhất", "High", "giá cao nhất"), (r"thấp nhất", "Low", "giá thấp nhất"),
    (r"khối lượng", "Volume", "khối lượng giao dịch"), (r"thay đổi|biến động|tăng|giảm", "ret", "mức thay đổi (%)"),
    (r"đóng cửa|giá", "Close", "giá đóng cửa"),
]
GOLD_FIELDS = [(r"brent", "OF_Price"), (r"wti", "OS_Price"), (r"s&p|sp500|s\&p 500", "SP_close"),
               (r"dow jones", "DJ_close"), (r"eur/usd|euro", "EU_Price"), (r"lợi suất|trái phiếu", "USB_Price"),
               (r"chỉ số usd|đồng usd|usd index", "USDI_Price"), (r"gdx|mỏ vàng", "GDX_Close"),
               (r"\buso\b|quỹ dầu", "USO_Close")]
PERIOD_FIELDS = [(r"cao nhất", "high"), (r"thấp nhất", "low"), (r"trung bình", "mean"),
                 (r"biến động|độ lệch", "vol"), (r"thay đổi|tăng|giảm|hiệu suất", "chg"), (r"đầu", "first"),
                 (r"cuối", "last")]


class Domain:
    def __init__(self, name, build_graph_now=True):
        self.name = name
        self.df, self.docs = build_corpus(name)
        self.ids = [d["id"] for d in self.docs]
        self.texts = [d["text"] for d in self.docs]
        self.meta = {d["id"]: d for d in self.docs}
        self.g = None
        if build_graph_now:
            self.build_graph()
        if name != "diabetes":
            self.by_date = {f"{d:%Y-%m-%d}": i for i, d in enumerate(self.df.Date)}

    # ------------------------------------------------------------ đồ thị
    def triples(self):
        if self.name == "diabetes":
            return diabetes_triples(self.df)
        if self.name == "stock":
            return market_triples(self.df, "TradingDay", "did")
        return gold_triples(self.df)

    def build_graph(self):
        self.g = build_graph(self.triples())
        return self.g

    def stats(self):
        return graph_stats(self.g)

    def node_of(self, doc_id):
        """id tài liệu bản ghi -> nút thực thể tương ứng trong KG."""
        if self.name == "diabetes":
            return f"Person_{doc_id}"
        return ("TradingDay_" if self.name == "stock" else "MarketDay_") + doc_id[1:]

    # ------------------------------------------------------------ liên kết thực thể -> tài liệu
    def link(self, q):
        if self.name == "diabetes":
            return self._link_diabetes(q)
        pre = "D" if self.name == "stock" else "G"
        months, years = link_periods(q)
        return [pre + d for d in link_dates(q)] + [f"M-{m}" for m in months] + [f"M-{y}" for y in years]

    def numeric_filters(self, q):
        """Giá trị số nêu trong câu hỏi hồ sơ: tuổi, BMI, HbA1c, đường huyết."""
        ql = q.lower()
        f = {}
        for key, pat in (("age", NUM + r"\s*tuổi|tuổi\s*" + NUM), ("bmi", r"(?:bmi|khối cơ thể)\D{0,12}?" + NUM),
                         ("HbA1c_level", r"hba1c\D{0,12}?" + NUM),
                         ("blood_glucose_level", r"(?:đường huyết|glucose)\D{0,12}?" + NUM)):
            m = re.search(pat, ql)
            if m:
                f[key] = fnum(next(x for x in m.groups() if x))
        return f

    def _link_diabetes(self, q):
        ent = link_diabetes(q)
        if "pid" in ent:
            return [ent["pid"]]
        nums = self.numeric_filters(q)
        if len(nums) >= 2:
            if "age" in nums:
                from .ch5_data import age_group
                ent.setdefault("age_group", age_group(nums["age"]))
            cand = persons_matching(self.g, ent) if ent else [n for n in self.g if n.startswith("Person_")]
            attr = {"age": "has_age", "bmi": "has_bmi", "HbA1c_level": "has_hba1c_level",
                    "blood_glucose_level": "has_blood_glucose_level"}
            hit = [n for n in cand if all(abs(self.g.nodes[n].get(attr[k], -1e9) - v) < 1e-6 for k, v in nums.items())]
            return sorted(n.split("_", 1)[1] for n in hit)
        out = []
        if {"gender", "age_group", "hypertension"} <= ent.keys():
            out.append(f"S-{ent['gender']}|{ent['age_group']}|{ent['hypertension']}")
        for k in ("smoking_history", "hba1c_class", "bmi_class", "gender"):
            if k in ent:
                out.append(f"S-{k}={ent[k]}")
        return out

    # ------------------------------------------------------------ công cụ trả lời chính xác (tác tử)
    def facts(self, q):
        if self.name == "diabetes":
            return self._facts_diabetes(q)
        return self._facts_market(q)

    def _facts_market(self, q):
        ql = q.lower()
        dates = link_dates(q)
        months, years = link_periods(q)
        unit_name = "cổ phiếu Alphabet" if self.name == "stock" else "quỹ vàng GLD"
        if dates:
            col, label = "Close", "giá đóng cửa"
            if self.name == "gold":
                for pat, c in GOLD_FIELDS:
                    if re.search(pat, ql):
                        col, label = c, GOLD_COLS[c][0]
                        break
            if col == "Close":
                for pat, c, lab in MARKET_FIELDS:
                    if re.search(pat, ql):
                        col, label = c, lab
                        break
            lines, values, ev = [], {}, []
            for d in dates:
                if d not in self.by_date:
                    prev = [x for x in self.by_date if x < d]
                    note = f" Phiên gần nhất trước đó là {pd.Timestamp(prev[-1]):%d/%m/%Y}." if prev else ""
                    lines.append(f"Ngày {pd.Timestamp(d):%d/%m/%Y} không có phiên giao dịch trong dữ liệu.{note}")
                    continue
                r = self.df.iloc[self.by_date[d]]
                v = float(r[col])
                values[d] = v
                day = f"{pd.Timestamp(d):%d/%m/%Y}"
                if col in GOLD_COLS and col != "Close":
                    unit = GOLD_COLS[col][1]
                    lines.append(f"{label[0].upper() + label[1:]} ngày {day} là {vnum(v, 4 if col == 'EU_Price' else 2)}"
                                 f"{(' ' + unit) if unit else ''}.")
                else:
                    unit = {"ret": "%", "Volume": " cổ phiếu"}.get(col, " USD")
                    lines.append(f"{label[0].upper() + label[1:]} của {unit_name} ngày {day} là "
                                 f"{vnum(v, 0 if col == 'Volume' else 2)}{unit}.")
                ev += verbalize(self.g, self.node_of(("D" if self.name == "stock" else "G") + d), 9)
            if len(values) == 2 and col != "ret":
                (d1, v1), (d2, v2) = sorted(values.items())
                lines.append(f"Từ {pd.Timestamp(d1):%d/%m/%Y} đến {pd.Timestamp(d2):%d/%m/%Y} thay đổi "
                             f"{vnum(v2 - v1)} ({vnum(100 * (v2 / v1 - 1))}%).")
            return {"tool": "kg_lookup", "answer": " ".join(lines), "values": values, "evidence": ev}
        periods = months + years
        if periods:
            field = next((f for pat, f in PERIOD_FIELDS if re.search(pat, ql)), "chg")
            lines, values, ev = [], {}, []
            for p in periods:
                m = self.meta.get(f"M-{p}")
                if m is None:
                    lines.append(f"Dữ liệu không có giai đoạn {p}.")
                    continue
                values[p] = m[field]
                ev.append(m["text"])
                desc = {"high": f"giá cao nhất {vnum(m['high'])} USD (ngày {pd.Timestamp(m['high_date']):%d/%m/%Y})",
                        "low": f"giá thấp nhất {vnum(m['low'])} USD (ngày {pd.Timestamp(m['low_date']):%d/%m/%Y})",
                        "mean": f"giá đóng cửa trung bình {vnum(m['mean'])} USD",
                        "vol": f"độ biến động (độ lệch chuẩn lợi suất ngày) {vnum(m['vol'])}%",
                        "chg": f"thay đổi {vnum(m['chg'])}% (đóng cửa từ {vnum(m['first'])} USD "
                               f"{'lên' if m['chg'] >= 0 else 'xuống'} {vnum(m['last'])} USD)",
                        "first": f"đóng cửa phiên đầu {vnum(m['first'])} USD",
                        "last": f"đóng cửa phiên cuối {vnum(m['last'])} USD"}[field]
                name = f"tháng {p[5:]}/{p[:4]}" if len(p) == 7 else f"năm {p}"
                lines.append(f"Trong {name}, {unit_name} có {desc} ({m['n']} phiên).")
            return {"tool": "pandas_calc", "answer": " ".join(lines), "values": values, "evidence": ev}
        return None

    def _facts_diabetes(self, q):
        ent = link_diabetes(q)
        ids = self._link_diabetes(q)
        recs = [i for i in ids if not i.startswith("S-")]
        if recs:
            r = self.df.set_index("pid").loc[recs[:3]]
            lines = [f"Hồ sơ {pid}: {'MẮC' if row.diabetes else 'KHÔNG mắc'} tiểu đường (HbA1c {vnum(row.HbA1c_level, 1)}%, "
                     f"đường huyết {row.blood_glucose_level} mg/dL)." for pid, row in r.iterrows()]
            ev = sum((verbalize(self.g, f"Person_{p}", 8) for p in recs[:2]), [])
            return {"tool": "kg_lookup", "answer": " ".join(lines), "values": {p: int(r.loc[p].diabetes) for p in r.index},
                    "evidence": ev}
        if not ent or set(ent) == {"pid"}:
            return None
        persons = persons_matching(self.g, ent)
        if not persons:
            return {"tool": "kg_aggregate", "answer": "Không có hồ sơ nào thoả mọi điều kiện.", "values": {}, "evidence": []}
        nodes = self.g.nodes
        dia = np.array([nodes[p]["has_diabetes"] for p in persons])
        hba = np.array([nodes[p]["has_hba1c_level"] for p in persons])
        glu = np.array([nodes[p]["has_blood_glucose_level"] for p in persons])
        desc = []
        if "gender" in ent:
            desc.append(GENDER_VN[ent["gender"]])
        if "age_group" in ent:
            desc.append(f"nhóm tuổi {ent['age_group']}")
        if "hypertension" in ent:
            desc.append("có tăng huyết áp" if ent["hypertension"] else "không tăng huyết áp")
        if "smoking_history" in ent:
            desc.append(f"hút thuốc: {SMOKE_VN[ent['smoking_history']]}")
        for k in ("hba1c_class", "bmi_class"):
            if k in ent:
                desc.append(f"{'HbA1c' if k[0] == 'h' else 'BMI'} {ent[k]}")
        rate = float(dia.mean())
        ans = (f"Nhóm {', '.join(desc)} có {vnum(len(persons), 0)} hồ sơ; tỷ lệ mắc tiểu đường {vnum(100 * rate)}%, "
               f"HbA1c trung bình {vnum(hba.mean())}%, đường huyết trung bình {vnum(glu.mean(), 1)} mg/dL.")
        ev = [f"giao các tập láng giềng vào của: {', '.join(sorted(ent))}", f"|tập hồ sơ| = {len(persons)}"]
        return {"tool": "kg_aggregate", "answer": ans, "values": {"n": len(persons), "rate": 100 * rate}, "evidence": ev}
