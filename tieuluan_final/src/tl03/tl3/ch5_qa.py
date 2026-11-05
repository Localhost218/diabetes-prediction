"""Chương 5 — sinh tự động bộ câu hỏi đánh giá có đáp án từ dữ liệu bảng, và chấm câu trả lời.

Mỗi câu hỏi: {qid, domain, type, style, question, gold: [id tài liệu đúng], answer, answer_type}.
  type  : record (tra một bản ghi) | group (thống kê nhóm) | month | year | indicator
  style : lex (dùng đúng cách viết của tài liệu) | para (diễn đạt khác: đồng nghĩa, định dạng ngày khác)
Hai kiểu diễn đạt cho phép đo riêng điểm mạnh của truy xuất từ khoá và truy xuất ngữ nghĩa.
"""
import random
import re

import pandas as pd

from .ch5_data import GENDER_VN, SMOKE_VN, vnum

FIELD_VN = {"Close": "giá đóng cửa", "Open": "giá mở cửa", "High": "giá cao nhất", "Low": "giá thấp nhất",
            "Volume": "khối lượng giao dịch"}
FIELD_PARA = {"Close": "giá chốt phiên", "Open": "giá mở cửa", "High": "mức giá cao nhất", "Low": "mức giá thấp nhất",
              "Volume": "khối lượng khớp lệnh"}
GOLD_IND = {"OF_Price": "giá dầu Brent", "OS_Price": "giá dầu WTI", "SP_close": "giá quỹ S&P 500 (SPY)",
            "EU_Price": "tỷ giá EUR/USD", "USDI_Price": "chỉ số USD", "GDX_Close": "giá quỹ mỏ vàng GDX"}
PERIOD_Q = {"chg": "thay đổi bao nhiêu phần trăm", "high": "có giá cao nhất là bao nhiêu",
            "low": "có giá thấp nhất là bao nhiêu", "mean": "có giá đóng cửa trung bình là bao nhiêu"}


def _date_fmt(d, style, rng):
    if style == "lex":
        return f"{d:%d/%m/%Y}"
    return rng.choice([f"ngày {d.day} tháng {d.month} năm {d.year}", f"{d:%Y-%m-%d}"])


def market_questions(D, rng, n_day=40, n_para=20, n_month=20, n_year=8, n_ind=12):
    df, name = D.df, ("cổ phiếu Google (Alphabet)" if D.name == "stock" else "quỹ vàng GLD")
    pre = "D" if D.name == "stock" else "G"
    out = []
    rows = df.iloc[1:].sample(n_day + n_para + n_ind, random_state=rng.randint(0, 10 ** 6))
    for k, (_, r) in enumerate(rows.iterrows()):
        d = r.Date
        if k < n_day + n_para:
            style = "lex" if k < n_day else "para"
            f = rng.choice(list(FIELD_VN))
            lab = FIELD_VN[f] if style == "lex" else FIELD_PARA[f]
            q = (f"{lab[0].upper() + lab[1:]} của {name} ngày {_date_fmt(d, style, rng)} là bao nhiêu?" if style == "lex"
                 else f"Cho biết {lab} của {name} vào {_date_fmt(d, style, rng)}.")
            out.append({"type": "record", "style": style, "question": q, "gold": [f"{pre}{d:%Y-%m-%d}"],
                        "answer": float(r[f]), "field": f, "answer_type": "number"})
        elif D.name == "gold":
            c = rng.choice(list(GOLD_IND))
            q = f"{GOLD_IND[c][0].upper() + GOLD_IND[c][1:]} ngày {d:%d/%m/%Y} là bao nhiêu?"
            out.append({"type": "indicator", "style": "lex", "question": q, "gold": [f"{pre}{d:%Y-%m-%d}"],
                        "answer": float(r[c]), "field": c, "answer_type": "number"})
    months = [d for d in D.docs if d["kind"] == "summary" and len(d["group"]) == 7 and d["n"] >= 15]
    years = [d for d in D.docs if d["kind"] == "summary" and len(d["group"]) == 4 and d["n"] >= 100]
    for m in rng.sample(months, n_month):
        f = rng.choice(list(PERIOD_Q))
        q = f"Trong tháng {int(m['group'][5:])}/{m['group'][:4]}, {name} {PERIOD_Q[f]}?"
        out.append({"type": "month", "style": "lex", "question": q, "gold": [m["id"]], "answer": m[f], "field": f,
                    "answer_type": "number"})
    for y in rng.sample(years, min(n_year, len(years))):
        f = rng.choice(list(PERIOD_Q))
        q = f"Trong năm {y['group']}, {name} {PERIOD_Q[f]}?"
        out.append({"type": "year", "style": "lex", "question": q, "gold": [y["id"]], "answer": y[f], "field": f,
                    "answer_type": "number"})
    return out


def diabetes_questions(D, rng, n_rec=40, n_para=20, n_group=20, n_gpara=10, n_attr=10):
    df = D.df
    key = ["gender", "age", "bmi", "HbA1c_level", "blood_glucose_level"]
    uniq = df[~df.duplicated(key, keep=False) & (df.gender != "Other") & (df.age >= 2)]
    # lấy cân bằng hai nhãn để câu "có mắc không" không thiên về nhãn đa số
    pick = pd.concat([uniq[uniq.diabetes == 1].sample((n_rec + n_para) // 2, random_state=1),
                      uniq[uniq.diabetes == 0].sample((n_rec + n_para) - (n_rec + n_para) // 2, random_state=2)])
    pick = pick.sample(frac=1, random_state=3)
    out = []
    for k, r in enumerate(pick.itertuples()):
        if k < n_rec:
            q = (f"Bệnh nhân {GENDER_VN[r.gender]} {vnum(r.age, 0)} tuổi, BMI {vnum(r.bmi)}, HbA1c {vnum(r.HbA1c_level, 1)}%, "
                 f"đường huyết {r.blood_glucose_level} mg/dL có mắc tiểu đường không?")
            style = "lex"
        else:
            who = "phụ nữ" if r.gender == "Female" else "đàn ông"
            q = (f"Một {who} {vnum(r.age, 0)} tuổi có chỉ số khối cơ thể {vnum(r.bmi)}, glucose máu {r.blood_glucose_level}, "
                 f"HbA1c đo được {vnum(r.HbA1c_level, 1)} phần trăm — người này có bị đái tháo đường không?")
            style = "para"
        out.append({"type": "record", "style": style, "question": q, "gold": [r.pid], "answer": int(r.diabetes),
                    "answer_type": "label"})
    groups = [d for d in D.docs if d["id"].startswith("S-") and "|" in d["id"] and d["n"] >= 100]
    # chỉ có ≤ 20 nhóm giới × tuổi × huyết áp: câu diễn đạt lại được phép hỏi lại nhóm đã có câu khớp
    lex = rng.sample(groups, min(n_group, len(groups)))
    para = rng.sample(groups, min(n_gpara, len(groups)))
    for k, gdoc in enumerate(lex + para):
        gd, ag, ht = gdoc["group"].split("|")
        if k < len(lex):
            q = (f"Tỷ lệ mắc tiểu đường của bệnh nhân {GENDER_VN[gd]} nhóm tuổi {ag} "
                 f"{'có' if ht == '1' else 'không'} tăng huyết áp là bao nhiêu?")
            style = "lex"
        else:
            who = "phụ nữ" if gd == "Female" else "nam giới"
            q = (f"Trong số {who} {ag.replace('–', '-')} tuổi {'bị' if ht == '1' else 'không bị'} cao huyết áp, "
                 f"bao nhiêu phần trăm bị đái tháo đường?")
            style = "para"
        out.append({"type": "group", "style": style, "question": q, "gold": [gdoc["id"]], "answer": 100 * gdoc["rate"],
                    "answer_type": "number"})
    attrs = [d for d in D.docs if d["id"].startswith(("S-smoking_history=", "S-hba1c_class=", "S-bmi_class="))]
    for gdoc in rng.sample(attrs, n_attr):
        col, v = gdoc["id"][2:].split("=")
        label = {"smoking_history": f"tiền sử hút thuốc \"{SMOKE_VN.get(v, v)}\"", "hba1c_class": f"HbA1c {v}",
                 "bmi_class": f"BMI {v}"}[col]
        q = f"Tỷ lệ mắc tiểu đường trong nhóm {label} là bao nhiêu?"
        out.append({"type": "group", "style": "lex", "question": q, "gold": [gdoc["id"]], "answer": 100 * gdoc["rate"],
                    "answer_type": "number"})
    return out


OPEN_QUESTIONS = {
    "diabetes": ["Những yếu tố nào liên quan mạnh nhất tới tỷ lệ mắc tiểu đường trong dữ liệu?",
                 "Người hút thuốc có tỷ lệ tiểu đường cao hơn người chưa từng hút không?",
                 "Mức HbA1c ở nhóm tiền tiểu đường khác nhóm bình thường thế nào?"],
    "stock": ["Cổ phiếu Google biến động thế nào trong đại dịch năm 2020?",
              "Năm nào cổ phiếu Google tăng trưởng tốt nhất trong giai đoạn dữ liệu?",
              "Nhận xét diễn biến cổ phiếu Google trong năm 2022."],
    "gold": ["Quỹ vàng GLD diễn biến thế nào trong năm 2013?",
             "Giá vàng và giá dầu thay đổi ra sao trong năm 2015?",
             "Năm nào quỹ vàng GLD tăng mạnh nhất trong dữ liệu?"],
}


def make_questions(D, seed=42):
    rng = random.Random(seed)
    qs = diabetes_questions(D, rng) if D.name == "diabetes" else market_questions(D, rng)
    for i, q in enumerate(qs):
        q["qid"] = f"{D.name[:3]}{i:03d}"
        q["domain"] = D.name
    return qs


# ---------------------------------------------------------------- chấm câu trả lời
NUM_RE = re.compile(r"[-−]?\d{1,3}(?:\.\d{3})+(?:,\d+)?|[-−]?\d+,\d+|[-−]?\d+\.\d+|[-−]?\d+")


def parse_numbers(text):
    """Đọc mọi số trong văn bản, chấp nhận cả kiểu Việt (1.234,56) lẫn kiểu Anh (1234.56)."""
    out = []
    for tok in NUM_RE.findall(text):
        t = tok.replace("−", "-")
        if "," in t:
            t = t.replace(".", "").replace(",", ".")
        elif re.fullmatch(r"-?\d{1,3}(?:\.\d{3})+", t):
            t = t.replace(".", "")
        try:
            out.append(float(t))
        except ValueError:
            pass
    return out


def predicted_label(text):
    t = text.lower()
    if re.search(r"không (?:mắc|bị|có|phải)|chưa mắc|âm tính", t):
        return 0
    if re.search(r"(?:mắc|bị|có bệnh|dương tính)", t):
        return 1
    return None


def is_correct(q, text):
    """Đúng khi có một số trong câu trả lời khớp đáp án (sai lệch ≤ 0,5 % hoặc ≤ 0,011 tuyệt đối do làm tròn)."""
    if q["answer_type"] == "label":
        return predicted_label(text) == q["answer"]
    a = float(q["answer"])
    return any(abs(x - a) <= max(0.005 * abs(a), 0.011) for x in parse_numbers(text))
