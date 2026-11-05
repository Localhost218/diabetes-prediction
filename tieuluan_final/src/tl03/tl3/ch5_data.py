"""Chương 5 — nạp ba bộ dữ liệu bảng và chuyển mỗi dòng thành tài liệu văn bản cho kho tri thức (KB).

Ba miền: hồ sơ bệnh tiểu đường (100.000 dòng), giá cổ phiếu Alphabet 2015–2024, quỹ vàng GLD kèm các chỉ báo
thị trường (FINAL_USO.csv, 2011–2018). Mỗi miền sinh hai loại tài liệu:
  * tài liệu bản ghi — một dòng bảng, giữ định danh ổn định (P000123, D2020-03-16, G2015-04-24);
  * tài liệu tổng hợp — thống kê theo nhóm/tháng/năm, để trả lời câu hỏi mở về xu hướng.
"""
import numpy as np
import pandas as pd

from .common import data_dir

DATA = data_dir("ch5")
DIAB_CSV = DATA / "diabetes_prediction_dataset.csv"
STOCK_CSV = DATA / "GoogleStock_Dataset_V2.csv"
GOLD_CSV = DATA / "FINAL_USO.csv"

THU = ["thứ Hai", "thứ Ba", "thứ Tư", "thứ Năm", "thứ Sáu", "thứ Bảy", "Chủ nhật"]
SMOKE_VN = {"never": "chưa từng hút", "No Info": "không có thông tin", "former": "đã bỏ thuốc",
            "current": "đang hút", "not current": "hiện không hút", "ever": "từng hút"}
GENDER_VN = {"Female": "nữ", "Male": "nam", "Other": "giới tính khác"}


def vnum(x, d=2):
    """Số kiểu Việt trong tài liệu và câu hỏi: 1.234,56 (dùng nhất quán để truy xuất từ khoá khớp được)."""
    s = f"{x:,.{d}f}"
    return s.replace(",", "\0").replace(".", ",").replace("\0", ".")


# ---------------------------------------------------------------- khái niệm (tầng ngữ nghĩa của đồ thị)
def age_group(a):
    """Nhóm tuổi 5 bậc dùng chung cho tài liệu, đồ thị và câu hỏi."""
    if a < 20:
        return "dưới 20"
    if a < 40:
        return "20–39"
    if a < 60:
        return "40–59"
    if a < 80:
        return "60–79"
    return "từ 80"


def bmi_class(b):
    """Phân loại BMI theo WHO (người trưởng thành)."""
    if b < 18.5:
        return "thiếu cân"
    if b < 25:
        return "bình thường"
    if b < 30:
        return "thừa cân"
    return "béo phì"


def hba1c_class(h):
    """Ngưỡng HbA1c theo ADA: < 5,7 bình thường; 5,7–6,4 tiền tiểu đường; ≥ 6,5 ngưỡng chẩn đoán."""
    if h < 5.7:
        return "bình thường"
    if h < 6.5:
        return "tiền tiểu đường"
    return "ngưỡng chẩn đoán"


def glucose_class(g):
    """Ngưỡng tham chiếu đường huyết (mg/dL); bộ dữ liệu không ghi lúc đói hay ngẫu nhiên nên chỉ dùng để nhóm."""
    if g < 100:
        return "dưới 100"
    if g < 126:
        return "100–125"
    if g < 200:
        return "126–199"
    return "từ 200"


def movement(r):
    """Phân loại biến động một phiên theo lợi suất ngày (%)."""
    if np.isnan(r):
        return "phiên đầu"
    if r <= -3:
        return "giảm mạnh"
    if r < -0.5:
        return "giảm"
    if r <= 0.5:
        return "đi ngang"
    if r < 3:
        return "tăng"
    return "tăng mạnh"


# ---------------------------------------------------------------- nạp và làm sạch
def load_diabetes():
    """Đọc 100.000 hồ sơ, bỏ 3.854 dòng trùng lặp như dungvt, gán mã hồ sơ ổn định P000000…"""
    df = pd.read_csv(DIAB_CSV).drop_duplicates().reset_index(drop=True)
    df.insert(0, "pid", [f"P{i:06d}" for i in range(len(df))])
    df["age_group"] = df["age"].map(age_group)
    df["bmi_class"] = df["bmi"].map(bmi_class)
    df["hba1c_class"] = df["HbA1c_level"].map(hba1c_class)
    df["glucose_class"] = df["blood_glucose_level"].map(glucose_class)
    return df


def load_stock(start="2015-01-01"):
    """Giá Alphabet theo ngày (đã điều chỉnh chia tách 20:1 năm 2022), lọc 2015-01-02 → 2024-01-10."""
    df = pd.read_csv(STOCK_CSV, index_col=0)
    df = df[df["Date"] >= start].reset_index(drop=True)
    df["Date"] = pd.to_datetime(df["Date"])
    df["ret"] = df["Close"].pct_change() * 100
    df["move"] = df["ret"].map(movement)
    df["did"] = "D" + df["Date"].dt.strftime("%Y-%m-%d")
    df["month"] = df["Date"].dt.strftime("%Y-%m")
    df["year"] = df["Date"].dt.year
    return df


GOLD_COLS = {  # cột -> (tên tiếng Việt, đơn vị) — chỉ các cột dùng trong tài liệu
    "Close": ("giá đóng cửa quỹ vàng GLD", "USD"),
    "SP_close": ("S&P 500 (quỹ SPY)", "USD"),
    "DJ_close": ("chỉ số Dow Jones", "điểm"),
    "EU_Price": ("tỷ giá EUR/USD", ""),
    "OF_Price": ("dầu Brent", "USD/thùng"),
    "OS_Price": ("dầu WTI", "USD/thùng"),
    "USB_Price": ("lợi suất trái phiếu Mỹ", "%"),
    "PLT_Price": ("bạch kim", "USD/ounce"),
    "PLD_Price": ("palađi", "USD/ounce"),
    "USDI_Price": ("chỉ số USD", "điểm"),
    "GDX_Close": ("quỹ cổ phiếu mỏ vàng GDX", "USD"),
    "USO_Close": ("quỹ dầu USO", "USD"),
}


def load_gold():
    """FINAL_USO.csv: cột Open…Adj Close là quỹ vàng SPDR Gold Shares (GLD); USO_* là quỹ dầu, không phải vàng."""
    df = pd.read_csv(GOLD_CSV)
    df["Date"] = pd.to_datetime(df["Date"])
    df["ret"] = df["Close"].pct_change() * 100
    df["move"] = df["ret"].map(movement)
    df["gid"] = "G" + df["Date"].dt.strftime("%Y-%m-%d")
    df["month"] = df["Date"].dt.strftime("%Y-%m")
    df["year"] = df["Date"].dt.year
    return df


# ---------------------------------------------------------------- văn bản hoá (bản ghi -> tài liệu)
def diabetes_doc(r):
    """Một hồ sơ -> câu mô tả giữ đủ tên thuộc tính, giá trị và khái niệm suy ra (nhóm tuổi, phân loại BMI…)."""
    yes = lambda v: "có" if v else "không"
    return (f"Hồ sơ {r.pid}: bệnh nhân {GENDER_VN[r.gender]}, {vnum(r.age, 0 if r.age >= 2 else 2)} tuổi "
            f"(nhóm tuổi {r.age_group}), {yes(r.hypertension)} tăng huyết áp, {yes(r.heart_disease)} bệnh tim, "
            f"hút thuốc: {SMOKE_VN[r.smoking_history]}. Chỉ số BMI {vnum(r.bmi)} ({r.bmi_class}), "
            f"HbA1c {vnum(r.HbA1c_level, 1)}% ({r.hba1c_class}), đường huyết {r.blood_glucose_level} mg/dL "
            f"(nhóm {r.glucose_class}). Nhãn: {'mắc tiểu đường' if r.diabetes else 'không mắc tiểu đường'}.")


def stock_doc(r):
    chg = "" if np.isnan(r.ret) else f" Thay đổi so với phiên trước {vnum(r.ret)}% ({r.move})."
    d = r.Date
    return (f"Phiên giao dịch ngày {d:%d/%m/%Y} ({THU[d.weekday()]}) của cổ phiếu Alphabet (Google): "
            f"giá mở cửa {vnum(r.Open)} USD, cao nhất {vnum(r.High)} USD, thấp nhất {vnum(r.Low)} USD, "
            f"đóng cửa {vnum(r.Close)} USD, khối lượng {vnum(r.Volume, 0)} cổ phiếu.{chg}")


def gold_doc(r):
    d = r.Date
    parts = [f"Ngày {d:%d/%m/%Y} ({THU[d.weekday()]}): quỹ vàng GLD mở cửa {vnum(r.Open)} USD, cao nhất "
             f"{vnum(r.High)}, thấp nhất {vnum(r.Low)}, đóng cửa {vnum(r.Close)} USD, khối lượng {vnum(r.Volume, 0)}"]
    if not np.isnan(r.ret):
        parts[0] += f", thay đổi {vnum(r.ret)}% ({r.move})"
    for c in ("SP_close", "DJ_close", "EU_Price", "OF_Price", "OS_Price", "USB_Price", "USDI_Price",
              "GDX_Close", "USO_Close"):
        name, unit = GOLD_COLS[c]
        parts.append(f"{name} {vnum(getattr(r, c), 4 if c == 'EU_Price' else 2)}{(' ' + unit) if unit else ''}")
    return "; ".join(parts) + "."


# ---------------------------------------------------------------- tài liệu tổng hợp
def _rate(s):
    return f"{vnum(100 * s.mean())}%"


def diabetes_summaries(df):
    """Thống kê theo nhóm: giới × nhóm tuổi × tăng huyết áp, hút thuốc, phân loại HbA1c/BMI/đường huyết."""
    docs = []

    def add(key, title, g):
        docs.append({"id": f"S-{key}", "kind": "summary", "group": key,
                     "text": (f"Thống kê nhóm {title}: {vnum(len(g), 0)} hồ sơ, tỷ lệ mắc tiểu đường {_rate(g.diabetes)}, "
                              f"HbA1c trung bình {vnum(g.HbA1c_level.mean())}%, đường huyết trung bình "
                              f"{vnum(g.blood_glucose_level.mean(), 1)} mg/dL, BMI trung bình {vnum(g.bmi.mean())}, "
                              f"tuổi trung bình {vnum(g.age.mean(), 1)}."),
                     "n": len(g), "rate": float(g.diabetes.mean()), "hba1c": float(g.HbA1c_level.mean()),
                     "glucose": float(g.blood_glucose_level.mean()), "bmi": float(g.bmi.mean())})

    for (gd, ag, ht), g in df[df.gender != "Other"].groupby(["gender", "age_group", "hypertension"]):
        add(f"{gd}|{ag}|{ht}", f"bệnh nhân {GENDER_VN[gd]} nhóm tuổi {ag} "
            f"{'có' if ht else 'không'} tăng huyết áp", g)
    for col, label in (("smoking_history", "hút thuốc"), ("hba1c_class", "HbA1c"), ("bmi_class", "BMI"),
                       ("glucose_class", "đường huyết"), ("gender", "giới tính"), ("heart_disease", "bệnh tim")):
        for v, g in df.groupby(col):
            vv = SMOKE_VN.get(v, GENDER_VN.get(v, v)) if col in ("smoking_history", "gender") else (
                ("có" if v else "không") if col == "heart_disease" else v)
            add(f"{col}={v}", f"{label} {vv}", g)
    return docs


def _period_doc(prefix, key, g, what, unit="USD"):
    hi, lo = g.loc[g.High.idxmax()], g.loc[g.Low.idxmin()]
    chg = 100 * (g.Close.iloc[-1] / g.Close.iloc[0] - 1)
    vol = g.ret.std()
    return {"id": f"{prefix}-{key}", "kind": "summary", "group": key,
            "text": (f"Tổng hợp {what} giai đoạn {key}: {len(g)} phiên; đóng cửa phiên đầu {vnum(g.Close.iloc[0])} {unit}, "
                     f"phiên cuối {vnum(g.Close.iloc[-1])} {unit}, thay đổi {vnum(chg)}%; cao nhất {vnum(hi.High)} {unit} "
                     f"ngày {hi.Date:%d/%m/%Y}, thấp nhất {vnum(lo.Low)} {unit} ngày {lo.Date:%d/%m/%Y}; giá đóng cửa trung "
                     f"bình {vnum(g.Close.mean())} {unit}; độ biến động (độ lệch chuẩn lợi suất ngày) "
                     f"{vnum(0 if np.isnan(vol) else vol)}%; khối lượng trung bình {vnum(g.Volume.mean(), 0)}."),
            "n": len(g), "first": float(g.Close.iloc[0]), "last": float(g.Close.iloc[-1]), "chg": float(chg),
            "high": float(hi.High), "high_date": f"{hi.Date:%Y-%m-%d}", "low": float(lo.Low),
            "low_date": f"{lo.Date:%Y-%m-%d}", "mean": float(g.Close.mean()),
            "vol": float(0 if np.isnan(vol) else vol)}


def period_summaries(df, prefix, what):
    """Tài liệu tổng hợp theo tháng (YYYY-MM) và theo năm."""
    docs = [_period_doc(prefix, m, g.reset_index(drop=True), what) for m, g in df.groupby("month")]
    docs += [_period_doc(prefix, str(y), g.reset_index(drop=True), what) for y, g in df.groupby("year")]
    return docs


def build_corpus(domain):
    """Trả (DataFrame, list[dict] tài liệu {id, kind, text, ...}) cho một miền."""
    if domain == "diabetes":
        df = load_diabetes()
        docs = [{"id": r.pid, "kind": "record", "text": diabetes_doc(r)} for r in df.itertuples()]
        docs += diabetes_summaries(df)
    elif domain == "stock":
        df = load_stock()
        docs = [{"id": r.did, "kind": "record", "text": stock_doc(r)} for r in df.itertuples()]
        docs += period_summaries(df, "M", "cổ phiếu Alphabet")
    elif domain == "gold":
        df = load_gold()
        docs = [{"id": r.gid, "kind": "record", "text": gold_doc(r)} for r in df.itertuples()]
        docs += period_summaries(df, "M", "quỹ vàng GLD")
    else:
        raise KeyError(domain)
    return df, docs
