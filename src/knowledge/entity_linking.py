"""
Liên kết thực thể cho GraphRAG: tìm các nút của đồ thị tri thức Neo4j được nhắc tới trong câu hỏi tự do.

So khớp trên chuỗi đã chuẩn hóa (chữ thường, bỏ dấu tiếng Việt) theo tên nút và các từ đồng nghĩa / tên biệt dược
thường gặp (ADA 2024, Bộ Y tế QĐ 5481/QĐ-BYT, Vinmec, Nhà thuốc FPT Long Châu).
"""

import unicodedata
from typing import Iterable, List

ALIASES = {
    "Đái tháo đường típ 2": ["dai thao duong", "tieu duong", "dtd", "type 2", "tuyp 2", "tip 2"],
    "Tiền đái tháo đường": ["tien dai thao duong", "tien tieu duong", "prediabetes"],
    "Thừa cân/béo phì": ["thua can", "beo phi", "bmi", "can nang"],
    "Tăng huyết áp": ["tang huyet ap", "huyet ap", "cao huyet ap"],
    "Tiền sử gia đình ĐTĐ": ["gia dinh", "bo me", "di truyen"],
    "Rối loạn lipid máu": ["lipid", "mo mau", "hdl", "ldl", "triglyceride", "cholesterol"],
    "Ít vận động": ["it van dong", "luoi van dong", "ngoi nhieu"],
    "Béo bụng": ["vong eo", "beo bung", "bung to"],
    "HbA1c": ["hba1c", "a1c"],
    "Glucose huyết tương lúc đói": ["luc doi", "duong huyet doi", "fpg", "glucose doi"],
    "Nghiệm pháp dung nạp glucose (OGTT)": ["ogtt", "dung nap", "sau uong duong", "2 gio"],
    "Metformin": ["metformin", "biguanide", "panfor"],
    "Metformin (Glucophage 500mg/850mg)": ["glucophage", "glucophage xr", "metformin 500", "metformin 850"],
    "Ức chế SGLT2": ["sglt2", "sglt2i", "suy tim", "tha duong qua nuoc tieu"],
    "Dapagliflozin (Forxiga 10mg)": ["dapagliflozin", "forxiga", "xigduo"],
    "Empagliflozin (Jardiance 25mg)": ["empagliflozin", "jardiance", "synjardy"],
    "Sitagliptin (Januvia 100mg)": ["dpp-4", "dpp4", "sitagliptin", "januvia", "janumet", "galvus", "vildagliptin"],
    "Linagliptin (Trajenta 5mg)": ["linagliptin", "trajenta"],
    "Đồng vận GLP-1": ["glp-1", "glp1", "semaglutide", "liraglutide", "ozempic", "victoza", "giam can"],
    "Phối hợp Metformin + SGLT2i/GLP-1 RA": ["phoi hop thuoc", "phac do kep", "da tri lieu"],
    "Insulin": ["insulin", "tiem insulin"],
    "Insulin Glargine (Lantus SoloStar)": ["lantus", "solostar", "glargine", "insulin nen"],
    "Gliclazide MR (Diamicron MR 60mg)": ["diamicron", "gliclazide", "sulfonylurea", "amaryl", "glimepiride"],
    "Statin": ["statin", "rosuvastatin", "atorvastatin", "ha mo mau"],
    "Bệnh thận ĐTĐ": ["than", "egfr", "albumin", "suy than", "ckd"],
    "Suy thận nặng (eGFR < 30)": ["egfr < 30", "suy than nang", "chong chi dinh"],
    "Bệnh võng mạc ĐTĐ": ["mat", "vong mac", "day mat", "mu loa"],
    "Bệnh thần kinh ngoại biên": ["ban chan", "than kinh", "te bi", "loet chan"],
    "Bệnh tim mạch xơ vữa": ["tim mach", "xo vua", "nhoi mau", "dot quy", "ascvd"],
    "Hạ đường huyết": ["ha duong", "tut duong", "run tay", "va mo hoi"],
    "Hút thuốc lá": ["hut thuoc", "thuoc la"],
    "Tuổi ≥ 35": ["tuoi", "35 tuoi", " sàng lọc"],
}


def fold(s: str) -> str:
    """Chữ thường + bỏ dấu tiếng Việt (đ → d)."""
    s = unicodedata.normalize("NFD", s.lower().replace("đ", "d"))
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


def link_entities(question: str, nodes: Iterable[str] = None) -> List[str]:
    fq = fold(question)
    candidates = list(nodes) if nodes is not None else list(ALIASES.keys())
    return [n for n in candidates if fold(n) in fq or any(fold(a) in fq for a in ALIASES.get(n, []))]
