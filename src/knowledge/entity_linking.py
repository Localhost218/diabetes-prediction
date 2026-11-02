"""
Liên kết thực thể cho GraphRAG: tìm các nút của đồ thị tri thức được nhắc tới trong câu hỏi tự do.

So khớp trên chuỗi đã chuẩn hóa (chữ thường, bỏ dấu tiếng Việt) theo tên nút và các cách gọi thường gặp.
Dùng chung cho API (/api/lab/ch5/search) và thực nghiệm (lab/ch5_rag.py).
"""

import unicodedata
from typing import Iterable, List

ALIASES = {
    "Thừa cân/béo phì": ["thua can", "beo phi", "bmi"], "Tăng huyết áp": ["tang huyet ap", "huyet ap"],
    "Tiền sử gia đình ĐTĐ": ["gia dinh"], "Rối loạn lipid máu": ["lipid", "mo mau", "hdl", "ldl"],
    "Ít vận động": ["it van dong"], "Béo bụng": ["vong eo", "beo bung"], "HbA1c": ["hba1c"],
    "Glucose huyết tương lúc đói": ["luc doi", "duong huyet doi"], "Metformin": ["metformin"],
    "Ức chế SGLT2": ["sglt2", "suy tim"], "Đồng vận GLP-1": ["glp-1", "glp1", "giam can"], "Insulin": ["insulin"],
    "Statin": ["statin"], "Bệnh thận ĐTĐ": ["than", "egfr", "albumin"], "Bệnh võng mạc ĐTĐ": ["mat", "vong mac"],
    "Bệnh thần kinh ngoại biên": ["ban chan", "than kinh"], "Hạ đường huyết": ["ha duong"],
    "Tiền đái tháo đường": ["tien dai thao duong", "tien tieu duong"], "Bệnh tim mạch xơ vữa": ["tim mach"],
    "Suy thận nặng (eGFR < 30)": ["egfr"], "Nghiệm pháp dung nạp glucose (OGTT)": ["ogtt", "dung nap"],
    "Hút thuốc lá": ["hut thuoc", "thuoc la"], "Tuổi ≥ 35": ["tuoi"],
}


def fold(s: str) -> str:
    """Chữ thường + bỏ dấu tiếng Việt (đ → d)."""
    s = unicodedata.normalize("NFD", s.lower().replace("đ", "d"))
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


def link_entities(question: str, nodes: Iterable[str]) -> List[str]:
    fq = fold(question)
    return [n for n in nodes if fold(n) in fq or any(a in fq for a in ALIASES.get(n, []))]
