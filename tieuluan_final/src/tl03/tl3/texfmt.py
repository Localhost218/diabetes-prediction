"""Định dạng số kiểu Việt và sinh bảng booktabs cho báo cáo LaTeX (dùng chung bởi scripts/chN_tables.py)."""
import math
import re
import unicodedata

from .common import GEN


def num(v, decimals=2, sign=False):
    """12345.678 -> 12{.}345{,}68 (dấu chấm ngăn nghìn, phẩy thập phân, an toàn trong chế độ toán)."""
    s = f"{v:+,.{decimals}f}" if sign else f"{v:,.{decimals}f}"
    s = s.replace(",", "\0").replace(".", "{,}").replace("\0", "{.}")
    s = s.replace("-", "\\ensuremath{-}")
    return s.replace("+", "\\ensuremath{+}") if sign else s


def sci(v):
    if v == 0:
        return "0"
    m, e = f"{v:.2e}".split("e")
    return f"\\ensuremath{{{m.replace('.', '{,}')} \\times 10^{{{int(e)}}}}}"


def fmt(v):
    """Định dạng tự động theo độ lớn; dùng cho macro \\SL và ô bảng."""
    if isinstance(v, bool):
        return "đúng" if v else "sai"
    if isinstance(v, int):
        return num(v, 0)
    if isinstance(v, float):
        if math.isnan(v):
            return "NaN"
        if v != 0 and abs(v) < 1e-3:
            return sci(v)
        d = 4 if abs(v) < 1 else 3 if abs(v) < 10 else 2 if abs(v) < 1000 else 0
        return num(v, d)
    return tex_escape(str(v))


def pct(v, d=2):
    """0.9734 -> 97{,}34"""
    return num(100 * v, d)


def tex_escape(s):
    s = unicodedata.normalize("NFC", str(s))
    s = (s.replace("\\", "\\textbackslash{}").replace("{", "\\{").replace("}", "\\}").replace("_", "\\_")
          .replace("%", "\\%").replace("&", "\\&").replace("#", "\\#").replace("$", "\\$")
          .replace("^", "\\^{}").replace("~", "\\textasciitilde{}"))
    return re.sub(r'"([^"]*)"', r"``\1''", s)


def wrap_head(h, align):
    """Ngắt tiêu đề cột dài thành hai dòng (\\makecell) để bảng nhiều cột không bị co nhỏ chữ."""
    if len(h) <= 9 or " " not in h or "\\\\" in h or "$" in h:
        return h
    spaces = [i for i, ch in enumerate(h) if ch == " "]
    cut = min(spaces, key=lambda i: abs(i - len(h) / 2))
    return f"\\makecell[{align}]{{{h[:cut]}\\\\{h[cut + 1:]}}}"


def best(values, formatted, higher=True):
    """In đậm ô tốt nhất của một cột."""
    b = max(values) if higher else min(values)
    return [f"\\textbf{{{f}}}" if v == b else f for v, f in zip(values, formatted)]


def table(name, spec, header, rows, note=None):
    """Ghi Report/TieuLuan 03/generated/<name>.tex (+ <name>_note.tex). rows: list[list[str]] hoặc chuỗi TeX nguyên
    dòng (ví dụ '\\midrule'). name phải có tiền tố chương (chN_). Trong chương: \\bang{<name>}{chú thích}{nhãn}."""
    aligns = [c for c in re.sub(r"\{[^}]*\}", "", spec) if c in "lrcpLCRX"]
    header = [wrap_head(h, "l" if a in "lpL" else "r" if a in "rR" else "c") for h, a in zip(header, aligns)]
    out = [f"\\begin{{tabular}}{{{spec}}}", "\\toprule", " & ".join(header) + " \\\\", "\\midrule"]
    out += [r if isinstance(r, str) else " & ".join(str(c) for c in r) + " \\\\" for r in rows]
    out += ["\\bottomrule", "\\end{tabular}"]
    (GEN / f"{name}.tex").write_text("\n".join(out) + "\n", encoding="utf-8", newline="\n")
    note_path = GEN / f"{name}_note.tex"
    if note:
        note_path.write_text(note + "\n", encoding="utf-8", newline="\n")
    else:
        note_path.unlink(missing_ok=True)
    return name
