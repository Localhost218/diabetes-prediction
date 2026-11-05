"""Kiểu vẽ matplotlib dùng chung cho mọi hình dữ liệu của báo cáo (xuất PDF vector).

Hình được vẽ đúng bề rộng vùng chữ A4 (16 cm = 6,3 inch) với chữ 11,5-13 pt; chèn vào LaTeX ở 100 % bề rộng
nên chữ trong hình xấp xỉ cỡ chữ thân bài, không bị thu nhỏ.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FIG = Path(__file__).resolve().parents[3] / "Report" / "TieuLuan 02" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
W = 6.3
H1 = 3.6

C = {"blue": "#2563EB", "orange": "#F59E0B", "green": "#16A34A", "red": "#DC2626", "purple": "#7C3AED",
     "teal": "#0D9488", "pink": "#DB2777", "sky": "#0EA5E9", "gray": "#6B7280", "navy": "#123A75",
     "lime": "#65A30D", "brown": "#A16207"}
SEQ = [C[k] for k in ("blue", "orange", "green", "red", "purple", "teal", "pink", "sky", "brown", "gray")]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11.5, "axes.titlesize": 12.5, "axes.titleweight": "bold",
    "axes.labelsize": 12, "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 10.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.3,
    "axes.linewidth": 1.0, "lines.linewidth": 2.1, "legend.frameon": False, "figure.dpi": 150,
    "savefig.bbox": "tight", "axes.prop_cycle": plt.cycler(color=SEQ), "pdf.fonttype": 42, "axes.titlepad": 8,
})


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf")
    plt.close(fig)
    print("->", name)


def vn(x, d=0):
    """Định dạng số kiểu Việt: 12.345,6"""
    s = f"{x:,.{d}f}"
    return s.replace(",", "\0").replace(".", ",").replace("\0", ".")
