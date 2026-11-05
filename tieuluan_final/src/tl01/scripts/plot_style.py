"""Kiểu vẽ matplotlib dùng chung cho mọi hình dữ liệu của báo cáo (xuất PDF vector).

Mọi hình được vẽ ở đúng bề rộng vùng chữ của trang A4 (16 cm = 6,3 inch) với chữ 11-12 pt, nên khi
chèn vào LaTeX ở 100 % bề rộng, chữ trong hình có cùng cỡ với chữ thân bài, không bị thu nhỏ.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FIG = Path(__file__).resolve().parents[3] / "Report" / "TieuLuan 01" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
W = 6.3            # bề rộng vùng chữ (inch)
H1 = 3.4           # chiều cao chuẩn của một hàng biểu đồ

# Bảng màu an toàn cho người mù màu (Okabe-Ito), cố định một màu cho mỗi mô hình ở mọi hình
C = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "red": "#D55E00", "purple": "#CC79A7",
     "sky": "#56B4E9", "yellow": "#F0E442", "gray": "#7F7F7F", "navy": "#123A75"}
SEQ = [C["blue"], C["orange"], C["green"], C["red"], C["purple"], C["sky"], C["gray"]]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.labelsize": 11.5, "xtick.labelsize": 10.5, "ytick.labelsize": 10.5, "legend.fontsize": 10,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.3,
    "axes.linewidth": 1.0, "lines.linewidth": 2.0, "legend.frameon": False, "figure.dpi": 150,
    "savefig.bbox": "tight", "axes.prop_cycle": plt.cycler(color=SEQ), "pdf.fonttype": 42,
    "axes.titlepad": 8,
})


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf")
    plt.close(fig)
    print("->", name)


def vn(x, d=0):
    """Định dạng số kiểu Việt: 12.345,6"""
    s = f"{x:,.{d}f}"
    return s.replace(",", "\0").replace(".", ",").replace("\0", ".")
