"""Kiểu vẽ matplotlib dùng chung cho mọi hình dữ liệu của báo cáo (xuất PDF vector).

Hình vẽ đúng bề rộng vùng chữ A4 (16 cm = 6,3 inch) với chữ 11,5–13 pt; chèn vào LaTeX ở 100 % bề rộng nên chữ
trong hình xấp xỉ cỡ chữ thân bài, không bị thu nhỏ. Hình hai cột (\\haihinh) vẽ rộng W2 = 3,6 in.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .common import FIGS  # noqa: E402

W = 6.3
W2 = 3.6
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
    "axes.formatter.use_locale": False,
})


def save(fig, name, ext="pdf", dpi=200):
    """Ghi Report/TieuLuan 03/figures/<name>.<ext>; name phải có tiền tố chương (chN_)."""
    path = FIGS / f"{name}.{ext}"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print("->", path.name, flush=True)
    return path


def vn(x, d=0):
    """Định dạng số kiểu Việt trên trục/nhãn: 12.345,6"""
    s = f"{x:,.{d}f}"
    return s.replace(",", "\0").replace(".", ",").replace("\0", ".")


def vn_axis(ax, which="y", d=0):
    """Đặt nhãn trục theo kiểu số Việt."""
    from matplotlib.ticker import FuncFormatter
    f = FuncFormatter(lambda v, _: vn(v, d))
    (ax.yaxis if which == "y" else ax.xaxis).set_major_formatter(f)
