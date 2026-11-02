"""
Tiện ích dùng chung cho các thực nghiệm của báo cáo cuối khóa (lab/chX_*.py).

Mỗi script ghi số liệu vào lab/results/<chương>.json, hình vào report/figures/ và trọng số mô hình
vào data/models/ để tầng deploy (src/api/routes_lab.py) nạp lại.
"""

import json
import random
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DATA = ROOT / "lab" / "data"
RESULTS = ROOT / "lab" / "results"
FIGURES = ROOT / "report" / "figures"
MODELS = ROOT / "data" / "models"
for d in (RESULTS, FIGURES, MODELS):
    d.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Liberation Serif", "DejaVu Serif"], "font.size": 10,
    "axes.titlesize": 10.5, "axes.labelsize": 10, "legend.fontsize": 9, "figure.dpi": 110,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
    "savefig.bbox": "tight",
})
GREY, ACCENT = "#8c8c8c", "#2563eb"
PALETTE = ["#2563eb", "#f97316", "#16a34a", "#dc2626", "#9333ea", "#0891b2", "#ca8a04", "#64748b"]


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def save_fig(fig, name: str) -> None:
    fig.savefig(FIGURES / f"{name}.pdf")
    plt.close(fig)
    print(f"  hình → report/figures/{name}.pdf")


def save_results(chapter: str, payload: dict) -> None:
    path = RESULTS / f"{chapter}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(f"  số liệu → {path.relative_to(ROOT)}")


@contextmanager
def timer(store: dict, key: str):
    t0 = time.perf_counter()
    yield
    store[key] = round(time.perf_counter() - t0, 3)


def r4(x) -> float:
    return round(float(x), 4)
