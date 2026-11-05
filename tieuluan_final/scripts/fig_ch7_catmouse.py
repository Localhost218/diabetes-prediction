"""Hình Chương 7B: lưới mèo đuổi chuột + quỹ đạo một ván, đường cong học (B, C) và ma trận đối kháng (C)."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

TLF = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TLF / "src"))
from dm import catmouse_base as cm  # noqa: E402

FIG = TLF / "report" / "figures"
plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif", "DejaVu Serif"], "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 110})
BLUE, ORANGE, GREEN, RED, GREY = "#2563eb", "#f97316", "#16a34a", "#dc2626", "#8c8c8c"
R = json.loads((TLF / "outputs" / "metrics" / "ch7_catmouse.json").read_text())

# ---------------------------------------------------------------- (1) lưới + quỹ đạo Q-learning vs luật
env = cm.CatMouseEnv()
q = cm.QLearningCat(env)
q.Q = np.load(TLF / "models" / "ch7" / "cat_q_single.npy")
start = ((0, 0), (7, 7))
fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.2))
for ax, agent, title in ((axes[0], cm.RuleBasedCat(), "Mèo theo luật (Manhattan)"), (axes[1], q, "Mèo Q-learning")):
    ep = cm.run_episode(env, agent, cat=start[0], mouse=start[1], seed=3)
    for r in range(env.R):
        for c in range(env.C):
            ax.add_patch(plt.Rectangle((c, r), 1, 1, color="#334155" if env.layout[r][c] == "#" else "#f8fafc",
                                       ec="#cbd5e1", lw=0.6))
    cat = np.array([f["cat"] for f in ep["frames"]]) + 0.5
    mouse = np.array([f["mouse"] for f in ep["frames"]]) + 0.5
    ax.plot(cat[:, 1], cat[:, 0], "-o", color=BLUE, ms=3, lw=1.6, label="mèo")
    ax.plot(mouse[:, 1], mouse[:, 0], "-s", color=ORANGE, ms=3, lw=1.2, alpha=0.8, label="chuột")
    ax.text(cat[0, 1], cat[0, 0], "M", ha="center", va="center", color="white", fontsize=8, weight="bold",
            bbox=dict(boxstyle="circle,pad=0.2", fc=BLUE, ec="none"))
    ax.text(mouse[0, 1], mouse[0, 0], "C", ha="center", va="center", color="white", fontsize=8, weight="bold",
            bbox=dict(boxstyle="circle,pad=0.2", fc=ORANGE, ec="none"))
    res = "bắt được sau %d bước" % ep["steps"] if ep["caught"] else "không bắt được sau %d bước" % ep["steps"]
    ax.set(xlim=(0, env.C), ylim=(env.R, 0), aspect="equal", title=f"{title}: {res}", xticks=[], yticks=[])
axes[0].legend(loc="lower left", fontsize=8, framealpha=0.9)
fig.tight_layout()
fig.savefig(FIG / "ch7_catmouse_grid.pdf")
plt.close(fig)

# ---------------------------------------------------------------- (2) đường cong học A, B, C
fig, axes = plt.subplots(1, 2, figsize=(9, 3.3))
for key, lab, col in (("A", "A: một mèo Q vs chuột luật", BLUE), ("B", "B: hai mèo IQL vs chuột luật", GREEN),
                      ("C", "C: mèo Q vs chuột Q (đối kháng)", RED)):
    cv = R[key]["curve"]
    ep = [p["episode"] for p in cv]
    axes[0].plot(ep, [p["catch_rate"] for p in cv], color=col, lw=1.6, label=lab)
    axes[1].plot(ep, [p["avg_steps"] for p in cv], color=col, lw=1.6, label=lab)
axes[0].set(xlabel="tập huấn luyện", ylabel="tỷ lệ bắt được (khi huấn luyện, có ε)", ylim=(0, 1.02))
axes[1].set(xlabel="tập huấn luyện", ylabel="số bước trung bình mỗi ván")
axes[0].legend(fontsize=8, loc="lower right")
fig.tight_layout()
fig.savefig(FIG / "ch7_catmouse_curves.pdf")
plt.close(fig)

# ---------------------------------------------------------------- (3) ma trận đối kháng
tab = R["C"]["table"]
cats = list(dict.fromkeys(r["cat"] for r in tab))
mice = list(dict.fromkeys(r["mouse"] for r in tab))
M = np.array([[next(r["catch_rate"] for r in tab if r["cat"] == c and r["mouse"] == m) for m in mice] for c in cats])
fig, ax = plt.subplots(figsize=(6.4, 3.2))
im = ax.imshow(M, cmap="Blues", vmin=0, vmax=1)
for i in range(len(cats)):
    for j in range(len(mice)):
        ax.text(j, i, f"{M[i, j]:.0%}", ha="center", va="center", color="white" if M[i, j] > 0.6 else "black")
ax.set_xticks(range(len(mice)), [m.replace(" (", "\n(") for m in mice])
ax.set_yticks(range(len(cats)), ["mèo " + c for c in cats])
ax.set_title("Tỷ lệ mèo bắt được chuột (300 ván)", fontsize=10)
for s in ax.spines.values():
    s.set_visible(False)
fig.colorbar(im, ax=ax, fraction=0.04)
fig.tight_layout()
fig.savefig(FIG / "ch7_catmouse_matrix.pdf")
plt.close(fig)
print("đã vẽ 3 hình Chương 7B")
