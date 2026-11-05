"""Hình dữ liệu Chương 7 (figures/ch7_*.pdf) từ ch7_gsm8k.json."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tl3.common import load_json  # noqa: E402
from tl3.plot_style import C, W, plt, save, vn  # noqa: E402

SYS = ["direct", "cot", "pal", "multi"]
NAMES = {"direct": "Trực tiếp", "cot": "CoT", "pal": "PAL + Python", "multi": "Đa tác tử"}
SHORT = {"direct": "Trực\ntiếp", "cot": "CoT", "pal": "PAL", "multi": "Đa\ntác tử"}  # nhãn trục của ô hẹp
COL = {"direct": C["gray"], "cot": C["blue"], "pal": C["green"], "multi": C["purple"]}
ERRS = {"no_answer": ("không ra số", C["gray"]), "arithmetic": ("sai số học", C["orange"]),
        "reasoning": ("sai lập luận", C["red"]), "coordination": ("lỗi phối hợp", C["purple"])}


def fig_acc(R):
    fig, ax = plt.subplots(figsize=(W, 3.5))
    S = R["systems"]
    for i, s in enumerate(SYS):
        a = 100 * S[s]["acc"]
        ax.bar(i, a, 0.6, color=COL[s], yerr=[[a - 100 * S[s]["ci_lo"]], [100 * S[s]["ci_hi"] - a]], capsize=6)
        ax.text(i, 100 * S[s]["ci_hi"] + 1.5, f"{vn(a, 1)} %", ha="center", fontsize=12, fontweight="bold")
    ax.set_xticks(range(4), [NAMES[s] for s in SYS])
    ax.set_ylabel("accuracy (%)")
    ax.set_ylim(0, max(100 * S[s]["ci_hi"] for s in SYS) + 12)
    # không đặt tiêu đề: chú thích hình đã nói đủ, tiêu đề dài hơn trục làm hình bị co nhỏ khi chèn vào trang
    save(fig, "ch7_acc")


def fig_cost(R):
    S = R["systems"]
    fig, axes = plt.subplots(1, 3, figsize=(W, 3.3))
    for ax, (key, title) in zip(axes, [("calls", "lượt gọi LLM / câu"), ("tokens", "token / câu"),
                                       ("seconds", "giây / câu")]):
        v = [S[s][key] for s in SYS]
        ax.bar(range(4), v, color=[COL[s] for s in SYS])
        for i, x in enumerate(v):
            ax.text(i, x, vn(x, 1 if x < 100 else 0), ha="center", va="bottom", fontsize=10)
        ax.set_xticks(range(4), [SHORT[s] for s in SYS], fontsize=10.5)
        ax.set_title(title, fontsize=11.5)
        ax.margins(y=0.15)
    fig.tight_layout()
    save(fig, "ch7_cost")


def fig_pareto(R):
    S = R["systems"]
    fig, ax = plt.subplots(figsize=(W, 3.6))
    for s in SYS:
        ax.scatter(S[s]["tokens"], 100 * S[s]["acc"], s=60 + 40 * S[s]["calls"], color=COL[s], alpha=0.85,
                   edgecolor="black", zorder=3)
        # nhãn của điểm đầu biên Pareto đặt thấp hơn để không nằm trên đường nét đứt đi lên
        ax.annotate(f"{NAMES[s]}\n{vn(S[s]['calls'], 1)} lượt gọi", (S[s]["tokens"], 100 * S[s]["acc"]),
                    xytext=(10, -26 if s == "direct" else -4), textcoords="offset points", fontsize=10.5)
    order = sorted(SYS, key=lambda s: S[s]["tokens"])
    front, best = [], -1
    for s in order:
        if S[s]["acc"] > best:
            front.append(s)
            best = S[s]["acc"]
    ax.plot([S[s]["tokens"] for s in front], [100 * S[s]["acc"] for s in front], "--", color=C["gray"], zorder=1,
            label="biên Pareto")
    ax.set_xscale("log")
    ax.set_xlabel("token trung bình mỗi câu (log)")
    ax.set_ylabel("accuracy (%)")
    ax.legend(loc="lower right")
    ax.margins(x=0.25, y=0.2)
    save(fig, "ch7_pareto")


def fig_errors(R):
    S = R["systems"]
    fig, ax = plt.subplots(figsize=(W, 3.4))
    left = np.zeros(4)
    for e, (lab, c) in ERRS.items():
        v = np.array([S[s]["errors"][e] for s in SYS], float)
        ax.barh(range(4), v, left=left, color=c, label=lab)
        for i, (x, l0) in enumerate(zip(v, left)):
            if x >= 8:
                ax.text(l0 + x / 2, i, str(int(x)), ha="center", va="center", color="white", fontsize=10.5,
                        fontweight="bold")
        left += v
    ax.set_yticks(range(4), [NAMES[s] for s in SYS])
    ax.invert_yaxis()
    ax.set_xlabel(f"số câu sai (trên {R['n']} câu)")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.17))
    save(fig, "ch7_errors")


def fig_latency(R):
    S = R["systems"]
    fig, ax = plt.subplots(figsize=(W, 3.4))
    data = [S[s]["series"]["seconds"] for s in SYS]
    bp = ax.boxplot(data, vert=False, patch_artist=True, widths=0.55, showfliers=True,
                    flierprops={"markersize": 3, "alpha": 0.5})
    for p, s in zip(bp["boxes"], SYS):
        p.set_facecolor(COL[s])
        p.set_alpha(0.75)
    # nhãn p90 đặt bên phải điểm xa nhất của mỗi hàng để không đè lên hộp và râu
    for i, s in enumerate(SYS, 1):
        ax.text(max(S[s]["series"]["seconds"]) * 1.35, i, f"p90 = {vn(S[s]['seconds_p90'], 1)} s", fontsize=10,
                ha="left", va="center")
    ax.set_yticks(range(1, 5), [NAMES[s] for s in SYS])
    ax.set_xscale("log")
    ax.set_xlim(right=max(max(S[s]["series"]["seconds"]) for s in SYS) * 9)
    ax.set_xlabel("thời gian trả lời một câu (giây, log)")
    ax.invert_yaxis()
    save(fig, "ch7_latency")


def fig_only_correct(R):
    M = np.array(R["only_correct"]["rows"])
    fig, ax = plt.subplots(figsize=(W * 0.72, 3.9))
    im = ax.imshow(M, cmap="Purples")
    for i in range(4):
        for j in range(4):
            ax.text(j, i, "–" if i == j else str(M[i, j]), ha="center", va="center", fontsize=13,
                    color="white" if M[i, j] > M.max() * 0.6 else "black")
    labs = [NAMES[s] for s in SYS]
    ax.set_xticks(range(4), labs, rotation=20)
    ax.set_yticks(range(4), labs)
    ax.set_xlabel("... nhưng SAI ở hệ cột")
    ax.set_ylabel("ĐÚNG ở hệ hàng ...")
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.046)
    save(fig, "ch7_only_correct")


def fig_gate(R):
    m = R["multi"]
    fig, axes = plt.subplots(1, 2, figsize=(W, 3.3), gridspec_kw={"width_ratios": [1, 1.4]})
    # phần trăm nằm sâu trong miếng bánh, nhãn ở xa hơn, để hai lớp chữ không chồng nhau
    axes[0].pie([m["agree_n"], m["disagree_n"]], labels=[f"đồng thuận\n{m['agree_n']} câu",
                                                         f"bất đồng\n{m['disagree_n']} câu"],
                colors=[C["teal"], C["orange"]], autopct=lambda p: f"{vn(p, 1)} %", startangle=90,
                pctdistance=0.58, labeldistance=1.18,
                textprops={"fontsize": 10.5}, wedgeprops={"edgecolor": "white", "linewidth": 2})
    axes[0].set_title("Cổng đồng thuận")
    vals = [100 * m["agree_acc"], 100 * m["disagree_acc"], 100 * R["systems"]["multi"]["acc"], 100 * m["oracle_acc"]]
    labs = ["nhánh\nđồng\nthuận", "nhánh\nbất\nđồng", "cả hệ", "oracle\n(cận\ntrên)"]
    b = axes[1].bar(range(4), vals, color=[C["teal"], C["orange"], C["purple"], C["gray"]])
    for r, v in zip(b, vals):
        axes[1].text(r.get_x() + r.get_width() / 2, v + 1.5, f"{vn(v, 1)} %", ha="center", fontsize=10.5)
    axes[1].set_xticks(range(4), labs, fontsize=10)
    axes[1].set_ylabel("accuracy (%)")
    axes[1].set_ylim(0, 105)
    fig.tight_layout()
    save(fig, "ch7_gate")


def fig_difficulty(R):
    rows = R["difficulty"]["rows"]
    fig, ax = plt.subplots(figsize=(W, 3.4))
    x = np.arange(len(rows))
    for s in SYS:
        ax.plot(x, [100 * r[s] for r in rows], "o-", color=COL[s], label=NAMES[s], markersize=7)
    ax.set_xticks(x, [f"{r['bin']} phép\n({r['n']} câu)" for r in rows])
    ax.set_xlabel("độ khó: số phép tính trong lời giải mẫu")
    ax.set_ylabel("accuracy (%)")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.15))
    save(fig, "ch7_difficulty")


def fig_tokens(R):
    S = R["systems"]
    fig, ax = plt.subplots(figsize=(W, 3.3))
    bins = np.logspace(0, np.log10(max(max(S[s]["series"]["tokens_out"]) for s in SYS) + 1), 36)
    for s in SYS:
        ax.hist(np.array(S[s]["series"]["tokens_out"]) + 1, bins=bins, histtype="step", linewidth=2.2, color=COL[s],
                label=f"{NAMES[s]} (TB {vn(S[s]['tokens_out'], 0)})")
    ax.set_xscale("log")
    ax.set_xlabel("token sinh ra mỗi câu (log)")
    ax.set_ylabel("số câu")
    ax.legend(ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.0))  # ngoài trục: không che cột nào
    save(fig, "ch7_tokens")


def fig_tool(R):
    S = R["systems"]
    kinds = [("ok", "chạy được", C["green"]), ("runtime", "lỗi lúc chạy", C["red"]),
             ("no_function", "thiếu solution()", C["orange"]), ("import", "import bị chặn", C["purple"]),
             ("timeout", "quá thời gian", C["gray"])]
    have = [s for s in SYS if "tool" in S[s]]
    fig, ax = plt.subplots(figsize=(W, 2.6))
    for i, s in enumerate(have):
        t, left = S[s]["tool"], 0
        for k, lab, c in kinds:
            v = 100 * t[k] / t["runs"]
            ax.barh(i, v, left=left, color=c, label=lab if i == 0 else None)
            if v >= 4:
                ax.text(left + v / 2, i, f"{vn(v, 1)}", ha="center", va="center", color="white", fontsize=10)
            left += v
    ax.set_yticks(range(len(have)), [f"{NAMES[s]}\n({S[s]['tool']['runs']} lần chạy)" for s in have])
    ax.set_xlabel("tỷ lệ lần chạy mã trong sandbox (%)")
    ax.set_xlim(0, 100)
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.3), fontsize=9.5)
    save(fig, "ch7_tool")


def fig_calls(R):
    S = R["systems"]
    fig, ax = plt.subplots(figsize=(W, 3.1))
    w = 0.2
    maxc = max(max(S[s]["series"]["calls"]) for s in SYS)
    ks = np.arange(1, maxc + 1)
    for i, s in enumerate(SYS):
        c = np.array(S[s]["series"]["calls"])
        ax.bar(ks + (i - 1.5) * w, [(c == k).sum() for k in ks], w, color=COL[s], label=NAMES[s])
    ax.set_xticks(ks)
    ax.set_xlabel("số lượt gọi LLM cho một câu")
    ax.set_ylabel("số câu")
    ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0))  # ngoài trục: không che cột nào
    save(fig, "ch7_calls")


def fig_running(R):
    S = R["systems"]
    fig, ax = plt.subplots(figsize=(W, 3.1))
    for s in SYS:
        c = np.array(S[s]["series"]["correct"], float)
        ax.plot(np.arange(1, len(c) + 1), 100 * np.cumsum(c) / np.arange(1, len(c) + 1), color=COL[s],
                label=NAMES[s])
    ax.set_xlabel("số câu đã chấm")
    ax.set_ylabel("accuracy luỹ kế (%)")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.15))
    save(fig, "ch7_running")


def main():
    R = load_json("ch7_gsm8k")
    for f in (fig_acc, fig_cost, fig_pareto, fig_errors, fig_latency, fig_only_correct, fig_gate, fig_difficulty,
              fig_tokens, fig_tool, fig_calls, fig_running):
        f(R)


if __name__ == "__main__":
    main()
