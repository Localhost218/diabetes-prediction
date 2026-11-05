"""Sinh bảng LaTeX Chương 3 vào Report/TieuLuan 03/generated/ch3_*.tex từ outputs/metrics/ch3_*.json.

    python scripts/ch3_tables.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tl3.ch3_common import CFG, MODEL_IDS, MODEL_NAME, TF_PATIENCE  # noqa: E402
from tl3.ch3_data import TITLE  # noqa: E402
from tl3.common import METRICS, load_json  # noqa: E402
from tl3.texfmt import best, num, pct, table  # noqa: E402

EXPS = ["mnist", "fashion", "rps", "cifar10"]
SHORT = {"scratch": "NumPy", "torch_basic": "PT Basic", "tf_basic": "TF Basic",
         "torch_improved": "PT Improved", "tf_improved": "TF Improved"}


def metrics(exp):
    p = METRICS / f"ch3_{exp}.json"
    return load_json(f"ch3_{exp}") if p.exists() else {}


def t_data():
    d = load_json("ch3_data")
    rows = []
    for e in EXPS:
        m, c = metrics(e), CFG[e]
        s = m.get("split", {})
        rows.append([TITLE[e], num(s.get("train", 0), 0), num(s.get("val", 0), 0), num(d[e]["test"], 0),
                     f"${d[e]['h']}\\times{d[e]['w']}\\times{d[e]['c']}$", str(d[e]["classes"]),
                     f"{c['basic_size']} / {c['improved_size']}"])
    table("ch3_data_summary", "lrrrccc",
          ["Bộ dữ liệu", "Huấn luyện", "Validation", "Kiểm tra", "Kích thước gốc", "Số lớp", "Ảnh vào B/I"], rows,
          "Validation tách phân tầng 10\\,\\% từ tập huấn luyện gốc (hạt giống cố định). "
          "Ảnh vào B/I: cạnh ảnh đưa vào Basic CNN / Improved CNN.")


def t_cfg():
    rows = []
    for e in EXPS:
        c = CFG[e]
        aug = []
        if c["flip"]:
            aug.append("lật")
        if c["rot"]:
            aug.append(f"xoay {c['rot']}°")
        aug.append(f"dịch {num(100 * c['shift'], 1)}\\,\\%")
        if c["zoom"]:
            aug.append(f"zoom {num(100 * c['zoom'], 0)}\\,\\%")
        if c["color"]:
            aug.append(f"sáng/tương phản {num(100 * c['color'], 0)}\\,\\%")
        rows.append([TITLE[e], str(c["batch"]), str(c["scratch_ep"]), str(c["ep_basic"]), str(c["ep_improved"]),
                     str(c["ep_improved_tf"]), ", ".join(aug)])
    table("ch3_train_cfg", "lcccccp{5.2cm}",
          ["Bộ dữ liệu", "Lô", "Ep. NumPy", "Ep. Basic", "Ep. PT Imp.", "Ep. TF Imp. (tối đa)", "Tăng cường (Improved)"], rows,
          "Basic CNN: Adam $\\eta=10^{-3}$, không tăng cường. Improved CNN: AdamW (weight decay $5\\cdot10^{-4}$), "
          "lịch OneCycle (PyTorch) hoặc cosine có warm-up (Keras), đỉnh $3\\cdot10^{-3}$, label smoothing 0,1. "
          f"TF Improved chạy CPU: số epoch tối đa, dừng sớm sau {TF_PATIENCE} epoch val không cải thiện, khôi phục trọng số tốt nhất.")


def _conv(cin, cout, k=3, bias=True):
    return cin * cout * k * k + (cout if bias else 0)


def t_arch_basic():
    for tag, (h, c, k) in {"gray": (28, 1, 10), "rgb": (32, 3, 10)}.items():
        f = 64 * (h // 4) ** 2
        rows = [["Đầu vào", f"${h}\\times{h}\\times{c}$", "0"],
                ["Conv $3{\\times}3$, 32, pad 1 + ReLU", f"${h}\\times{h}\\times32$", num(_conv(c, 32), 0)],
                ["MaxPool $2{\\times}2$", f"${h // 2}\\times{h // 2}\\times32$", "0"],
                ["Conv $3{\\times}3$, 64, pad 1 + ReLU", f"${h // 2}\\times{h // 2}\\times64$", num(_conv(32, 64), 0)],
                ["MaxPool $2{\\times}2$", f"${h // 4}\\times{h // 4}\\times64$", "0"],
                ["Flatten", f"${num(f, 0)}$", "0"],
                ["Dense 128 + ReLU", "128", num(f * 128 + 128, 0)],
                ["Dense $K$ + Softmax", str(k), num(128 * k + k, 0)],
                "\\midrule",
                ["\\textbf{Tổng}", "", "\\textbf{" + num(_conv(c, 32) + _conv(32, 64) + f * 128 + 128 + 128 * k + k, 0) + "}"]]
        table(f"ch3_arch_basic_{tag}", "llr", ["Tầng", "Kích thước đầu ra", "Số tham số"], rows)


def t_arch_improved():
    import torch  # noqa: F401
    from tl3.ch3_torch import ImprovedCNN
    for e in ("mnist", "cifar10", "rps"):
        c, size = (1 if e == "mnist" else 3), CFG[e]["improved_size"]
        w = CFG[e]["widths"]
        m = ImprovedCNN(c, 10 if e != "rps" else 3, w)
        cnt = lambda mod: sum(p.numel() for p in mod.parameters())
        rows = [["Stem: Conv $3{\\times}3$ + BN + ReLU", f"${size}\\times{size}\\times{w[0]}$", num(cnt(m.stem), 0)]]
        s = size
        for i, wi in enumerate(w):
            blk = m.blocks[3 * i]
            s //= 2
            rows.append([f"ResBlock {i + 1} ({wi} kênh) $\\to$ MaxPool $\\to$ Dropout {num(0.1 + 0.1 * i, 1)}",
                         f"${s}\\times{s}\\times{wi}$", num(cnt(blk), 0)])
        rows += [["Global Average Pooling", str(w[-1]), "0"],
                 ["Dense 256 + ReLU + Dropout 0,4", "256", num(cnt(m.head[2]), 0)],
                 ["Dense $K$ + Softmax", str(10 if e != "rps" else 3), num(cnt(m.head[5]), 0)],
                 "\\midrule", ["\\textbf{Tổng}", "", "\\textbf{" + num(cnt(m), 0) + "}"]]
        table(f"ch3_arch_improved_{e}", "llr", ["Khối", "Kích thước đầu ra", "Số tham số"], rows)


def t_results(e):
    m = metrics(e)
    ms = [k for k in MODEL_IDS if k in m]
    if not ms:
        return
    col = lambda key: [m[k][key] for k in ms]
    acc = best(col("acc"), [pct(v) for v in col("acc")])
    pr = best(col("precision"), [pct(v) for v in col("precision")])
    rc = best(col("recall"), [pct(v) for v in col("recall")])
    f1 = best(col("f1"), [pct(v) for v in col("f1")])
    tt = best(col("train_time"), [num(v, 0) for v in col("train_time")], higher=False)
    inf = best(col("infer_ms"), [num(v, 3) for v in col("infer_ms")], higher=False)
    rows = [[MODEL_NAME[k], acc[i], pr[i], rc[i], f1[i], num(m[k]["params"], 0), f"{m[k]['best_epoch']}/{m[k]['epochs']}",
             tt[i], inf[i]] for i, k in enumerate(ms)]
    table(f"ch3_{e}_results", "lrrrrrcrr",
          ["Mô hình", "Acc (\\%)", "Prec. (\\%)", "Recall (\\%)", "F1 (\\%)", "Tham số", "Ep. tốt/ tổng",
           "Huấn luyện (s)", "Suy luận (ms/ảnh)"], rows,
          "Precision, Recall, F1 là trung bình macro trên tập kiểm tra. Thời gian suy luận đo trên toàn tập kiểm tra "
          "(PyTorch: GPU; NumPy, TensorFlow: CPU). Chữ đậm: tốt nhất cột.")


def t_overview():
    rows = []
    for e in EXPS:
        m = metrics(e)
        vals = [m[k]["acc"] if k in m else None for k in MODEL_IDS]
        got = [v for v in vals if v is not None]
        cells = [("\\textbf{" + pct(v) + "}" if v == max(got) else pct(v)) if v is not None else "—" for v in vals]
        rows.append([TITLE[e], *cells])
    table("ch3_overview", "lccccc", ["Bộ dữ liệu", *[SHORT[k] for k in MODEL_IDS]], rows,
          "Độ chính xác (\\%) trên tập kiểm tra gốc của mỗi bộ dữ liệu.")


def t_gain():
    rows = []
    for e in EXPS:
        m = metrics(e)
        for fw, nm in (("torch", "PyTorch"), ("tf", "TensorFlow")):
            g = m.get(f"gain_{fw}")
            if g:
                rows.append([TITLE[e], nm, pct(m[f"{fw}_basic"]["acc"]), pct(m[f"{fw}_improved"]["acc"]),
                             num(g["acc_pp"], 2, sign=True), num(g["f1_pp"], 2, sign=True), pct(g["err_cut"], 1)])
    if rows:
        table("ch3_gain", "llrrrrr", ["Bộ dữ liệu", "Framework", "Basic (\\%)", "Improved (\\%)", "$\\Delta$Acc (đ.\\%)",
                                       "$\\Delta$F1 (đ.\\%)", "Giảm lỗi (\\%)"], rows,
              "Giảm lỗi = $1-\\text{lỗi}_{\\text{Improved}}/\\text{lỗi}_{\\text{Basic}}$; đ.\\%: điểm phần trăm.")


if __name__ == "__main__":
    t_data()
    t_cfg()
    t_arch_basic()
    t_arch_improved()
    for e in EXPS:
        t_results(e)
    t_overview()
    t_gain()
    print("ok")
