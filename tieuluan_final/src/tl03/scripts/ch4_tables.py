"""Sinh các bảng LaTeX của Chương 4 vào Report/TieuLuan 03/generated/ch4_*.tex từ outputs/metrics/ch4_*.json."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tl3.common import load_json  # noqa: E402
from tl3.texfmt import best, num, pct, sci, table  # noqa: E402

ARCHS = ("rnn", "lstm", "gru", "bilstm")
AN = {"rnn": "SimpleRNN", "lstm": "LSTM", "gru": "GRU", "bilstm": "BiLSTM"}
FW = {"torch": "PyTorch", "keras": "Keras"}
EXPN = {"ecommerce": "E-commerce", "aapl": "AAPL", "gold": "Vàng GLD"}
DEC = {"ecommerce": 0, "aapl": 3, "gold": 3}


def datasets():
    e = load_json("ch4_eda")
    s = e["shapes"]
    rows = [["E-commerce (TR)", f"{e['ecom']['start']} → {e['ecom']['end']}", num(e["ecom"]["days"], 0) + " ngày",
             "9", "Total\\_Amount", "TRY"],
            ["AAPL", f"{e['aapl']['start']} → {e['aapl']['end']}", num(e["aapl"]["rows"], 0) + " phiên", "6",
             "close", "USD"],
            ["Vàng (FINAL\\_USO)", f"{e['gold']['start']} → {e['gold']['end']}", num(e["gold"]["rows"], 0) + " phiên",
             "13", "GLD\\_Close", "USD"]]
    table("ch4_datasets", "llrrll", ["Bộ dữ liệu", "Khoảng thời gian", "Số bước", "Đặc trưng", "Mục tiêu", "Đơn vị"],
          rows)
    rows = []
    for k, n in (("ecommerce", "E-commerce"), ("aapl", "AAPL"), ("gold", "Vàng GLD")):
        sh = s[k]
        rows.append([n] + [f"({num(sh[p][0], 0)}, {sh[p][1]}, {sh[p][2]})" for p in ("train", "val", "test")])
    table("ch4_shapes", "lrrr", ["Thực nghiệm", "X train", "X validation", "X test"], rows,
          note="Mỗi mẫu là một tensor (số cửa sổ, 30 bước, số đặc trưng); chia 70/15/15 theo thứ tự thời gian.")


def results(exp):
    mt, mk = load_json(f"ch4_{exp}_torch"), load_json(f"ch4_{exp}_keras")
    summ = load_json("ch4_summary")[exp]
    d = DEC[exp]
    entries = [("Naive (hôm qua)", "--", mt["naive"]["test"], None, None)]
    if exp == "ecommerce":
        entries += [("TB trượt 7 ngày", "--", summ["ma7"], None, None),
                    ("TB trượt 30 ngày", "--", summ["ma30"], None, None)]
    for fw, m in (("torch", mt), ("keras", mk)):
        for a in ARCHS:
            entries.append((AN[a], FW[fw], m[a]["test"], m[a]["epochs"], m[a]["sec_per_epoch"]))
    cols = {k: [e[2][k] for e in entries] for k in ("mae", "rmse", "mape", "r2")}
    f = {"mae": best(cols["mae"], [num(v, d) for v in cols["mae"]], False),
         "rmse": best(cols["rmse"], [num(v, d) for v in cols["rmse"]], False),
         "mape": best(cols["mape"], [num(v, 2) for v in cols["mape"]], False),
         "r2": best(cols["r2"], [num(v, 4) for v in cols["r2"]], True)}
    rows = []
    for i, (name, fw, m, ep, spe) in enumerate(entries):
        if i == (3 if exp == "ecommerce" else 1) or (fw == "Keras" and entries[i - 1][1] == "PyTorch"):
            rows.append("\\midrule")
        da = pct(m["dir_acc"], 1) if m.get("dir_acc") and name.startswith(("Simple", "LSTM", "GRU", "BiLSTM")) else "--"
        rows.append([name, fw, f["mae"][i], f["rmse"][i], f["mape"][i], f["r2"][i], da,
                     num(ep, 0) if ep else "--", num(spe, 2) if spe else "--"])
    table(f"ch4_{exp}_results", "llrrrrrrr",
          ["Mô hình", "Framework", "MAE", "RMSE", "MAPE (\\%)", "R²", "Đúng hướng (\\%)", "Epoch", "Giây/epoch"], rows,
          note="Chỉ số tính trên tập test, đơn vị gốc của biến mục tiêu; in đậm là giá trị tốt nhất của cột. "
               "Đúng hướng: tỉ lệ ngày mô hình đoán đúng chiều tăng/giảm so với hôm trước.")


def fw_compare():
    rows = []
    for exp in ("ecommerce", "aapl", "gold"):
        mt, mk = load_json(f"ch4_{exp}_torch"), load_json(f"ch4_{exp}_keras")
        d = DEC[exp]
        for a in ARCHS:
            t, k = mt[a]["test"]["rmse"], mk[a]["test"]["rmse"]
            rows.append([EXPN[exp] if a == "rnn" else "", AN[a], num(t, d), num(k, d), num(100 * (k / t - 1), 1, True),
                         "PyTorch" if t < k else "Keras"])
        if exp != "gold":
            rows.append("\\midrule")
    table("ch4_fw_compare", "llrrrl", ["Thực nghiệm", "Kiến trúc", "RMSE PyTorch", "RMSE Keras", "Keras so với PyTorch (\\%)",
                                        "Tốt hơn"], rows)


def params():
    mt, mk = load_json("ch4_aapl_torch"), load_json("ch4_aapl_keras")
    rows = [[AN[a], num(mt[a]["params"], 0), num(mk[a]["params"], 0), num(mt[a]["sec_per_epoch"], 2),
             num(mk[a]["sec_per_epoch"], 2)] for a in ARCHS]
    table("ch4_params", "lrrrr", ["Kiến trúc", "Tham số PyTorch", "Tham số Keras", "Giây/epoch PyTorch (GPU)",
                                  "Giây/epoch Keras (CPU)"], rows,
          note="Thực nghiệm AAPL: 6 đặc trưng, ẩn 64, cửa sổ 30, batch 64. PyTorch có thêm một vector bias $b_{hh}$ mỗi cổng "
               "nên nhiều tham số hơn Keras ở cùng cấu hình.")


def extras():
    ex = load_json("ch4_extras")
    rows = []
    for exp in ("aapl", "gold"):
        m = ex["minmax"][exp]
        for k, lab in (("minmax_lstm", "MinMax toàn cục + LSTM"), ("minmax_gru", "MinMax toàn cục + GRU"),
                       ("relative_lstm", "Tương đối + LSTM"), ("relative_gru", "Tương đối + GRU")):
            rows.append([EXPN[exp] if k == "minmax_lstm" else "", lab, num(m[k]["mae"], 3), num(m[k]["rmse"], 3),
                         num(m[k]["r2"], 4)])
        rows.append("\\midrule") if exp == "aapl" else None
    table("ch4_minmax", "llrrr", ["Dữ liệu", "Chuẩn hoá + mô hình", "MAE", "RMSE", "R²"], [r for r in rows if r])
    ws = (10, 20, 30, 60, 90)
    rows = []
    for exp in ("ecommerce", "aapl", "gold"):
        for a in ("lstm", "gru"):
            vals = [ex["window"][exp][f"{a}_w{w}"]["ratio"] for w in ws]
            rows.append([EXPN[exp] if a == "lstm" else "", AN[a]] + best(vals, [num(v, 3) for v in vals], False))
    table("ch4_window", "llrrrrr", ["Thực nghiệm", "Mô hình"] + [f"w = {w}" for w in ws], rows,
          note="Giá trị là RMSE test / RMSE naive (càng nhỏ càng tốt; < 1 là thắng dự báo ngây thơ).")
    rows = []
    for exp in ("ecommerce", "aapl", "gold"):
        h = ex["horizon"][exp]
        d = DEC[exp]
        rows.append([EXPN[exp], "LSTM đệ quy"] + [num(h["rmse_h"][i], d) for i in (0, 2, 4, 9)])
        rows.append(["", "Naive"] + [num(h["naive_h"][i], d) for i in (0, 2, 4, 9)])
    table("ch4_horizon", "llrrrr", ["Thực nghiệm", "Phương pháp", "h = 1", "h = 3", "h = 5", "h = 10"], rows,
          note="RMSE theo tầm dự báo h trên các gốc thời gian cách 3 bước trong tập test.")
    g = ex["gradient"]["summary"]
    rows = []
    for k, lab in (("rnn", "SimpleRNN (tanh)"), ("lstm", "LSTM"), ("lstm_fb3", "LSTM, bias cổng quên = 3"),
                   ("gru", "GRU"), ("lin_rho0.5", "RNN-ReLU, ρ = 0,5"),
                   ("lin_rho0.9", "RNN-ReLU, ρ = 0,9"), ("lin_rho1.1", "RNN-ReLU, ρ = 1,1"),
                   ("lin_rho1.5", "RNN-ReLU, ρ = 1,5")):
        rows.append([lab] + [sci(g[k][c]) if (g[k][c] < 1e-3 or g[k][c] > 1e4) else num(g[k][c], 4)
                             for c in ("k0", "k20", "k50", "k99")])
    table("ch4_gradflow", "lrrrr", ["Mạng", "T − t = 0", "T − t = 20", "T − t = 50", "T − t = 99"], rows,
          note="Chuẩn $\\|\\partial h_T/\\partial x_t\\|$ trung bình trên 20 khởi tạo, chuỗi 100 bước, ẩn 64.")
    s = load_json("ch4_scratch")
    rows = [[f"\\texttt{{{k}}}", sci(v)] for k, v in s["grad_rel_err"].items()]
    table("ch4_gradcheck", "lr", ["Tham số", "Sai số tương đối lớn nhất"], rows)


if __name__ == "__main__":
    datasets()
    for e in ("ecommerce", "aapl", "gold"):
        results(e)
    fw_compare()
    params()
    extras()
    print("ok")
