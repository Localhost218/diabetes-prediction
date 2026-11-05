"""Bảng kết quả Chương 6 -> Report/TieuLuan 03/generated/ch6_*.tex"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tl3.common import load_json  # noqa: E402
from tl3.texfmt import best, num, pct, table  # noqa: E402

NAMES = {"logreg": "Hồi quy logistic", "mlp": "MLP 2 lớp", "gcn": "GCN", "sage": "GraphSAGE-mean", "gat": "GAT (8 đầu)"}
GRAPH = {"logreg": "không", "mlp": "không", "gcn": "có", "sage": "có", "gat": "có"}
DS = ["Cora", "CiteSeer"]


def t_datasets(d):
    keys = [("nodes", "Số nút", 0), ("edges", "Số cạnh (vô hướng)", 0), ("features", "Số đặc trưng (từ)", 0),
            ("classes", "Số lớp", 0), ("train", "Nút train", 0), ("val", "Nút validation", 0), ("test", "Nút test", 0),
            ("deg_mean", "Bậc trung bình", 2), ("deg_median", "Bậc trung vị", 0), ("deg_max", "Bậc lớn nhất", 0),
            ("isolated", "Nút cô lập", 0), ("feat_nnz_mean", "Số từ khác 0 / nút", 1), ("homophily", "Edge homophily", 3)]
    rows = [[lab] + [num(d[ds][k], dec) for ds in DS] for k, lab, dec in keys]
    table("ch6_datasets", "lrr", ["Thuộc tính", "Cora", "CiteSeer"], rows,
          note="Homophily: tỷ lệ cạnh nối hai nút cùng nhãn. Cạnh đếm sau đối xứng hoá, không tính khuyên.")


def t_node(n):
    rows = []
    for ds in DS:
        m = [n[ds][k]["test_mean"] for k in NAMES]
        cells = best(m, [f"{pct(n[ds][k]['test_mean'])} $\\pm$ {pct(n[ds][k]['test_std'])}" for k in NAMES])
        for k, c in zip(NAMES, cells):
            r = n[ds][k]
            rows.append([ds if k == "logreg" else "", NAMES[k], GRAPH[k], c, pct(r["val_mean"]), num(r["params"], 0),
                         num(r["seconds"], 2)])
        if ds == "Cora":
            rows.append("\\midrule")
    table("ch6_node", "llcrrrr", ["Dữ liệu", "Mô hình", "Dùng cạnh", "Acc test (\\%)", "Acc val (\\%)", "Tham số",
                                  "Giây / lần"], rows,
          note="Trung bình $\\pm$ độ lệch chuẩn trên 5 seed (0–4), phép chia Planetoid chuẩn; huấn luyện trên CPU 3 luồng.")


def t_depth(dp):
    rows = []
    for r in dp["rows"]:
        rows.append([str(r["layers"]), "có" if r["residual"] else "không",
                     f"{pct(r['test_mean'])} $\\pm$ {pct(r['test_std'])}", num(r["mad"], 4), num(r["mad_neighbors"], 4)])
    table("ch6_depth", "ccrrr", ["Số lớp", "Nối tắt", "Acc test (\\%)", "MAD toàn cục", "MAD cạnh"], rows,
          note="GCN ẩn 64, 3 seed. MAD: khoảng cách cosine trung bình giữa biểu diễn ẩn cuối (toàn bộ cặp nút / chỉ cặp kề).")


def t_perturb(p):
    # bỏ 100% cạnh thì không còn cạnh nào để đo homophily (JSON ghi NaN): in "--" thay vì "nan"
    rows = [["bỏ cạnh" if r["mode"] == "drop" else "nối lại", pct(r["p"], 0),
             f"{pct(r['test_mean'])} $\\pm$ {pct(r['test_std'])}",
             "--" if r["homophily"] is None or math.isnan(r["homophily"]) else num(r["homophily"], 3)]
            for r in p["rows"]]
    table("ch6_perturb", "lrrr", ["Can thiệp", "Tỷ lệ (\\%)", "Acc GCN (\\%)", "Homophily"], rows,
          note="Nối lại: đầu cuối của cạnh bị chọn được thay bằng một nút ngẫu nhiên, giữ nguyên số cạnh. "
               "Bỏ 100\\% cạnh: không còn cạnh nên homophily không xác định (--).")


def t_link_pyg(link, extra, node):
    rows = [["GAE (GCN 2 lớp, tích vô hướng)", "AUC test", f"{pct(link['test_auc_mean'])} $\\pm$ {pct(link['test_auc_std'])}"],
            ["", "AP test", f"{pct(link['test_ap_mean'])} $\\pm$ {pct(link['test_ap_std'])}"], "\\midrule"]
    for k, lab in (("gcn", "GCN"), ("gat", "GAT")):
        rows.append([f"{lab} tự cài (5 seed)", "Acc Cora", f"{pct(node['Cora'][k]['test_mean'])} $\\pm$ {pct(node['Cora'][k]['test_std'])}"])
        rows.append([f"{lab} của PyG (3 seed)", "Acc Cora", f"{pct(extra['pyg'][k]['test_mean'])} $\\pm$ {pct(extra['pyg'][k]['test_std'])}"])
    table("ch6_link_pyg", "llr", ["Mô hình", "Chỉ số", "Giá trị (\\%)"], rows,
          note=f"Dự đoán liên kết: {num(link['train_edges'], 0)} cạnh train, {num(link['val_edges'], 0)} val, "
               f"{num(link['test_edges'], 0)} test, mẫu âm cùng số lượng.")


def main():
    t_datasets(load_json("ch6_data"))
    n = load_json("ch6_node")
    t_node(n)
    t_depth(load_json("ch6_depth"))
    t_perturb(load_json("ch6_perturb"))
    t_link_pyg(load_json("ch6_link"), load_json("ch6_extra"), n)


if __name__ == "__main__":
    main()
