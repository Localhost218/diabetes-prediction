"""Sinh bảng LaTeX của Chương 2 từ outputs/metrics/ch2_*.json -> Report/TieuLuan 03/generated/ch2_*.tex."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tl3.ch2_models import CLF_NAMES, REG_NAMES  # noqa: E402
from tl3.ch2_text import TEXT_NAMES  # noqa: E402
from tl3.common import load_json  # noqa: E402
from tl3.texfmt import best, num, pct, table, tex_escape  # noqa: E402


def clf_rows(block, names, train=None):
    cols = {k: [block[n][k] for n in names] for k in ("acc", "prec", "rec", "f1", "auc", "ap")}
    fmt = {k: best(v, [num(x, 4) for x in v]) for k, v in cols.items()}
    rows = []
    for i, n in enumerate(names):
        r = [CLF_NAMES[n], num(block[n]["thr"], 2)] + [fmt[k][i] for k in cols]
        if train:
            r.append(num(train[n]["train_s"], 1))
        rows.append(r)
    return rows


def diabetes():
    d = load_json("ch2_diabetes")
    desc = d["survey"]["describe"]
    table("ch2_diab_describe", "lrrrrrr", ["Thuộc tính", "Trung bình", "Độ lệch chuẩn", "Min", "Trung vị", "Max", "Q3"],
          [[tex_escape(c), num(v["mean"], 2), num(v["std"], 2), num(v["min"], 2), num(v["50%"], 2), num(v["max"], 2),
            num(v["75%"], 2)] for c, v in desc.items()])
    o = d["audit_raw"]["outliers"]
    table("ch2_diab_outliers", "lrrrrr", ["Thuộc tính", "Q1", "Q3", "Cận dưới", "Cận trên", "Số ngoại lai"],
          [[tex_escape(c), num(v["q1"], 2), num(v["q3"], 2), num(v["lo"], 2), num(v["hi"], 2), num(v["n"], 0)]
           for c, v in o.items()],
          note="Cận = Q1 − 1,5·IQR và Q3 + 1,5·IQR. Ngoại lai được giữ lại vì là giá trị lâm sàng có thật.")
    sp = d["split"]
    table("ch2_diab_split", "lrrr", ["Tập", "Số mẫu", "Tỷ lệ (\\%)", "Tỷ lệ lớp 1 (\\%)"],
          [[n, num(sp[k], 0), pct(sp[k] / (sp["train"] + sp["val"] + sp["test"]), 1), pct(sp[f"pos_{k}"], 2)]
           for n, k in [("Train", "train"), ("Validation", "val"), ("Test", "test")]])
    names = list(d["val"])
    table("ch2_diab_val", "lrrrrrrrr", ["Mô hình", "Ngưỡng", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC",
                                         "PR-AUC", "Huấn luyện (s)"],
          clf_rows(d["val"], names, d["models"]),
          note="Ngưỡng chọn trên Validation để F1 lớn nhất. Đậm: tốt nhất mỗi cột. SVM huấn luyện trên tập con 20.000 mẫu.")
    rows = []
    for n in names[1:]:
        a, b = d["val05"][n], d["val"][n]
        rows.append([CLF_NAMES[n], num(a["prec"], 4), num(a["rec"], 4), num(a["f1"], 4), num(b["thr"], 2),
                     num(b["prec"], 4), num(b["rec"], 4), num(b["f1"], 4), num(b["f1"] - a["f1"], 4, sign=True)])
    table("ch2_diab_thr", "lrrrrrrrr", ["Mô hình", "P (0,5)", "R (0,5)", "F1 (0,5)", "Ngưỡng*", "P*", "R*", "F1*",
                                         "ΔF1"], rows)
    table("ch2_diab_test", "lrrrrrrr", ["Mô hình", "Ngưỡng", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"],
          clf_rows(d["test"], names))
    cv = d["cv"]
    table("ch2_diab_cv", "lrrr", ["Mô hình", "ROC-AUC", "F1", "Accuracy"],
          [[CLF_NAMES[n]] + [f"{num(cv[n][k]['mean'], 4)} ± {num(cv[n][k]['std'], 4)}"
                             for k in ("test_roc_auc", "test_f1", "test_accuracy")] for n in cv])
    t = d["test"][d["best"]["name"]]
    table("ch2_diab_best_cm", "lrr", ["", "Dự đoán 0", "Dự đoán 1"],
          [["Thật 0 (không)", num(t["tn"], 0), num(t["fp"], 0)], ["Thật 1 (tiểu đường)", num(t["fn"], 0), num(t["tp"], 0)]])


def house():
    h = load_json("ch2_house")
    desc = h["survey"]["describe"]
    table("ch2_house_describe", "lrrrrr", ["Thuộc tính", "Trung bình", "Độ lệch chuẩn", "Min", "Trung vị", "Max"],
          [[tex_escape(c), num(v["mean"], 2), num(v["std"], 2), num(v["min"], 0), num(v["50%"], 0), num(v["max"], 0)]
           for c, v in desc.items()])
    for split in ("val", "test"):
        names = list(h[split])
        cols = {k: [h[split][n][k] for n in names] for k in ("mae", "rmse", "r2", "mape")}
        f = {"mae": best(cols["mae"], [num(v, 0) for v in cols["mae"]], False),
             "rmse": best(cols["rmse"], [num(v, 0) for v in cols["rmse"]], False),
             "r2": best(cols["r2"], [num(v, 4) for v in cols["r2"]]),
             "mape": best(cols["mape"], [pct(v, 2) for v in cols["mape"]], False)}
        rows = [[REG_NAMES[n], f["mae"][i], f["rmse"][i], f["r2"][i], f["mape"][i]] +
                ([num(h["models"][n]["train_s"], 2)] if split == "val" else []) for i, n in enumerate(names)]
        head = ["Mô hình", "MAE", "RMSE", "R²", "MAPE (\\%)"] + (["Huấn luyện (s)"] if split == "val" else [])
        table(f"ch2_house_{split}", "l" + "r" * (len(head) - 1), head, rows)
    cv = h["cv"]
    table("ch2_house_cv", "lrrr", ["Mô hình", "R² trung bình", "Độ lệch chuẩn", "RMSE trung bình"],
          [[REG_NAMES[n], num(v["r2_mean"], 4), num(v["r2_std"], 4), num(v["rmse_mean"], 0)] for n, v in cv.items()])
    table("ch2_house_quartile", "lrrr", ["Nhóm giá (Test)", "Từ", "Đến", "MAE"],
          [[f"Phân vị {i + 1}", num(q["lo"], 0), num(q["hi"], 0), num(q["mae"], 0)]
           for i, q in enumerate(h["err_by_quartile"])])


def comments():
    c = load_json("ch2_comments")
    names = list(c["test"])
    for split in ("val", "test"):
        cols = {k: [c[split][n].get(k, 0) for n in names] for k in ("acc", "macro_f1", "f1_neg", "f1_pos", "auc")}
        f = {k: best(v, [num(x, 4) for x in v]) for k, v in cols.items()}
        rows = [[TEXT_NAMES[n]] + [f[k][i] for k in cols] + ([num(c["models"][n]["train_s"], 2)] if split == "val" else [])
                for i, n in enumerate(names)]
        head = ["Mô hình", "Accuracy", "Macro-F1", "F1 lớp 0", "F1 lớp 1", "ROC-AUC"] + \
               (["Huấn luyện (s)"] if split == "val" else [])
        table(f"ch2_cmt_{split}", "l" + "r" * (len(head) - 1), head, rows)
    r = c["rating"]
    table("ch2_cmt_rating", "lrrrrr", ["Rating", "1", "2", "3", "4", "5"],
          [["F1 từng lớp"] + [num(v, 4) for v in r["f1_per_class"]]],
          note=f"Accuracy {num(r['acc'], 4)}; macro-F1 {num(r['macro_f1'], 4)}; sai lệch không quá 1 sao: "
               f"{pct(r['within1'], 2)}\\%.")


def misc():
    s = load_json("ch2_scratch")
    table("ch2_scratch", "llrr", ["Thuật toán", "Chỉ số (Test)", "NumPy tự viết", "scikit-learn"],
          [["Hồi quy tuyến tính (phương trình chuẩn)", "R²", num(s["linreg"]["r2_normal_eq"], 6), num(s["linreg"]["r2_sklearn"], 6)],
           ["Hồi quy tuyến tính (GD)", "R²", num(s["linreg"]["r2_gd"], 6), num(s["linreg"]["r2_sklearn"], 6)],
           ["Hồi quy logistic (mini-batch GD)", "ROC-AUC", num(s["logreg"]["auc_numpy"], 6), num(s["logreg"]["auc_sklearn"], 6)],
           ["Hồi quy logistic (mini-batch GD)", "F1 (ngưỡng 0,5)", num(s["logreg"]["f1_numpy"], 4), num(s["logreg"]["f1_sklearn"], 4)],
           ["Hồi quy logistic", "Thời gian (s)", num(s["logreg"]["time_numpy_s"], 3), num(s["logreg"]["time_sklearn_s"], 3)]])
    u = load_json("ch2_unsup")
    table("ch2_unsup_profiles", "lrrrrrr", ["Cụm", "Số mẫu", "Tuổi", "BMI", "HbA1c", "Glucose", "Tỷ lệ bệnh (\\%)"],
          [[str(p["cluster"]), num(p["n"], 0), num(p["age"], 1), num(p["bmi"], 1), num(p["HbA1c_level"], 2),
            num(p["blood_glucose_level"], 1), pct(p["diabetes_rate"], 2)] for p in u["profiles"]])


if __name__ == "__main__":
    diabetes()
    house()
    comments()
    misc()
    print("ok")
