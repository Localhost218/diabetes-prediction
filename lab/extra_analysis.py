"""
Phân tích bổ sung cho báo cáo, dùng lại mô hình đã huấn luyện (không huấn luyện lại):
  - Chương 4: sai số dự báo 30 phút theo dải glucose (hạ / trong khoảng / tăng đường huyết)
  - Chương 6: ma trận nhầm lẫn của GAT triển khai, phân tích trọng số chú ý theo nhãn láng giềng và theo bậc

    .venv/bin/python lab/extra_analysis.py
"""

import json

import numpy as np
import torch

from common import MODELS, RESULTS, r4
from ch4_rnn import HORIZON, load, predict_seq, split_patients, windows
from src.models.deep_graph import CLASSES, get_pubmed_classifier
from src.models.rnn_temporal import SeqModel


def ch4_by_range():
    ckpt = torch.load(MODELS / "ch4_glucose_rnn.pt", map_location="cpu")
    model = SeqModel(ckpt["cell"], 64, ckpt["layers"], ckpt["horizon"])
    model.load_state_dict(ckpt["state"])
    df = load()
    df = df[(df.cgm >= 20) & (df.cgm <= 500)]
    test = split_patients(df)["test"]
    X, Y, H = windows(df, sorted(df[df.patient.isin(test)].record.unique()))
    P = predict_seq(model, X, H, ckpt["mu"], ckpt["sd"])
    y, p, last = Y[:, 1], P[:, 1], X[:, -1]
    out = {}
    for name, m in [("< 70 (hạ)", y < 70), ("70–180 (mục tiêu)", (y >= 70) & (y <= 180)), ("> 180 (tăng)", y > 180)]:
        out[name] = {"n": int(m.sum()), "rmse_model": r4(np.sqrt(np.mean((y[m] - p[m]) ** 2))),
                     "rmse_persistence": r4(np.sqrt(np.mean((y[m] - last[m]) ** 2))),
                     "mard_model": r4(np.mean(np.abs(y[m] - p[m]) / y[m]) * 100)}
    # cảnh báo hạ đường huyết: dự báo < 70 sau 30 phút
    hypo, pred_hypo = y < 70, p < 70
    tp = int((hypo & pred_hypo).sum())
    out["hypo_alert"] = {"sensitivity": r4(tp / max(hypo.sum(), 1)), "precision": r4(tp / max(pred_hypo.sum(), 1)),
                         "events": int(hypo.sum())}
    # thay đổi nhanh (> 2 mg/dL/phút trong 30 phút)
    fast = np.abs(y - last) > 60
    out["fast_change"] = {"n": int(fast.sum()), "rmse_model": r4(np.sqrt(np.mean((y[fast] - p[fast]) ** 2))),
                          "rmse_persistence": r4(np.sqrt(np.mean((y[fast] - last[fast]) ** 2)))}
    return {"model": f"{ckpt['cell']} x{ckpt['layers']}", **out}


def ch6_attention():
    clf = get_pubmed_classifier()
    y = clf.y
    pred = clf.prob.argmax(1)
    test = np.array(sorted(clf.test_idx))
    cm = np.zeros((3, 3), int)
    for t, q in zip(y[test], pred[test]):
        cm[t, q] += 1
    mask = clf.src != clf.dst
    s, d, a = clf.src[mask], clf.dst[mask], clf.alpha[mask]
    same = y[s] == y[d]
    deg = np.bincount(d, minlength=len(y))
    # α trung bình chuẩn hóa theo bậc: α·deg(v) = 1 nghĩa là chú ý đều như GCN
    a_norm = a * (deg[d] + 1)
    by_degree = {}
    for lo, hi in [(1, 2), (3, 5), (6, 10), (11, 1000)]:
        m = (deg[test] >= lo) & (deg[test] <= hi)
        by_degree[f"{lo}–{hi if hi < 1000 else '∞'}"] = {"n": int(m.sum()),
                                                         "acc": r4((pred[test][m] == y[test][m]).mean())}
    self_alpha = clf.alpha[~mask]
    return {"confusion": cm.tolist(), "classes": CLASSES, "test_acc": r4((pred[test] == y[test]).mean()),
            "per_class_recall": [r4(cm[i, i] / cm[i].sum()) for i in range(3)],
            "alpha_same_label_norm": r4(a_norm[same].mean()), "alpha_diff_label_norm": r4(a_norm[~same].mean()),
            "share_same_label_edges": r4(same.mean()),
            "self_loop_alpha_mean": r4(self_alpha.mean()), "acc_by_degree": by_degree}


def main():
    res = {"ch4_by_range": ch4_by_range(), "ch6_attention": ch6_attention()}
    (RESULTS / "extra.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
