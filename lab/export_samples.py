"""
Xuất một ít mẫu của tập KIỂM TRA để trang /lab có dữ liệu thật cho người dùng thử (không cần bộ dữ liệu gốc):
  data/models/ch3_samples.npz  — 2 ảnh đáy mắt mỗi mức DR (128×128) + nhãn
  data/models/ch4_samples.json — 6 đoạn CGM 24 giờ của bệnh nhân thuộc tập kiểm tra

    .venv/bin/python lab/export_samples.py
"""

import json

import numpy as np

from ch4_rnn import load, split_patients
from common import DATA, MODELS


def main():
    r = np.load(DATA / "retina" / "retinamnist_128.npz", allow_pickle=False)
    y = r["test_labels"].ravel()
    idx = np.concatenate([np.where(y == c)[0][:2] for c in range(5)])
    np.savez_compressed(MODELS / "ch3_samples.npz", images=r["test_images"][idx], labels=y[idx])

    df = load()
    df = df[(df.cgm >= 20) & (df.cgm <= 500)]
    test = split_patients(df)["test"]
    out = []
    for rec in sorted(df[df.patient.isin(test)].record.unique())[:6]:
        d = df[df.record == rec].iloc[200:200 + 96]
        out.append({"record": rec, "type": d.type.iloc[0], "start": str(d.time.iloc[0]),
                    "glucose": d.cgm.round(1).tolist(), "hours": (d.time.dt.hour + d.time.dt.minute / 60).round(2).tolist()})
    (MODELS / "ch4_samples.json").write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print("đã xuất", len(idx), "ảnh và", len(out), "đoạn CGM")


if __name__ == "__main__":
    main()
