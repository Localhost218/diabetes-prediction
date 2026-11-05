"""Sửa một lần các bản ghi TF Improved sinh trước hai bản vá của ch3_train/ch3_keras:

* train_time của lượt tiếp tục từ điểm lưu chỉ đo tiến trình cuối -> thay bằng tổng thời gian các epoch trong lịch sử;
* lượt chạy đủ số epoch tối đa nhưng bị ghi lý do "time_budget" (kiểm tra ngân sách đứng trước kiểm tra epoch cuối).

    python scripts/ch3_fix_tf_results.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tl3.ch3_common import has_result, load_result, save_result, RESULTS  # noqa: E402

import numpy as np  # noqa: E402

for exp in ["mnist", "fashion", "rps", "cifar10"]:
    if not has_result(exp, "tf_improved"):
        continue
    r = load_result(exp, "tf_improved")
    if "stop" not in r:
        continue
    r["train_time"] = float(sum(r["history"]["time"]))
    if r["epochs"] == r["max_epochs"]:
        r["stop"] = "max_epochs"
    save_result(exp, "tf_improved", r, np.load(RESULTS / f"{exp}__tf_improved_proba.npy"))
    print(exp, r["epochs"], r["max_epochs"], r["stop"], round(r["train_time"]))
