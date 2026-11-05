"""Dự báo phiên tới cho cả 5 tài sản bằng công cụ du_bao_bilstm của tác tử, ghi outputs/metrics/forecast_live.json.

Tách riêng bất định mô hình (độ lệch chuẩn MC Dropout) với sai số dự báo thực tế (RMSE kiểm thử) để báo cáo
thấy vì sao khoảng dự báo phải cộng thêm phần dư.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tl2.common import device, save_json  # noqa: E402
from tl2.dl_models import Forecaster  # noqa: E402

TARGETS = ["FPT", "HPG", "VCB", "VIC", "GOLD_VND"]


def main():
    dev = device()
    out = {}
    for t in TARGETS:
        f = Forecaster(t, dev)
        r = f.predict()
        r["rmse_pct"] = 100 * f.meta["test"]["rmse_ret"]
        r["width_pct"] = 100 * (r["interval90"][1] - r["interval90"][0]) / r["last_price"]
        out[t] = r
        print(t, r["last_date"], round(r["pred_return_pct"], 3), round(r["mc_std_pct"], 3), round(r["prob_up"], 3))
    save_json("forecast_live", out)


if __name__ == "__main__":
    main()
