"""Phân tích dữ liệu thị trường (CPU) và gắn bằng chứng thực nghiệm vào các cạnh của KG tài chính.

Với mỗi cạnh nhân quả có chuỗi số liệu ở cả hai đầu, đo tương quan giữa thay đổi 20 phiên của biến nguồn và
lợi suất 20 phiên của biến đích, rồi so dấu với dấu lý thuyết của cạnh. Dấu không khớp không làm sai cạnh
(kênh truyền dẫn có thể bị kênh khác lấn át) nhưng phải được tác tử nêu ra.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tl2.common import save_json  # noqa: E402
from tl2.data import gold_frame, market_frame  # noqa: E402
from tl2.kg_finance import SIGN, build_finance_kg  # noqa: E402

COLS = ["FPT", "HPG", "VCB", "VIC", "VN30ETF", "GOLD_VND", "XAU", "USDVND", "DXY"]


def main():
    df = market_frame()
    gold = gold_frame()
    r1 = np.log(df[COLS]).diff().dropna()
    r20 = np.log(df[COLS]).diff(20).dropna()
    out = {"range": [df.index[0].strftime("%Y-%m-%d"), df.index[-1].strftime("%Y-%m-%d")], "n_days": len(df),
           "corr_daily": r1.corr().round(4).to_dict(), "corr_20d": r20.corr().round(4).to_dict(), "cols": COLS}
    # thống kê mô tả lợi suất ngày
    out["describe"] = {c: {"last": float(df[c].iloc[-1]), "ann_ret": float(r1[c].mean() * 252),
                           "ann_vol": float(r1[c].std() * np.sqrt(252)),
                           "max_dd": float((df[c] / df[c].cummax() - 1).min()),
                           "total_ret": float(df[c].iloc[-1] / df[c].iloc[0] - 1)} for c in COLS}
    # tương quan trượt 1 năm của các cặp có trong KG
    pairs = [("USDVND", "FPT"), ("USDVND", "HPG"), ("USDVND", "VIC"), ("XAU", "GOLD_VND"), ("DXY", "XAU"),
             ("USDVND", "VN30ETF")]
    roll = {}
    for a, b in pairs:
        s = r20[a].rolling(252).corr(r20[b]).dropna()
        s = s.iloc[::5]
        roll[f"{a}-{b}"] = {"date": [d.strftime("%Y-%m-%d") for d in s.index], "corr": s.round(4).tolist()}
    out["rolling_corr"] = roll
    # tương quan trễ: thay đổi tỷ giá 20 phiên trước với lợi suất 20 phiên sau của cổ phiếu
    lag = {}
    for b in ("FPT", "HPG", "VCB", "VIC"):
        lag[b] = [float(r20["USDVND"].corr(r20[b].shift(-k))) for k in range(0, 61, 5)]
    out["lag_corr"] = {"lags": list(range(0, 61, 5)), "series": lag}
    # chuỗi giá chuẩn hóa để vẽ (mỗi tuần một điểm)
    w = df[COLS].iloc[::5]
    out["prices"] = {"date": [d.strftime("%Y-%m-%d") for d in w.index], **{c: w[c].round(4).tolist() for c in COLS}}
    gw = gold.iloc[::5]
    out["gold"] = {"date": [d.strftime("%Y-%m-%d") for d in gw.index], "xau": gw["XAU"].round(2).tolist(),
                   "usdvnd": gw["USDVND"].round(1).tolist(), "gold_vnd": gw["GOLD_VND"].round(3).tolist(),
                   "last": {"date": gold.index[-1].strftime("%Y-%m-%d"), "xau": float(gold["XAU"].iloc[-1]),
                            "usdvnd": float(gold["USDVND"].iloc[-1]), "gold_vnd": float(gold["GOLD_VND"].iloc[-1])}}
    # gắn bằng chứng vào KG
    kg = build_finance_kg()
    series = {n: d.get("series") for n, d in kg.G.nodes(data=True)}
    alias = {"GOLD_VND": "GOLD_VND"}
    edges = []
    for s, p, o, d in kg.facts():
        if p not in SIGN or not series.get(s) or not series.get(o):
            continue
        a, b = alias.get(series[s], series[s]), alias.get(series[o], series[o])
        if a not in r20 or b not in r20:
            continue
        c = float(r20[a].corr(r20[b]))
        edges.append({"source": s, "target": o, "relation": p, "theory_sign": SIGN[p], "corr_20d": c,
                      "agree": bool(np.sign(c) == SIGN[p])})
    out["kg_edges_evidence"] = edges
    save_json("market", out)
    for e in edges:
        print(e)


if __name__ == "__main__":
    main()
