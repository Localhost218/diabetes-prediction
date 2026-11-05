"""Tải lại dữ liệu thị trường của Tiểu luận 02 từ Yahoo Finance vào data/market/.

Bốn cổ phiếu VN30 dùng cùng mã và khoảng ngày với Tiểu luận 01 (tl/data_seq.py); bốn chuỗi vĩ mô tải từ 2015.
Yahoo có thể điều chỉnh lại giá quá khứ, nên chạy lại sẽ cho số liệu lệch nhẹ so với báo cáo; tệp CSV đi kèm kho
mã là bản đã dùng để ra mọi con số trong báo cáo.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tl2.common import DATA  # noqa: E402

OUT = DATA / "market"
STOCKS = {f"{t}.VN": t for t in ("FPT", "VCB", "HPG", "VIC")}
MACRO = {"GC=F": "XAU", "VND=X": "USDVND", "DX-Y.NYB": "DXY", "E1VFVN30.VN": "VN30ETF"}
STOCK_RANGE, MACRO_RANGE = ("2012-01-01", "2026-10-01"), ("2015-01-01", "2026-10-22")


def fetch(ticker, start, end, tries=3):
    import yfinance as yf
    for k in range(tries):
        df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
        if len(df):
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
            return df
        time.sleep(5 * (k + 1))
    raise RuntimeError(f"yfinance không trả dữ liệu cho {ticker} sau {tries} lần thử")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for ticker, name in STOCKS.items():
        df = fetch(ticker, *STOCK_RANGE)[["Open", "High", "Low", "Close", "Volume"]]
        df = df[df.Volume > 0]                       # bỏ ngày nghỉ lễ mà Yahoo vẫn ghi giá đứng yên
        df.to_csv(OUT / f"{name}.csv")
        print(name, len(df), df.index[0].date(), df.index[-1].date())
    for ticker, name in MACRO.items():
        df = fetch(ticker, *MACRO_RANGE)
        df.to_csv(OUT / f"{name}.csv")
        print(name, len(df), df.index[0].date(), df.index[-1].date())


if __name__ == "__main__":
    main()
