"""Chụp giao diện các phân hệ Web App cho báo cáo bằng Playwright (Chromium headless).

Chạy app trước (python app/app.py), rồi:  python app/screenshots.py
Ảnh ghi vào Report/TieuLuan 01/assets/ui/; kịch bản dừng nếu trang có lỗi JavaScript.
Khung nhìn rộng 940 px: ảnh co về bề rộng trang A4 ít hơn nên chữ trên ảnh đủ lớn để đọc.
"""
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

URL = "http://127.0.0.1:8090/"
OUT = Path(__file__).resolve().parents[3] / "Report" / "TieuLuan 01" / "assets" / "ui"
WAIT = 1200
NEG = "Cheap material, see-through and the seams came apart after one wash. Very disappointed, returning it."


def shot(page, tab, button, name, before=None):
    page.click(f'.tab[data-tab="{tab}"]')
    page.wait_for_timeout(300)
    if before:
        before(page)
    page.click(button)
    page.wait_for_timeout(WAIT)
    page.locator(f"#{tab}").screenshot(path=OUT / f"{name}.png")
    print("->", name)


def launch(p):
    try:
        return p.chromium.launch()
    except Exception:                      # bản trình duyệt khớp phiên bản playwright chưa tải: dùng bản có sẵn
        exe = sorted(Path.home().glob("AppData/Local/ms-playwright/chromium_headless_shell-*/*/chrome-headless-shell.exe"))
        if not exe:
            raise
        return p.chromium.launch(executable_path=str(exe[-1]))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = launch(p)
        page = browser.new_page(viewport={"width": 940, "height": 820}, device_scale_factor=2)
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.on("console", lambda m: m.type == "error" and errors.append(m.text))
        page.goto(URL)
        page.wait_for_timeout(500)
        page.click("#d-run")
        page.wait_for_timeout(WAIT)
        page.screenshot(path=OUT / "ui_overview.png", clip={"x": 0, "y": 0, "width": 940, "height": 600})
        print("-> ui_overview")
        shot(page, "diabetes", "#d-run", "ui_diabetes")
        shot(page, "house", "#h-run", "ui_house")
        shot(page, "sentiment", "#s-run", "ui_sentiment")
        shot(page, "sentiment", "#s-run", "ui_sentiment_neg", lambda pg: pg.fill("#s-text", NEG))
        shot(page, "vision", "#v-run", "ui_vision")

        def svhn(pg):
            pg.select_option("#v-ds", "svhn")
            pg.select_option("#v-arch", "mobilenetv2")
            pg.fill("#v-idx", "42")
        shot(page, "vision", "#v-run", "ui_vision_svhn", svhn)

        def days(n, cell=None):
            def f(pg):
                if cell:
                    pg.click(f'#st-cell button[data-c="{cell}"]')
                pg.fill("#st-days", str(n))
                pg.dispatch_event("#st-days", "input")
            return f
        shot(page, "stock", "#st-run", "ui_stock", days(160, "lstm"))
        shot(page, "stock", "#st-run", "ui_stock_gru", days(700, "gru"))
        shot(page, "churn", "#c-run", "ui_churn", lambda pg: pg.click('#c-group button[data-g="churn"]'))
        shot(page, "churn", "#c-run", "ui_churn_loyal", lambda pg: pg.click('#c-group button[data-g="loyal"]'))
        browser.close()
    if errors:
        sys.exit(f"trang có lỗi JavaScript: {errors}")


if __name__ == "__main__":
    main()
