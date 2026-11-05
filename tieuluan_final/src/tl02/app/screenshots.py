"""Chụp giao diện Web App cho báo cáo bằng Playwright (Chromium headless).

Chạy app trước (python app/app.py), rồi:  python app/screenshots.py
Ảnh ghi vào Report/TieuLuan 02/assets/ui/; kịch bản dừng nếu trang có lỗi JavaScript.
"""
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

URL = "http://127.0.0.1:8092/"
OUT = Path(__file__).resolve().parents[3] / "Report" / "TieuLuan 02" / "assets" / "ui"
AGENT_WAIT = 60000


def launch(p):
    try:
        return p.chromium.launch()
    except Exception:                      # bản trình duyệt khớp phiên bản playwright chưa tải: dùng bản có sẵn
        exe = sorted(Path.home().glob("AppData/Local/ms-playwright/chromium_headless_shell-*/*/chrome-headless-shell.exe"))
        if not exe:
            raise
        return p.chromium.launch(executable_path=str(exe[-1]))


def run_agent(page, pane, button, name, question=None):
    page.click(f'.tab[data-tab="{pane}"]')
    page.wait_for_timeout(300)
    if question:
        page.fill("#h-q" if pane == "health" else "#f-q", question)
    page.eval_on_selector(f"#{pane[0]}-ans", "e => e.textContent = ''")   # lượt chạy trước còn câu trả lời cũ
    page.click(button)
    page.wait_for_selector(f"#{pane[0]}-ans:not(:empty)", timeout=AGENT_WAIT)
    page.wait_for_selector(f"#{pane[0]}-trace li.muted", state="detached", timeout=AGENT_WAIT)
    page.wait_for_timeout(600)
    page.locator(f"#{pane}").screenshot(path=OUT / f"{name}.png")
    print("->", name)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = launch(p)
        page = browser.new_page(viewport={"width": 1100, "height": 900}, device_scale_factor=2)
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.on("console", lambda m: m.type == "error" and errors.append(m.text))
        page.goto(URL)
        page.wait_for_timeout(800)
        run_agent(page, "health", "#h-run", "ui_health")
        page.screenshot(path=OUT / "ui_overview.png", clip={"x": 0, "y": 0, "width": 1100, "height": 640})
        print("-> ui_overview")
        page.select_option("#h-useprof", "0")
        run_agent(page, "health", "#h-run", "ui_health_food", "Người tiểu đường ăn cơm trắng thế nào cho đúng?")
        run_agent(page, "finance", "#f-run", "ui_finance")
        run_agent(page, "finance", "#f-run", "ui_finance_forecast", "Dự báo giá cổ phiếu FPT phiên tới?")
        for kg in ("health", "finance"):
            page.click('.tab[data-tab="kg"]')
            page.select_option("#kg-name", kg)
            page.wait_for_timeout(1200)
            page.locator("#kg").screenshot(path=OUT / f"ui_kg_{kg}.png")
            print("-> ui_kg_" + kg)
        page.click('.tab[data-tab="rag"]')
        page.click("#r-run")
        page.wait_for_selector("#r-out .col", timeout=AGENT_WAIT)
        page.wait_for_timeout(500)
        page.locator("#rag").screenshot(path=OUT / "ui_rag.png")
        print("-> ui_rag")
        browser.close()
    if errors:
        sys.exit("Lỗi JavaScript: " + "; ".join(errors))


if __name__ == "__main__":
    main()
