"""Chụp ảnh các trang thực nghiệm của web app (đã điền ví dụ và bấm Chạy) cho mục Demo trên localhost của báo cáo.

    python scripts/shoot_app.py ch2_diabetes ch3_retina ...     # ghi report/figures/app_<exp>.png
    python scripts/shoot_app.py --home                           # trang tổng quan
Cần web app đang chạy ở http://127.0.0.1:8095 và Playwright (dùng Chrome của hệ thống).
"""
import asyncio
import sys
from pathlib import Path

from playwright.async_api import async_playwright

FIG = Path(__file__).resolve().parents[1] / "report" / "figures"
BASE = "http://127.0.0.1:8095"


async def main(args):
    async with async_playwright() as p:
        b = await p.chromium.launch(executable_path="/usr/bin/google-chrome")
        pg = await b.new_page(viewport={"width": 1440, "height": 900}, device_scale_factor=1.5)
        if "--home" in args:
            await pg.goto(BASE + "/")
            await pg.wait_for_timeout(1200)
            await pg.screenshot(path=FIG / "app_home.png")
            print("app_home.png")
        if "--catmouse" in args:
            await pg.goto(BASE + "/catmouse")
            await pg.wait_for_timeout(2500)
            await pg.screenshot(path=FIG / "app_catmouse.png")
            print("app_catmouse.png")
        for exp in [a for a in args if not a.startswith("--")]:
            await pg.goto(f"{BASE}/exp/{exp}")
            await pg.wait_for_timeout(800)
            await pg.click("#fill-example")
            await pg.wait_for_timeout(600)
            await pg.click("#run")
            await pg.wait_for_selector("#result:not(.empty) .verdict, #result .errbox, #result .chat", timeout=240000)
            await pg.wait_for_timeout(1500)
            await pg.screenshot(path=FIG / f"app_{exp}.png", full_page=False)
            print(f"app_{exp}.png")
        await b.close()

asyncio.run(main(sys.argv[1:]))
