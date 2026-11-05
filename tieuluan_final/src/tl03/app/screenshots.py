"""Chụp giao diện Web App cho báo cáo bằng Playwright (Chromium headless).

    python app/screenshots.py                       # tự bật app ở cổng 8093, chụp mọi exp_id có serve thật
    python app/screenshots.py --only ch3_mnist ch5_stock
    python app/screenshots.py --mock --out <thư mục> # thử giao diện với app/mock_serve.py (không ghi assets/ui)

Mỗi exp_id: điền ví dụ trong SPECS, bấm "Dự đoán", chờ kết quả rồi chụp
  <exp_id>_web.png     khung desktop rộng 1100, cắt sát khối nội dung (chừa 16 px), tỉ lệ điểm ảnh 2
  <exp_id>_mobile.png  khung điện thoại 390x844, một màn hình, cuộn tới vùng kết quả, tỉ lệ 2
Trang chủ: ch0_home_web.png, ch0_home_mobile.png. --no-mobile chỉ chụp lại ảnh desktop. App bị tắt sau khi chụp xong.
"""
import argparse
import contextlib
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

APP = Path(__file__).resolve().parent
OUT = APP.parents[2] / "Report" / "TieuLuan 03" / "assets" / "ui"
PREDICT_WAIT = 600_000          # chatbot/tác tử có thể nạp LLM lần đầu
# 1100 px thay vì 1440: ảnh đặt vừa bề rộng trang PDF thì chữ giao diện to hơn ~1,3 lần
DESK = {"width": 1100, "height": 800}
# trang chủ rộng hơn để 7 thẻ chương xếp 4 cột (2 hàng): ở 1100 px chỉ 3 cột, ảnh cao gấp ~2 bề rộng, vượt trang A4
HOME_DESK = {"width": 1280, "height": 800}
MOBILE = {"width": 390, "height": 844}
PAD = 16                        # lề giữ lại quanh khối nội dung khi cắt ảnh desktop
# bộ nhớ commit còn trống (GB): cần đủ mới bật app, tụt dưới ngưỡng dừng thì chờ trước mỗi trang
START_FREE_COMMIT_GB, MIN_FREE_COMMIT_GB = 3.5, 2.5
COMMIT_WAIT_S = 1800
# nét chữ số "7" (toạ độ 0..1) khi SPECS của MNIST không có ảnh ví dụ
DIGIT_7 = [[[0.22, 0.2], [0.78, 0.2], [0.45, 0.85]], [[0.35, 0.52], [0.68, 0.52]]]


def free_commit_gb():
    """Bộ nhớ commit còn trống (GB) theo GetPerformanceInfo của Windows; None trên hệ khác."""
    if os.name != "nt":
        return None
    import ctypes
    from ctypes import wintypes

    class PerfInfo(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD)] + [(n, ctypes.c_size_t) for n in (
            "CommitTotal", "CommitLimit", "CommitPeak", "PhysicalTotal", "PhysicalAvailable", "SystemCache",
            "KernelTotal", "KernelPaged", "KernelNonpaged", "PageSize")] + [
            (n, wintypes.DWORD) for n in ("HandleCount", "ProcessCount", "ThreadCount")]

    pi = PerfInfo(cb=ctypes.sizeof(PerfInfo))
    if not ctypes.windll.kernel32.K32GetPerformanceInfo(ctypes.byref(pi), pi.cb):
        return None
    return (pi.CommitLimit - pi.CommitTotal) * pi.PageSize / 2 ** 30


def wait_for_commit(min_gb, what):
    """Chờ tới khi bộ nhớ commit trống >= min_gb; quá COMMIT_WAIT_S thì báo lỗi thay vì chạy liều."""
    t0 = time.time()
    while (free := free_commit_gb()) is not None and free < min_gb:
        if time.time() - t0 > COMMIT_WAIT_S:
            raise RuntimeError(f"commit trống {free:.1f} GB < {min_gb} GB quá {COMMIT_WAIT_S}s ({what})")
        print(f".. tạm dừng trước {what}: commit trống {free:.1f} GB < {min_gb} GB", flush=True)
        time.sleep(30)
    return free


def launch(p):
    try:
        return p.chromium.launch()
    except Exception:                      # bản trình duyệt khớp phiên bản playwright chưa tải: dùng bản có sẵn
        exe = sorted(Path.home().glob("AppData/Local/ms-playwright/chromium_headless_shell-*/*/chrome-headless-shell.exe"))
        if not exe:
            raise
        return p.chromium.launch(executable_path=str(exe[-1]))


def start_app(port, mock):
    env = {**os.environ, "TL3_PORT": str(port), "PYTHONIOENCODING": "utf-8"}
    if mock:
        env["TL3_MOCK"] = "1"
    log = open(APP.parent / "outputs" / "logs" / f"app_{'mock' if mock else 'real'}.log", "w", encoding="utf-8")
    proc = subprocess.Popen([sys.executable, str(APP / "app.py")], cwd=APP.parent, env=env, stdout=log, stderr=log)
    url = f"http://127.0.0.1:{port}/"
    for _ in range(120):
        try:
            urllib.request.urlopen(url + "api/exps", timeout=60)
            return proc, url
        except Exception:
            time.sleep(1)
    proc.terminate()
    raise RuntimeError("App không khởi động được, xem outputs/logs/app_*.log")


def predict_and_wait(page, exp_id):
    page.evaluate("window.fillExample()")
    page.wait_for_timeout(400)
    if "mnist" in exp_id and page.locator("canvas").count():
        has_img = page.evaluate("[...document.querySelectorAll('.imgbox img.preview')].some(i => i.src)")
        if not has_img:
            page.evaluate(f"window.drawOnCanvas({json.dumps(DIGIT_7)})")
            page.wait_for_timeout(200)
    page.click("#run")
    page.wait_for_selector("#result[data-done]", timeout=PREDICT_WAIT)
    page.wait_for_timeout(900)            # Chart.js và thanh xác suất
    return page.get_attribute("#result", "data-done")


def shoot_desktop(page, path):
    """Chụp từ đầu trang tới hết nội dung chính, cắt sát khối thanh tiêu đề + nội dung (chừa PAD px), bỏ chân
    trang và khoảng nền trống. Thanh tiêu đề sticky phải thành tĩnh và trang cuộn về đầu, nếu không ảnh
    full_page vẽ thanh tiêu đề đè lên giữa nội dung."""
    page.add_style_tag(content=".topbar{position:static !important}")
    page.evaluate("window.scrollTo(0, 0)")
    page.wait_for_timeout(200)
    left, right, bottom = page.evaluate("""() => {
      const r = [...document.querySelectorAll('.topbar > *, main.wrap > *')]
        .map((e) => e.getBoundingClientRect()).filter((b) => b.width && b.height);
      return [Math.min(...r.map((b) => b.left)), Math.max(...r.map((b) => b.right)),
              Math.max(...r.map((b) => b.bottom)) + window.scrollY];
    }""")
    x0, x1 = max(0, left - PAD), min(page.viewport_size["width"], right + PAD)
    page.screenshot(path=path, full_page=True, clip={"x": x0, "y": 0, "width": x1 - x0, "height": bottom + PAD})


def shoot(browser, url, exp_id, out, mobile=True):
    """Chụp điện thoại trước (lần gọi đầu gánh thời gian nạp dữ liệu/mô hình) rồi mới desktop, để ảnh desktop
    ghi độ trễ suy luận thật mà không tốn thêm lượt LLM."""
    errors = []
    state = None
    if mobile:
        mob = browser.new_page(viewport=MOBILE, device_scale_factor=2, is_mobile=True, has_touch=True)
        mob.on("pageerror", lambda e: errors.append(str(e)))
        mob.goto(f"{url}exp/{exp_id}")
        state = predict_and_wait(mob, exp_id)
        mob.evaluate("document.getElementById('result-pane').scrollIntoView(); window.scrollBy(0, -14)")
        mob.wait_for_timeout(400)
        mob.screenshot(path=out / f"{exp_id}_mobile.png")
        mob.close()
        if state != "1":
            return state, errors

    desk = browser.new_page(viewport=DESK, device_scale_factor=2)
    desk.on("pageerror", lambda e: errors.append(str(e)))
    desk.goto(f"{url}exp/{exp_id}")
    warm = state == "1"
    state = predict_and_wait(desk, exp_id)
    # chưa có lần gọi nào trước (--no-mobile): gọi lại để bỏ thời gian nạp; trừ chat, mỗi lượt LLM tốn GPU
    if not warm and state == "1" and desk.evaluate("window.EXP.spec.kind") != "chat":
        desk.evaluate("document.getElementById('result').removeAttribute('data-done')")
        state = predict_and_wait(desk, exp_id)
    shoot_desktop(desk, out / f"{exp_id}_web.png")
    desk.close()
    return state, errors


def shoot_home(browser, url, out, mobile=True):
    desk = browser.new_page(viewport=HOME_DESK, device_scale_factor=2)
    desk.goto(url)
    desk.wait_for_timeout(800)
    shoot_desktop(desk, out / "ch0_home_web.png")
    desk.close()
    if not mobile:
        return
    mob = browser.new_page(viewport=MOBILE, device_scale_factor=2, is_mobile=True, has_touch=True)
    mob.goto(url)
    mob.wait_for_timeout(800)
    mob.screenshot(path=out / "ch0_home_mobile.png")
    mob.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--port", type=int, default=8093)
    ap.add_argument("--no-home", action="store_true")
    ap.add_argument("--home-only", action="store_true", help="chỉ chụp trang chủ, không gọi predict")
    ap.add_argument("--no-mobile", action="store_true", help="chỉ chụp ảnh desktop, giữ nguyên ảnh điện thoại")
    a = ap.parse_args()
    if a.mock and a.out is None:
        sys.exit("--mock phải kèm --out (ảnh mock không được ghi vào assets/ui)")
    out = a.out or OUT
    out.mkdir(parents=True, exist_ok=True)
    views = ("web",) if a.no_mobile else ("web", "mobile")
    free = wait_for_commit(START_FREE_COMMIT_GB, "khởi động app")
    print(".. commit trống", "không đo được" if free is None else f"{free:.1f} GB", flush=True)
    summary = {"ok": [], "error": {}, "waiting": {}}
    # khoá GPU (nếu cần) được nhả SAU khi app đã tắt: LLM nằm trong VRAM suốt đời tiến trình Flask
    with contextlib.ExitStack() as gpu_hold:
        proc, url = start_app(a.port, a.mock)
        try:
            run(a, url, out, views, summary, gpu_hold)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()
    print(json.dumps(summary, ensure_ascii=False, indent=1))


def run(a, url, out, views, summary, gpu_hold):
    exps = json.loads(urllib.request.urlopen(url + "api/exps", timeout=600).read())
    summary["waiting"] = exps["waiting"]
    todo = [] if a.home_only else [e for e in exps["ready"] if not a.only or e in a.only]
    # serve chat (LLM cục bộ) không tự giữ tl3.gpu.heavy -> giữ hộ cho cả lượt chụp
    kinds = {e: json.loads(urllib.request.urlopen(f"{url}api/{e}/spec", timeout=60).read()).get("kind") for e in todo}
    if not a.mock and "chat" in kinds.values():
        sys.path.insert(0, str(APP.parent))
        from tl3.gpu import heavy
        gpu_hold.enter_context(heavy("webapp screenshots " + " ".join(e for e, k in kinds.items() if k == "chat")))
    with sync_playwright() as p:
        browser = launch(p)
        if not a.no_home:
            shoot_home(browser, url, out, mobile=not a.no_mobile)
            print("-> ch0_home", flush=True)
        for exp_id in todo:
            try:
                wait_for_commit(MIN_FREE_COMMIT_GB, exp_id)
                state, errs = shoot(browser, url, exp_id, out, mobile=not a.no_mobile)
                if state != "1" or errs:
                    summary["error"][exp_id] = f"kết quả={state} lỗi JS={errs[:2]}"
                    for v in views:     # không để ảnh trang lỗi lẫn vào báo cáo (chỉ ảnh của lần chụp này)
                        (out / f"{exp_id}_{v}.png").unlink(missing_ok=True)
                else:
                    summary["ok"].append(exp_id)
                print("->", exp_id, state, errs[:1], flush=True)
            except Exception as e:
                summary["error"][exp_id] = f"{type(e).__name__}: {str(e)[:200]}"
                print("!!", exp_id, e, flush=True)
        browser.close()


if __name__ == "__main__":
    main()
