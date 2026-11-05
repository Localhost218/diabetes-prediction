"""Web App Tiểu luận 03 — một ứng dụng Flask phục vụ mọi thực nghiệm của bảy chương.

Giao diện không viết riêng cho từng bài: mỗi chương cung cấp `tl3/serve_chN.py` với `SPECS` (mô tả đầu vào, mô hình,
ví dụ) và `predict(exp_id, payload, model_id)`; trang thực nghiệm dựng form từ SPECS và hiển thị kết quả theo `kind`.
Mô-đun serve được nạp lười theo tiền tố `chN_` của exp_id; mô-đun chưa có hoặc lỗi chỉ làm trang đó hiện
"đang cập nhật", không làm sập cả ứng dụng.

    python app/app.py                 # http://127.0.0.1:8093/
    TL3_MOCK=1 python app/app.py      # dùng app/mock_serve.py để thử mọi kiểu giao diện
"""
import base64
import importlib
import io
import math
import os
import sys
import time
import traceback
from pathlib import Path

from flask import Flask, abort, jsonify, render_template, request, send_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

MOCK = os.environ.get("TL3_MOCK") == "1"
PORT = int(os.environ.get("TL3_PORT", "8093"))

# Thông tin trình bày của từng chương (màu, biểu tượng, mô tả); danh sách thực nghiệm lấy từ SPECS của serve.
CHAPTERS = {
    "ch1": {"no": 1, "title": "Lịch sử phát triển AI", "icon": "🏛️", "color": "#64748b",
            "desc": "Từ Turing, Dartmouth, các mùa đông AI đến học sâu, Transformer và tác tử.", "serve": None},
    "ch2": {"no": 2, "title": "Học máy cơ bản", "icon": "📊", "color": "#2563eb",
            "desc": "Phân loại tiểu đường, hồi quy giá nhà, phân loại nhận xét khách hàng.", "serve": "serve_ch2"},
    "ch3": {"no": 3, "title": "Mạng nơ-ron tích chập", "icon": "🖼️", "color": "#7c3aed",
            "desc": "MNIST, Fashion-MNIST, Kéo–Búa–Bao, CIFAR-10 với CNN tự viết, PyTorch, TensorFlow.",
            "serve": "serve_ch3"},
    "ch4": {"no": 4, "title": "Mạng nơ-ron hồi quy", "icon": "📈", "color": "#0d9488",
            "desc": "Dự báo doanh thu thương mại điện tử, giá cổ phiếu AAPL và giá vàng.", "serve": "serve_ch4"},
    "ch5": {"no": 5, "title": "Knowledge Graph và RAG", "icon": "🕸️", "color": "#ea580c",
            "desc": "Chatbot hỏi đáp có căn cứ trên đồ thị tri thức: tiểu đường, chứng khoán, giá vàng.",
            "serve": "serve_ch5"},
    "ch6": {"no": 6, "title": "Học sâu trên đồ thị", "icon": "🔗", "color": "#db2777",
            "desc": "GCN, GraphSAGE, GAT phân loại nút trên mạng trích dẫn Cora.", "serve": "serve_ch6"},
    "ch7": {"no": 7, "title": "Hệ đa tác tử", "icon": "🤖", "color": "#16a34a",
            "desc": "Giải toán GSM8K: LLM trực tiếp, chuỗi suy luận, công cụ Python và đa tác tử.",
            "serve": "serve_ch7"},
}
KIND_LABEL = {"tabular": "Dữ liệu bảng", "image": "Ảnh", "series": "Chuỗi thời gian", "chat": "Hỏi đáp",
              "text": "Văn bản", "graph": "Đồ thị"}

def preload_torch():
    """Nạp torch trọn vẹn một lần trước mọi serve: nếu một serve nạp torch dở chừng (WinError 1455 khi bộ nhớ ảo
    cạn), module torch nằm lại trong sys.modules ở trạng thái thiếu (`torch._utils`) và làm hỏng serve khác."""
    try:
        import torch  # noqa: F401
        import torch._utils  # noqa: F401
    except Exception as e:
        sys.modules.pop("torch", None)
        print(f"[app] không nạp được torch lúc khởi động: {e}", flush=True)


preload_torch()
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
_modules, _errors = {}, {}


def load_serve(ch):
    """Nạp lười tl3.serve_chN (hoặc mock); trả mô-đun hoặc None và ghi lý do lỗi vào _errors."""
    name = CHAPTERS[ch]["serve"]
    if not name:
        return None
    if ch in _modules:
        return _modules[ch]
    try:
        mod = importlib.import_module("mock_serve") if MOCK else importlib.import_module(f"tl3.{name}")
        if MOCK:
            mod = mod.for_chapter(ch)
        _modules[ch] = mod
        _errors.pop(ch, None)
        return mod
    except Exception as e:  # mô-đun chưa có hoặc lỗi: chỉ trang của chương này báo "đang cập nhật"
        _errors[ch] = f"{type(e).__name__}: {e}"
        return None


def chapter_of(exp_id):
    ch = exp_id.split("_", 1)[0]
    if ch not in CHAPTERS:
        abort(404)
    return ch


def specs_of(ch):
    mod = load_serve(ch)
    return getattr(mod, "SPECS", {}) if mod else {}


def jsonable(o):
    """Chuyển kết quả serve (numpy, PIL, tuple…) thành kiểu JSON; NaN/inf thành None."""
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [jsonable(v) for v in o]
    if hasattr(o, "tolist") and not isinstance(o, (str, bytes)):
        return jsonable(o.tolist())
    if hasattr(o, "item") and not isinstance(o, (str, bytes)):
        try:
            return jsonable(o.item())
        except (TypeError, ValueError):
            pass
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        return None
    if hasattr(o, "save") and hasattr(o, "size"):        # PIL.Image -> data URL để hiển thị
        buf = io.BytesIO()
        o.save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    if isinstance(o, (str, int, bool)) or o is None or isinstance(o, float):
        return o
    if isinstance(o, Path):
        return str(o)
    return str(o)


def read_payload(spec):
    """Gom đầu vào từ form multipart hoặc JSON; ảnh thành PIL.Image, số thành float, giữ đúng tên trong SPECS."""
    from PIL import Image
    data = request.get_json(silent=True) or {}
    if not data:
        data = {k: v for k, v in request.form.items()}
    payload = {}
    for inp in spec.get("inputs", []):
        name, typ = inp["name"], inp.get("type", "text")
        if typ == "image":
            f = request.files.get(name)
            raw = f.read() if f else None
            if raw is None and isinstance(data.get(name), str) and data[name].startswith("data:image"):
                raw = base64.b64decode(data[name].split(",", 1)[1])
            if raw:
                payload[name] = Image.open(io.BytesIO(raw)).convert("RGB")
            continue
        if name not in data or data[name] in ("", None):
            continue
        v = data[name]
        if typ == "number":
            try:
                v = float(v)
                v = int(v) if v.is_integer() and inp.get("step", 1) in (1, "1") else v
            except (TypeError, ValueError):
                pass
        payload[name] = v
    for k, v in data.items():                 # trường phụ không khai báo (history, mode...) vẫn chuyển tiếp
        if k not in payload and k != "model_id" and k not in {i["name"] for i in spec.get("inputs", [])}:
            payload[k] = v
    return payload, (data.get("model_id") or None)


@app.route("/")
def index():
    chapters = []
    for ch, meta in CHAPTERS.items():
        specs = specs_of(ch)
        chapters.append({**meta, "id": ch, "status": "ok" if specs else ("none" if not meta["serve"] else "wait"),
                         "exps": [{"id": e, "title": s.get("title", e), "kind": KIND_LABEL.get(s.get("kind"), s.get("kind")),
                                   "n_models": len(s.get("models", []))} for e, s in specs.items()]})
    total = sum(len(c["exps"]) for c in chapters)
    return render_template("index.html", chapters=chapters, total=total, mock=MOCK)


@app.route("/exp/<exp_id>")
def exp_page(exp_id):
    ch = chapter_of(exp_id)
    specs = specs_of(ch)
    meta = {**CHAPTERS[ch], "id": ch}
    siblings = [{"id": e, "title": s.get("title", e)} for e, s in specs.items()]
    if exp_id not in specs:
        return render_template("pending.html", meta=meta, exp_id=exp_id, error=_errors.get(ch), mock=MOCK), 200
    spec = specs[exp_id]
    return render_template("exp.html", meta=meta, exp_id=exp_id, spec=jsonable(spec), siblings=siblings,
                           kind_label=KIND_LABEL.get(spec.get("kind"), spec.get("kind")), mock=MOCK)


@app.route("/api/exps")
def api_exps():
    """Danh sách exp_id đang phục vụ được và chương còn chờ (dùng bởi screenshots.py)."""
    ready, waiting = [], {}
    for ch, meta in CHAPTERS.items():
        if not meta["serve"]:
            continue
        specs = specs_of(ch)
        ready += list(specs)
        if not specs:
            waiting[ch] = _errors.get(ch, "chưa có SPECS")
    return jsonify({"ready": ready, "waiting": waiting, "mock": MOCK})


@app.route("/api/<exp_id>/spec")
def api_spec(exp_id):
    specs = specs_of(chapter_of(exp_id))
    if exp_id not in specs:
        return jsonify({"error": "đang cập nhật"}), 404
    return jsonify(jsonable(specs[exp_id]))


@app.route("/api/<exp_id>/example/<name>")
def api_example(exp_id, name):
    """Ảnh ví dụ khai báo trong SPECS[exp]['example'][name] (đường dẫn tệp hoặc PIL.Image)."""
    spec = specs_of(chapter_of(exp_id)).get(exp_id) or abort(404)
    ex = (spec.get("example") or {}).get(name)
    if ex is None:
        abort(404)
    if hasattr(ex, "save"):
        buf = io.BytesIO()
        ex.save(buf, format="PNG")
        buf.seek(0)
        return send_file(buf, mimetype="image/png")
    p = Path(str(ex))
    if not p.is_absolute():
        p = ROOT / p
    if not p.exists():
        abort(404)
    return send_file(p)


@app.route("/api/<exp_id>/predict", methods=["POST"])
def api_predict(exp_id):
    ch = chapter_of(exp_id)
    mod = load_serve(ch)
    spec = (getattr(mod, "SPECS", {}) if mod else {}).get(exp_id)
    if spec is None:
        return jsonify({"error": "Thực nghiệm đang cập nhật", "detail": _errors.get(ch)}), 503
    payload, model_id = read_payload(spec)
    t0 = time.perf_counter()
    try:
        out = mod.predict(exp_id, payload, model_id)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500
    out = jsonable(out if isinstance(out, dict) else {"value": out})
    out.setdefault("latency_ms", (time.perf_counter() - t0) * 1000)
    out.setdefault("model", model_id or (spec.get("models") or [None])[0])
    return jsonify(out)


@app.context_processor
def inject_nav():
    return {"nav": [{"id": c, **m} for c, m in CHAPTERS.items()]}


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=PORT, debug=False, threaded=False)
