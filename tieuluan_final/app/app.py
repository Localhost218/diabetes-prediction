"""Web App tiểu luận final — "Diabetes AI Lab": một ứng dụng Flask phục vụ thực nghiệm của bảy chương.

Kế thừa cơ chế phục vụ của Web App Tiểu luận 03 (thành viên nhóm 07 Nguyễn Duy Nghĩa): mỗi mô-đun serve cung cấp
`SPECS` (mô tả đầu vào, mô hình, ví dụ) và `predict(exp_id, payload, model_id)`; trang thực nghiệm dựng form từ SPECS
và hiển thị kết quả theo `kind`. Khác bản gốc: một chương gom thực nghiệm từ NHIỀU mô-đun (mạch chính tiểu đường
trong gói `dm`, miền đối chiếu từ `tl3`), và giao diện được thiết kế lại hoàn toàn (thanh bên, bảng điều khiển).

    python app/app.py                 # http://127.0.0.1:8095/
    TLF_MOCK=1 python app/app.py      # dùng mô-đun giả để thử mọi kiểu giao diện
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

ROOT = Path(__file__).resolve().parents[1]           # tieuluan_final/
TL3 = ROOT / "src" / "tl03"
for p in (ROOT / "src", TL3, TL3 / "app", Path(__file__).resolve().parent):
    sys.path.insert(0, str(p))
os.chdir(TL3)                                          # mô-đun tl3 đọc models/, data/ theo đường dẫn tương đối

MOCK = os.environ.get("TLF_MOCK") == "1"
PORT = int(os.environ.get("TLF_PORT", "8095"))

# Thông tin trình bày của từng chương; "serve" là danh sách mô-đun (mạch tiểu đường trước, miền đối chiếu sau).
CHAPTERS = {
    "ch1": {"no": 1, "title": "Lịch sử phát triển AI", "icon": "history", "color": "#475569",
            "focus": "Pima Indians Diabetes", "others": "—",
            "desc": "Từ hệ chuyên gia, perceptron đến học sâu và mô hình nền tảng, tái hiện trên dữ liệu tiểu đường Pima.",
            "serve": []},
    "ch2": {"no": 2, "title": "Học máy cơ bản", "icon": "table", "color": "#0f766e",
            "focus": "NHANES, Diabetes Prediction 100k", "others": "Giá nhà, nhận xét khách hàng",
            "desc": "Phân loại nguy cơ tiểu đường; đối chiếu hồi quy giá nhà và phân loại văn bản.",
            "serve": ["dm.serve_ch2", "tl3.serve_ch2"]},
    "ch3": {"no": 3, "title": "Mạng nơ-ron tích chập", "icon": "image", "color": "#7c3aed",
            "focus": "Ảnh đáy mắt (bệnh võng mạc ĐTĐ)", "others": "MNIST, Fashion-MNIST, CIFAR-10",
            "desc": "Phân độ bệnh võng mạc tiểu đường; đối chiếu ảnh chữ số, thời trang và vật thể.",
            "serve": ["dm.serve_ch3", "tl3.serve_ch3"]},
    "ch4": {"no": 4, "title": "Mạng nơ-ron hồi quy", "icon": "wave", "color": "#0369a1",
            "focus": "Chuỗi glucose CGM", "others": "Cổ phiếu AAPL, vàng GLD, doanh thu TMĐT",
            "desc": "Dự báo đường huyết 15–60 phút; đối chiếu chuỗi giá tài chính và doanh thu.",
            "serve": ["dm.serve_ch4", "tl3.serve_ch4"]},
    "ch5": {"no": 5, "title": "Knowledge Graph và RAG", "icon": "graph", "color": "#c2410c",
            "focus": "KB ADA 2024, KG tiểu đường", "others": "Chứng khoán, giá vàng",
            "desc": "Hỏi đáp có căn cứ trên đồ thị tri thức tiểu đường; đối chiếu miền tài chính.",
            "serve": ["dm.serve_ch5", "tl3.serve_ch5"]},
    "ch6": {"no": 6, "title": "Học sâu trên đồ thị", "icon": "nodes", "color": "#be185d",
            "focus": "Pubmed-Diabetes", "others": "Cora, CiteSeer",
            "desc": "Phân loại y văn tiểu đường bằng GCN, GraphSAGE, GAT; đối chiếu mạng trích dẫn khoa học máy tính.",
            "serve": ["dm.serve_ch6", "tl3.serve_ch6"]},
    "ch7": {"no": 7, "title": "Hệ đa tác tử", "icon": "agents", "color": "#15803d",
            "focus": "Hỏi đáp tiểu đường nhiều tác tử", "others": "GSM8K (suy luận toán)",
            "desc": "So sánh trả lời trực tiếp, chuỗi suy luận, công cụ và đa tác tử tranh luận.",
            "serve": ["dm.serve_ch7", "tl3.serve_ch7"]},
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
    """Nạp lười các mô-đun serve của chương; trả danh sách mô-đun nạp được, ghi lỗi của mô-đun hỏng vào _errors."""
    if ch in _modules:
        return _modules[ch]
    mods, errs = [], []
    for name in CHAPTERS[ch]["serve"]:
        try:
            if MOCK:
                mod = importlib.import_module("mock_serve").for_chapter(ch)
                mods.append(mod)
                break
            mods.append(importlib.import_module(name))
        except Exception as e:  # mô-đun chưa có hoặc lỗi: chỉ phần thực nghiệm của mô-đun đó vắng mặt
            errs.append(f"{name}: {type(e).__name__}: {e}")
    if errs:
        _errors[ch] = "; ".join(errs)
    if mods:
        _modules[ch] = mods
    return mods


def module_for(ch, exp_id):
    for mod in load_serve(ch):
        if exp_id in getattr(mod, "SPECS", {}):
            return mod
    return None


def is_diabetes(ch, exp_id):
    """Thực nghiệm thuộc mạch tiểu đường: mô-đun dm (tiền tố chN_dm) hoặc thực nghiệm tiểu đường của tl3."""
    return exp_id.startswith(ch + "_dm") or any(k in exp_id for k in ("diabetes", "retina", "pubmed", "cgm"))


def chapter_of(exp_id):
    ch = exp_id.split("_", 1)[0]
    if ch not in CHAPTERS:
        abort(404)
    return ch


def specs_of(ch):
    specs = {}
    for mod in load_serve(ch):
        specs.update(getattr(mod, "SPECS", {}))
    return specs


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
                                   "n_models": len(s.get("models", [])), "diabetes": is_diabetes(ch, e)}
                                  for e, s in specs.items()]})
    total = sum(len(c["exps"]) for c in chapters)
    n_dm = sum(e["diabetes"] for c in chapters for e in c["exps"])
    return render_template("index.html", chapters=chapters, total=total, n_dm=n_dm, mock=MOCK)


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
                           diabetes=is_diabetes(ch, exp_id),
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
    mod = module_for(ch, exp_id)
    spec = getattr(mod, "SPECS", {}).get(exp_id) if mod else None
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
    nav = []
    for c, m in CHAPTERS.items():
        nav.append({"id": c, **m, "exps": [{"id": e, "title": sp.get("title", e), "diabetes": is_diabetes(c, e)}
                                           for e, sp in specs_of(c).items()]})
    return {"nav": nav, "current": request.view_args.get("exp_id") if request.view_args else None}


# ---------------------------------------------------------------- Chương 7B: mèo đuổi chuột (bài Agent của tác giả)
_CM = {}


def _catmouse_train(cfg):
    from dm import catmouse_base as cm
    t0 = time.perf_counter()
    env = cm.CatMouseEnv(mouse_mode=cfg.get("mouse_mode", "flee"), mouse_move_prob=float(cfg.get("mouse_move_prob", 0.8)))
    agent = cm.QLearningCat(env, alpha=float(cfg.get("alpha", 0.2)), gamma=float(cfg.get("gamma", 0.95)))
    hist = agent.train(episodes=int(min(30000, max(500, int(cfg.get("episodes", 15000))))), seed=int(cfg.get("seed", 0)))
    agents = {"random": cm.RandomCat(), "rule_based": cm.RuleBasedCat(), "q_learning": agent}
    evaluation = [{"id": k, **{a: round(b, 3) if isinstance(b, float) else b for a, b in cm.evaluate(env, ag).items()}}
                  for k, ag in agents.items()]
    summary = {"config": cfg, "layout": env.layout, "cells": [list(c) for c in env.cells], "n_states": env.n_states,
               "rewards": {"step": cm.R_STEP, "wall": cm.R_WALL, "catch": cm.R_CATCH},
               "train_time_s": round(time.perf_counter() - t0, 2),
               "learning_curve": [{k: round(v, 3) for k, v in p.items()} for p in cm.learning_curve(hist)],
               "evaluation": evaluation,
               "q_values": agent.Q.reshape(env.n_cells, env.n_cells, 4).round(1).tolist()}
    _CM.update(env=env, agents=agents, summary=summary)
    return summary


@app.route("/catmouse")
def catmouse_page():
    return send_file(Path(__file__).resolve().parent / "static" / "catmouse" / "cat_mouse.html")


@app.route("/api/v1/catmouse/model")
def catmouse_model():
    return jsonify(_CM.get("summary") or _catmouse_train({}))


@app.route("/api/v1/catmouse/train", methods=["POST"])
def catmouse_train():
    return jsonify(_catmouse_train(request.get_json(silent=True) or {}))


@app.route("/api/v1/catmouse/episode", methods=["POST"])
def catmouse_episode():
    from dm import catmouse_base as cm
    if "env" not in _CM:
        _catmouse_train({})
    req = request.get_json(silent=True) or {}
    env = _CM["env"]
    cat = tuple(req["cat"]) if req.get("cat") else None
    mouse = tuple(req["mouse"]) if req.get("mouse") else None
    for cell in (cat, mouse):
        if cell is not None and cell not in env.index:
            return jsonify({"detail": f"Ô {list(cell)} là tường hoặc nằm ngoài lưới"}), 422
    if (cat is None) != (mouse is None) or (cat is not None and cat == mouse):
        return jsonify({"detail": "Cần cả vị trí mèo và chuột, và hai ô phải khác nhau"}), 422
    return jsonify(cm.run_episode(env, _CM["agents"][req.get("agent", "q_learning")], cat=cat, mouse=mouse,
                                  seed=int(req.get("seed", 0))))


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=PORT, debug=False, threaded=False)
