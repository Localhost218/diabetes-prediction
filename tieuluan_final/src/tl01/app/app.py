"""Web App của tiểu luận: sáu phân hệ thông minh trên một máy chủ Flask (mục 2.6, 3.5, 4.5).

    python app/app.py            # http://127.0.0.1:8090/

Tầng suy luận nạp sẵn mọi mô hình lên RAM lúc khởi động và chạy trên CPU; mỗi phân hệ là một
endpoint POST nhận JSON và trả JSON. Bộ tiền xử lý dựng lại bằng chính hàm load_* đã dùng khi
huấn luyện (cùng seed, cùng cách chia) nên đầu vào web đi qua đúng phép biến đổi của mô hình.
"""
import base64
import io
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from flask import Flask, jsonify, render_template, request
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tl import data_seq as dq  # noqa: E402
from tl import data_tabular as dt  # noqa: E402
from tl import data_vision as dv  # noqa: E402
from tl.cnn_torch import ARCHS  # noqa: E402
from tl.common import MODELS, load_json  # noqa: E402
from tl.mlp_torch import TorchMLP  # noqa: E402
from tl.rnn_torch import CELLS, LABELS, SeqModel  # noqa: E402

torch.set_num_threads(4)
app = Flask(__name__)
M2, M3, M4 = MODELS / "ch2", MODELS / "ch3", MODELS / "ch4"


def load_torch(model, path):
    model.load_state_dict(torch.load(path, map_location="cpu"))
    return model.eval()


# ------------------------------------------------------------------ nạp mô hình Chương 2
DIAB = dt.load_diabetes()
HOUSE = dt.load_house()
REV = dt.load_reviews()
MLP_INFO = load_json("ch2_mlp_torch")
diab_mlp = load_torch(TorchMLP(DIAB["X_train"].shape[1]), M2 / "diabetes_mlp_torch.pt")
diab_rf = joblib.load(M2 / "diabetes_rf.joblib")
house_mlp = load_torch(TorchMLP(HOUSE["X_train"].shape[1]), M2 / "house_mlp_torch.pt")
house_rf = joblib.load(M2 / "house_rf.joblib")
rev_mlp = load_torch(TorchMLP(REV["X_train"].shape[1], 3), M2 / "reviews_mlp_torch.pt")
rev_lr = joblib.load(M2 / "reviews_logreg.joblib")

# ------------------------------------------------------------------ Chương 3
VISION = {}
for ds in ("svhn", "eurosat"):
    for arch in ("resnet18", "mobilenetv2"):
        ck = torch.load(M3 / f"{ds}_{arch}.pt", map_location="cpu")
        m = ARCHS[arch](10, **ck.get("kwargs", {}))
        m.load_state_dict(ck["state"])
        VISION[(ds, arch)] = (m.eval(), ck["mean"], ck["std"], ck["classes"])
VTEST = {ds: dv.load(ds) for ds in ("svhn", "eurosat")}

# ------------------------------------------------------------------ Chương 4
STOCK = dq.stock_windows("FPT")
stock_models = {c: load_torch(SeqModel(c, input_dim=STOCK["X_test"].shape[2]), M4 / f"stock_{c}_torch.pt") for c in CELLS}
CHURN = dq.churn_data()
churn_models = {c: load_torch(SeqModel(c, input_dim=8), M4 / f"churn_{c}_torch.pt") for c in CELLS}
CH4 = load_json("ch4_torch")


def timed(fn):
    t0 = time.perf_counter()
    out = fn()
    return out, round((time.perf_counter() - t0) * 1000, 2)


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/api/diabetes")
def api_diabetes():
    f = request.get_json()
    row = pd.DataFrame([{k: f[k] for k in dt.DIAB_NUM + dt.DIAB_BIN + dt.DIAB_CAT}])
    x = torch.as_tensor(DIAB["pre"].transform(row).astype(np.float32))

    def run():
        with torch.no_grad():
            p_mlp = torch.sigmoid(diab_mlp(x)).item()
        return p_mlp, float(diab_rf.predict_proba(row)[0, 1])
    (p_mlp, p_rf), ms = timed(run)
    level = "cao" if p_mlp >= 0.5 else "trung bình" if p_mlp >= 0.2 else "thấp"
    advice = {"cao": "Nên làm xét nghiệm đường huyết lúc đói và HbA1c tại cơ sở y tế trong thời gian sớm.",
              "trung bình": "Nên theo dõi chế độ ăn, vận động và kiểm tra lại chỉ số định kỳ 6 tháng.",
              "thấp": "Nguy cơ thấp; duy trì lối sống lành mạnh và khám sức khỏe định kỳ."}[level]
    return jsonify(p_mlp=p_mlp, p_rf=p_rf, level=level, advice=advice, ms=ms)


@app.post("/api/house")
def api_house():
    f = request.get_json()
    row = pd.DataFrame([{"log_area": np.log1p(float(f["area"])), "Frontage": f.get("frontage"),
                         "Access Road": f.get("road"), "Floors": f.get("floors"), "Bedrooms": f.get("bedrooms"),
                         "Bathrooms": f.get("bathrooms"), "province": f["province"], "district": f["district"],
                         "Legal status": f.get("legal") or None, "Furniture state": f.get("furniture") or None,
                         "House direction": None}]).astype({"Frontage": float, "Access Road": float, "Floors": float,
                                                            "Bedrooms": float, "Bathrooms": float})
    info = MLP_INFO["house"]

    def run():
        with torch.no_grad():
            mlp = house_mlp(torch.as_tensor(HOUSE["pre"].transform(row).astype(np.float32))).item()
        return mlp * info["y_sd"] + info["y_mu"], float(house_rf.predict(row)[0])
    (p_mlp, p_rf), ms = timed(run)
    rmse = load_json("ch2_sklearn")["house/rf"]["rmse"]
    return jsonify(p_mlp=p_mlp, p_rf=p_rf, low=max(p_rf - rmse, 0), high=p_rf + rmse,
                   per_m2=p_rf * 1000 / float(f["area"]), ms=ms)


@app.post("/api/sentiment")
def api_sentiment():
    text = request.get_json()["text"]

    def run():
        x = torch.as_tensor(REV["pre"].transform([text]).toarray().astype(np.float32))
        with torch.no_grad():
            p_mlp = torch.softmax(rev_mlp(x), 1)[0].numpy()
        return p_mlp, rev_lr.predict_proba([text])[0]
    (p_mlp, p_lr), ms = timed(run)
    vocab = REV["pre"].vocabulary_
    coef = rev_lr.named_steps["model"].coef_
    toks = [t for t in dt.clean_text(text).split() if t in vocab]
    k = int(np.argmax(p_mlp))
    words = sorted({t: float(coef[k, vocab[t]]) for t in toks}.items(), key=lambda kv: -kv[1])[:6]
    return jsonify(labels=dt.SENT_NAMES, p_mlp=p_mlp.tolist(), p_lr=p_lr.tolist(), label=dt.SENT_NAMES[k],
                   score=round(float(p_mlp @ np.array([1, 3, 5])), 2), words=words, ms=ms)


def cam_png(model, arch, x, cls):
    """Class Activation Map: tổng có trọng số các bản đồ đặc trưng cuối theo trọng số FC của lớp dự đoán."""
    with torch.no_grad():
        fmap = model.feature_maps(x) if arch == "resnet18" else model.features(x)
        w = model.fc.weight if arch == "resnet18" else model.fc[1].weight
        cam = torch.einsum("c,chw->hw", w[cls], fmap[0])
    cam = torch.relu(cam)
    cam = (cam / (cam.max() + 1e-8)).numpy()
    return cam


def to_b64(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


@app.post("/api/vision")
def api_vision():
    f = request.get_json()
    ds, arch = f["dataset"], f["arch"]
    model, mean, std, classes = VISION[(ds, arch)]
    size = 32 if ds == "svhn" else 64
    if f.get("image"):
        raw = Image.open(io.BytesIO(base64.b64decode(f["image"].split(",")[1]))).convert("RGB").resize((size, size))
        arr, truth = np.asarray(raw).transpose(2, 0, 1), None
    else:
        i = int(f.get("index", 0)) % len(VTEST[ds]["y_test"])
        arr, truth = VTEST[ds]["x_test"][i], classes[int(VTEST[ds]["y_test"][i])]
    x = ((torch.as_tensor(arr[None]).float() / 255) - mean) / std

    def run():
        with torch.no_grad():
            return torch.softmax(model(x), 1)[0].numpy()
    prob, ms = timed(run)
    top = np.argsort(prob)[::-1][:5]
    cam = cam_png(model, arch, x, int(top[0]))
    import matplotlib
    heat = (matplotlib.colormaps["jet"](np.asarray(Image.fromarray((cam * 255).astype(np.uint8)).resize((size, size), Image.BILINEAR)) / 255)[:, :, :3] * 255)
    base = arr.transpose(1, 2, 0).astype(float)
    overlay = (0.55 * base + 0.45 * heat).clip(0, 255).astype(np.uint8)
    big = lambda a: Image.fromarray(a.astype(np.uint8)).resize((192, 192), Image.NEAREST)  # noqa: E731
    acc = load_json(f"ch3_{ds}")[arch]["acc"]
    return jsonify(image=to_b64(big(base)), cam=to_b64(big(overlay)), truth=truth, ms=ms, acc=acc,
                   top=[{"label": classes[j], "p": float(prob[j])} for j in top])


@app.post("/api/stock")
def api_stock():
    cell = request.get_json().get("cell", "lstm")
    n = int(request.get_json().get("days", 120))
    m = stock_models[cell]
    X = torch.as_tensor(STOCK["X_test"][-n:])

    def run():
        with torch.no_grad():
            return m(X).numpy().ravel()
    ret, ms = timed(run)
    pred = dq.to_price(ret, STOCK["close_now_test"][-n:]) / 1000
    true = STOCK["close_next_test"][-n:] / 1000
    df = STOCK["frame"]
    last = df[dq.FEATURES].to_numpy(np.float64)[-dq.LOOKBACK:]
    xw = last.copy()
    lc = xw[-1, 3]
    xw[:, dq.PRICE_COLS] = (xw[:, dq.PRICE_COLS] / lc - 1) * 100
    logv = np.log1p(df.Volume.to_numpy(np.float64))
    n_tr = len(STOCK["y_train"]) + dq.LOOKBACK
    xw[:, 4] = (np.log1p(last[:, 4]) - logv[:n_tr].mean()) / logv[:n_tr].std()
    xw[:, 7] /= 100
    with torch.no_grad():
        nxt = m(torch.as_tensor(xw[None].astype(np.float32))).item()
    r = CH4[f"stock/{cell}"]
    return jsonify(dates=[str(d.date()) for d in STOCK["dates_test"][-n:]], true=true.tolist(), pred=pred.tolist(),
                   last_date=str(df.index[-1].date()), last_close=lc / 1000, next_ret=nxt,
                   next_price=lc * (1 + nxt / 100) / 1000, rmse=r["rmse"], mae=r["mae"], dir_acc=r["dir_acc"],
                   naive_rmse=CH4["stock/naive"]["rmse"], ms=ms, rsi=float(df.RSI14.iloc[-1]),
                   vol30=float(df.Vol30.iloc[-1]), label=LABELS[cell])


@app.post("/api/churn")
def api_churn():
    f = request.get_json()
    rng = np.random.default_rng(int(f.get("seed", 0)))
    want = f.get("group", "any")
    yt = CHURN["y_test"]
    pool = np.flatnonzero(yt == 1) if want == "churn" else np.flatnonzero(yt == 0) if want == "loyal" else np.arange(len(yt))
    i = int(rng.choice(pool))
    x = torch.as_tensor(CHURN["X_test"][i:i + 1])

    def run():
        with torch.no_grad():
            return {c: torch.sigmoid(m(x)).item() for c, m in churn_models.items()}
    probs, ms = timed(run)
    xr = CHURN["X_test"][i] * CHURN["sd"] + CHURN["mu"]          # về lại thang log1p ban đầu
    return jsonify(index=i, truth=int(yt[i]), probs=probs, thr={c: CH4[f"churn/{c}"]["threshold"] for c in CELLS},
                   secs=np.expm1(xr[:, 6]).round().tolist(), unq=np.expm1(xr[:, 5]).round().tolist(),
                   active=xr[:, 7].round().tolist(), labels=LABELS, ms=ms)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8090, debug=False)
