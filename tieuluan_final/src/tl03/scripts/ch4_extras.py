"""Các thí nghiệm bổ trợ của Chương 4 (PyTorch, GPU nhẹ):

1. Chuẩn hoá MinMax toàn cục (cách của bài mẫu) so với chuẩn hoá tương đối theo cửa sổ.
2. Độ dài cửa sổ 10–90 bước.
3. Sai số theo tầm dự báo (1–10 bước, đệ quy).
4. Gradient biến mất / bùng nổ: chuẩn ‖∂h_T/∂x_t‖ theo khoảng cách T − t.
Ghi outputs/metrics/ch4_extras.json và outputs/preds/ch4_minmax.npz.
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tl3.ch4_data import Prepared, experiment  # noqa: E402
from tl3.ch4_forecast import recursive_forecast  # noqa: E402
from tl3.ch4_metrics import forecast_metrics  # noqa: E402
from tl3.ch4_torch import SeqRegressor, fit, predict  # noqa: E402
from tl3.common import OUT, device, limit_threads, model_dir, save_json, seed_all  # noqa: E402
from tl3.gpu import cap  # noqa: E402

DEV = device()


def test_metrics(data, model):
    p = predict(model, data.part("test")[0], DEV)
    yhat = data.to_original(p, "test")
    return forecast_metrics(data.truth(), yhat, data.naive()), yhat


def minmax_ablation():
    """Huấn luyện LSTM/GRU với MinMax toàn cục (fit trên train) — giá test vượt vùng train không ngoại suy được."""
    out, store = {}, {}
    for name in ("aapl", "gold"):
        exp = experiment(name)
        out[name] = {}
        for mode in ("minmax", "relative"):
            data = Prepared(exp, mode=mode)
            for arch in ("lstm", "gru"):
                seed_all()
                model, hist, _ = fit(arch, data, DEV, verbose=False)
                m, yhat = test_metrics(data, model)
                out[name][f"{mode}_{arch}"] = m
                store[f"{name}_{mode}_{arch}"] = yhat
                print(name, mode, arch, round(m["rmse"], 3), round(m["r2"], 4), flush=True)
            store[f"{name}_truth"] = data.truth()
            out[name]["train_max"] = float(exp.frame[exp.target].iloc[:int(round(len(exp.frame) * 0.7))].max())
            out[name]["test_max"] = float(data.truth().max())
            out[name]["test_min"] = float(data.truth().min())
    np.savez(OUT / "preds" / "ch4_minmax.npz", **store)
    return out


def window_ablation():
    """RMSE trên test / RMSE naive khi đổi độ dài cửa sổ."""
    out = {}
    for name in ("ecommerce", "aapl", "gold"):
        out[name] = {}
        for w in (10, 20, 30, 60, 90):
            data = Prepared(experiment(name), window=w)
            naive = forecast_metrics(data.truth(), data.naive())["rmse"]
            for arch in ("lstm", "gru"):
                seed_all()
                model, hist, secs = fit(arch, data, DEV, verbose=False)
                m, _ = test_metrics(data, model)
                out[name][f"{arch}_w{w}"] = {"rmse": m["rmse"], "ratio": m["rmse"] / naive, "secs": secs}
                print(name, w, arch, round(m["rmse"] / naive, 4), flush=True)
    return out


def horizon_eval(steps=10, stride=3):
    """Dự báo đệ quy 1..steps bước từ nhiều gốc thời gian trong test bằng LSTM PyTorch đã huấn luyện."""
    out = {}
    for name in ("ecommerce", "aapl", "gold"):
        data = Prepared(experiment(name))
        model = SeqRegressor(data.X.shape[-1], "lstm").to(DEV)
        model.load_state_dict(torch.load(model_dir("ch4") / f"{name}_torch_lstm.pt", map_location=DEV))
        model.eval()
        fn = lambda x: predict(model, x, DEV)  # noqa: E731
        raw = data.exp.frame.values.astype(np.float64)
        t = data.t_idx
        start = data.window + data.sl["test"].start          # chỉ số hàng thô của giá trị test đầu tiên
        errs, nerrs = [[] for _ in range(steps)], [[] for _ in range(steps)]
        for o in range(start, len(raw) - steps, stride):
            f = recursive_forecast(data, fn, raw[o - data.window:o], steps)
            truth = raw[o:o + steps, t]
            for h in range(steps):
                errs[h].append(f[h] - truth[h])
                nerrs[h].append(raw[o - 1, t] - truth[h])
        rm = [float(np.sqrt(np.mean(np.square(e)))) for e in errs]
        rn = [float(np.sqrt(np.mean(np.square(e)))) for e in nerrs]
        out[name] = {"rmse_h": rm, "naive_h": rn, "origins": len(errs[0]), "h1": rm[0], "h10": rm[-1],
                     "naive_h10": rn[-1], "growth": rm[-1] / rm[0]}
        print(name, "horizon", [round(v, 2) for v in rm], flush=True)
    return out


def gradient_flow(T=100, seeds=20, hidden=64):
    """‖∂h_T/∂x_t‖ (trung bình nhiều khởi tạo) cho RNN/LSTM/GRU, và cho RNN khi nhân W_hh với các hệ số."""
    def norms(cell_fn, scale=None):
        acc = np.zeros(T)
        for s in range(seeds):
            torch.manual_seed(s)
            cell = cell_fn()
            if scale is not None:
                with torch.no_grad():
                    w = cell.weight_hh_l0
                    rho = torch.linalg.eigvals(w).abs().max()
                    w.mul_(scale / rho)                       # đặt bán kính phổ của W_hh = scale
            x = torch.randn(1, T, 8, requires_grad=True)
            out, _ = cell(x)
            out[0, -1].sum().backward()
            acc += x.grad[0].norm(dim=1).numpy()
        return (acc / seeds)[::-1]                            # chỉ số k = T − t (khoảng cách tới bước cuối)
    def lstm_forget_bias(b=3.0):
        """LSTM với bias cổng quên = b (f_t ≈ σ(b) ≈ 0,95): đường cell state gần như giữ nguyên gradient."""
        cell = torch.nn.LSTM(8, hidden, batch_first=True)
        with torch.no_grad():
            for bias in (cell.bias_ih_l0, cell.bias_hh_l0):
                bias[hidden:2 * hidden] = b / 2               # thứ tự cổng của PyTorch: i, f, g, o
        return cell
    res = {"rnn": norms(lambda: torch.nn.RNN(8, hidden, batch_first=True)),
           "lstm": norms(lambda: torch.nn.LSTM(8, hidden, batch_first=True)),
           "lstm_fb3": norms(lstm_forget_bias),
           "gru": norms(lambda: torch.nn.GRU(8, hidden, batch_first=True))}
    for sc in (0.5, 0.9, 1.1, 1.5):
        res[f"rnn_rho{sc}"] = norms(lambda: torch.nn.RNN(8, hidden, batch_first=True, nonlinearity="tanh"), sc)
        res[f"lin_rho{sc}"] = norms(lambda: torch.nn.RNN(8, hidden, batch_first=True, nonlinearity="relu"), sc)
    summary = {k: {"k0": float(v[0]), "k20": float(v[20]), "k50": float(v[50]), "k99": float(v[99]),
                   "decay_50": float(v[50] / v[0])} for k, v in res.items()}
    return {"summary": summary, "curve": {k: v.tolist() for k, v in res.items()}}


if __name__ == "__main__":
    limit_threads(3)
    cap(0.15)
    torch.backends.cudnn.benchmark = False
    if "--only-gradient" in sys.argv:                      # chạy lại riêng phần 4, giữ kết quả 1–3 đã có
        from tl3.common import load_json
        res = load_json("ch4_extras")
        res["gradient"] = gradient_flow()
        save_json("ch4_extras", res)
        sys.exit(0)
    res = {"gradient": gradient_flow()}
    print("gradient ok", flush=True)
    res["horizon"] = horizon_eval()
    res["minmax"] = minmax_ablation()
    res["window"] = window_ablation()
    save_json("ch4_extras", res)
