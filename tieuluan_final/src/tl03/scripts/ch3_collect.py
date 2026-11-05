"""Gom kết quả huấn luyện Chương 3 (models/ch3/results/*.json) thành outputs/metrics/ch3_<exp>.json cho macro \\SL,
và chạy kiểm tra gradient của Basic CNN NumPy -> outputs/metrics/ch3_gradcheck.json.

    python scripts/ch3_collect.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tl3.ch3_common import CFG, MODEL_IDS, RESULTS, TF_PATIENCE, has_result, load_result  # noqa: E402
from tl3.ch3_data import CLASSES, load, load_raw  # noqa: E402
from tl3.common import save_json  # noqa: E402

EXPS = ["mnist", "fashion", "rps", "cifar10"]
STOP_VN = {"early_stop": "dừng sớm", "time_budget": "hết ngân sách thời gian",
           "max_epochs": "chạy đủ số epoch tối đa"}


def top_confusions(cm, names):
    """Hai cặp nhầm lẫn lớn nhất (thật → đoán) ngoài đường chéo, dạng chữ để dùng trong \\SL."""
    off = cm.copy()
    np.fill_diagonal(off, 0)
    out = {}
    for k, flat in enumerate(np.argsort(off.ravel())[::-1][:2], 1):
        i, j = divmod(int(flat), cm.shape[0])
        if off[i, j] == 0:              # ít hơn hai cặp nhầm thật sự: không ghi ô đường chéo
            break
        out[f"conf{k}"] = f"{names[i]} → {names[j]}"
        out[f"conf{k}_n"] = int(off[i, j])
    return out


def speedup(slow, fast):
    """Tỉ số thời gian một epoch, làm tròn một chữ số thập phân dạng chữ (macro SL in nguyên văn)."""
    return f"{slow['epoch_time'] / fast['epoch_time']:.1f}".replace(".", ",")


def summarize(exp):
    out, cfg = {}, CFG[exp]
    (xtr, _), (xva, _), (xte, yte) = load(exp)
    out["split"] = {"train": len(xtr), "val": len(xva), "test": len(xte), "classes": len(CLASSES[exp])}
    out["cfg"] = {k: cfg[k] for k in ("basic_size", "improved_size", "batch", "ep_basic", "ep_improved",
                                      "ep_improved_tf", "tf_budget_min", "scratch_ep")}
    out["cfg"]["tf_patience"] = TF_PATIENCE
    out["cfg"]["scratch_n"] = cfg["scratch_n"] or len(xtr)
    out["aug"] = {"rot": cfg["rot"], "shift": cfg["shift"], "zoom": cfg["zoom"], "color": cfg["color"],
                  "flip": "có" if cfg["flip"] else "không"}
    _, ytr_raw, _, _ = load_raw(exp)
    cnt = np.bincount(ytr_raw)
    out["balance"] = {"max_class": CLASSES[exp][int(cnt.argmax())], "max_n": int(cnt.max()),
                      "min_class": CLASSES[exp][int(cnt.argmin())], "min_n": int(cnt.min()),
                      "ratio": float(cnt.max() / cnt.min())}
    sub = xtr[np.random.default_rng(0).choice(len(xtr), min(1000, len(xtr)), replace=False)]
    out["pixel"] = {"mean": float(sub.mean()), "std": float(sub.std()), "zero_frac": float((sub == 0).mean()),
                    **{f"mean_{c}": float(sub[..., i].mean()) for i, c in enumerate("rgb") if sub.shape[3] == 3}}
    h, c = cfg["basic_size"], xtr.shape[3]
    flat = 64 * (h // 4) ** 2
    basic_total = (9 * c + 1) * 32 + (9 * 32 + 1) * 64 + flat * 128 + 128 + 128 * len(CLASSES[exp]) + len(CLASSES[exp])
    out["dense_share"] = (flat * 128 + 128) / basic_total
    out["flat"] = flat
    out["steps"] = int(np.ceil(len(xtr) / cfg["batch"]))
    out["params_per_sample"] = basic_total / len(xtr)
    for m in MODEL_IDS:
        if not has_result(exp, m):
            continue
        r = load_result(exp, m)
        f1c = np.array(r["f1_per_class"])
        h = r["history"]
        out[m] = {"acc": r["accuracy"], "precision": r["precision"], "recall": r["recall"], "f1": r["f1"],
                  "params": r["params"], "train_time": r["train_time"], "train_min": r["train_time"] / 60,
                  "epoch_time": float(np.mean(h["time"])), "infer_ms": r["infer_ms"],
                  "best_epoch": r["best_epoch"], "epochs": r["epochs"],
                  "val_best": max(h["val_acc"]), "train_acc_last": h["acc"][-1],
                  "errors": int(round((1 - r["accuracy"]) * len(xte))),
                  "worst_class": CLASSES[exp][int(f1c.argmin())], "worst_f1": float(f1c.min()),
                  "best_class": CLASSES[exp][int(f1c.argmax())], "best_f1": float(f1c.max()),
                  "val_first": h["val_acc"][0], "train_loss_last": h["loss"][-1], "val_loss_min": min(h["val_loss"]),
                  **top_confusions(np.array(r["cm"]), CLASSES[exp])}
        p = np.load(RESULTS / f"{exp}__{m}_proba.npy")
        wrong = p.argmax(1) != yte
        out[m]["conf_mean"] = float(p.max(1).mean())
        out[m]["wrong_conf_max"] = float(p.max(1)[wrong].max()) if wrong.any() else 0.0
        out[m]["wrong_conf_mean"] = float(p.max(1)[wrong].mean()) if wrong.any() else 0.0
        if "stop" in r:     # lý do dừng của lượt có dừng sớm, dạng chữ cho \SL
            out[m]["stop_vn"] = STOP_VN[r["stop"]]
        if "gpu_peak_mb" in r:
            out[m]["gpu_mb"] = r["gpu_peak_mb"]
        if "train_used" in r:
            out[m]["train_used"] = r["train_used"]
    done = [m for m in MODEL_IDS if m in out]
    if done:
        b = max(done, key=lambda m: out[m]["acc"])
        out["best"] = {"model": b, "acc": out[b]["acc"], "f1": out[b]["f1"]}
    for fw in ("torch", "tf"):
        if f"{fw}_basic" in out and f"{fw}_improved" in out:
            ba, im = out[f"{fw}_basic"], out[f"{fw}_improved"]
            # cặp nhầm lớn nhất của Basic: Improved còn nhầm bao nhiêu ảnh ở đúng ô đó
            cb, ci = np.array(load_result(exp, f"{fw}_basic")["cm"]), np.array(load_result(exp, f"{fw}_improved")["cm"])
            off = cb.copy()
            np.fill_diagonal(off, 0)
            i, j = divmod(int(off.argmax()), cb.shape[0])
            # "acc" là hiệu dạng tỉ lệ để macro ...Pct in đúng hai chữ số thập phân (điểm phần trăm)
            out[f"gain_{fw}"] = {"acc": im["acc"] - ba["acc"], "acc_pp": 100 * (im["acc"] - ba["acc"]), "f1_pp": 100 * (im["f1"] - ba["f1"]),
                                 "err_cut": 1 - (1 - im["acc"]) / max(1e-9, 1 - ba["acc"]),
                                 "errors_cut": ba["errors"] - im["errors"],
                                 "pair": ba["conf1"], "pair_before": int(cb[i, j]), "pair_after": int(ci[i, j])}
    basics = [out[m]["acc"] for m in ("scratch", "torch_basic", "tf_basic") if m in out]
    if len(basics) > 1:
        out["basic_spread_pp"] = 100 * (max(basics) - min(basics))
        out["basic_spread"] = max(basics) - min(basics)
    if "torch_improved" in out and "tf_improved" in out:  # Improved PyTorch (GPU) hơn Improved TensorFlow (CPU) bao nhiêu
        out["improved_fw_gap"] = out["torch_improved"]["acc"] - out["tf_improved"]["acc"]
    if "torch_basic" in out and "tf_basic" in out:      # cùng kiến trúc, cùng tập huấn luyện, chỉ khác framework
        out["fw_spread"] = abs(out["torch_basic"]["acc"] - out["tf_basic"]["acc"])
    if "scratch" in out and "torch_basic" in out:
        out["speedup_torch_vs_numpy"] = speedup(out["scratch"], out["torch_basic"])
    if "tf_basic" in out and "torch_basic" in out:
        out["speedup_torch_vs_tf"] = speedup(out["tf_basic"], out["torch_basic"])
    save_json(f"ch3_{exp}", out)
    print(exp, {m: round(out[m]["acc"], 4) for m in done})


def gradcheck():
    from tl3.ch3_scratch import gradient_check
    errs = gradient_check()
    big = gradient_check(eps=1e-3)          # bước lớn: minh hoạ sai số giả do vượt điểm gãy ReLU/MaxPool
    save_json("ch3_gradcheck", {"max": float(max(errs.values())), **{k: float(v) for k, v in errs.items()},
                                "eps_large_max": float(max(big.values()))})
    print("gradcheck", errs)


if __name__ == "__main__":
    for e in EXPS:
        summarize(e)
    gradcheck()
