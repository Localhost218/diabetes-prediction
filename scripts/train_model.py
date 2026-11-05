"""
Huấn luyện và so sánh ML (Gradient Boosting) với DL (MLP PyTorch) cho nhánh DB → ML/DL.

    python scripts/fetch_nhanes.py                  # một lần: tải dữ liệu NHANES
    python scripts/train_model.py                   # train trên cả NHANES và bộ dữ liệu VN
    python scripts/train_model.py --dataset nhanes  # chỉ một bộ
    python scripts/fetch_brfss.py && python scripts/train_model.py --dataset brfss   # 3,17 triệu người
    python scripts/train_model.py --reload          # nạp lại CSV vào DB

Dữ liệu được đọc từ DB quan hệ (bảng nhanes_samples / training_samples). Chia 70/10/20 (train/val/test),
tăng cường bằng cách che ngẫu nhiên đặc trưng; đánh giá trên tập test đầy đủ, tập test bị che (mô phỏng
người dùng chatbot chỉ khai một phần) và nhóm con (người gốc Á với NHANES).
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from src.config import settings  # noqa: E402
from src.models.datasets import DATASETS, load_dataframe, mask_augment  # noqa: E402
from src.models.deep_mlp import MLPRisk  # noqa: E402
from src.models.tabular_ml import GradientBoostingRisk  # noqa: E402
from src.storage.db import engine, init_db  # noqa: E402


def scores(y, p) -> dict:
    return {"roc_auc": round(float(roc_auc_score(y, p)), 4), "pr_auc": round(float(average_precision_score(y, p)), 4),
            "brier": round(float(brier_score_loss(y, p)), 4)}


def run(name: str, reload: bool) -> dict:
    spec = DATASETS[name]
    if name in ("nhanes", "brfss") and not Path(spec.csv_path).exists():
        raise SystemExit(f"Chưa có dữ liệu {name.upper()} — chạy: python scripts/fetch_{name}.py")
    df = load_dataframe(spec, engine, reload)
    X, y = df[spec.features].astype(float), df[spec.label].astype(int)
    group = df[spec.subgroup] if spec.subgroup else pd.Series(0, index=df.index)

    X_tmp, X_te, y_tmp, y_te, g_tmp, g_te = train_test_split(X, y, group, test_size=0.2, stratify=y, random_state=42)
    X_tr, X_val, y_tr, y_val = train_test_split(X_tmp, y_tmp, test_size=0.125, stratify=y_tmp, random_state=42)
    X_aug, y_aug = mask_augment(X_tr, y_tr, spec, seed=0)
    X_val_aug, y_val_aug = mask_augment(X_val, y_val, spec, seed=1)
    X_te_masked, _ = mask_augment(X_te, y_te, spec, seed=2)
    X_te_masked = X_te_masked.iloc[len(X_te):].reset_index(drop=True)

    print(f"\n=== {spec.title}: {len(df):,} mẫu, tỷ lệ dương tính {y.mean():.1%}, {len(spec.features)} đặc trưng ===")
    gbdt = GradientBoostingRisk.train(spec, X_aug, y_aug)
    # Bộ hàng triệu mẫu: batch lớn, ít epoch hơn (mỗi epoch đã đủ nhiều bước cập nhật)
    big = len(X_aug) > 1_000_000
    mlp = MLPRisk.train(spec, X_aug, y_aug, X_val_aug, y_val_aug,
                        **({"epochs": 20, "patience": 3, "batch_size": 4096} if big else {}))
    logreg = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
                           LogisticRegression(max_iter=3000)).fit(X_aug, y_aug)

    report = {"dataset": spec.title, "n_samples": int(len(df)), "positive_rate": round(float(y.mean()), 4),
              "n_train": int(len(X_tr)), "n_test": int(len(X_te)), "models": {}}
    predictors = {"gbdt": gbdt.predict_proba, "mlp": mlp.predict_proba,
                  "logistic_regression": lambda d: logreg.predict_proba(d)[:, 1]}
    for key, predict in predictors.items():
        p_full = predict(X_te)
        m = {"test": scores(y_te, p_full), "test_partial_data": scores(y_te, predict(X_te_masked))}
        if spec.subgroup and g_te.sum() > 50:
            mask = (g_te == 1).to_numpy()
            m["test_subgroup"] = {"name": "Người gốc Á", "n": int(mask.sum()), **scores(y_te[mask], p_full[mask])}
        report["models"][key] = m

    ens = (gbdt.predict_proba(X_te) + mlp.predict_proba(X_te)) / 2
    agreement = float(np.mean(np.abs(gbdt.predict_proba(X_te) - mlp.predict_proba(X_te)) < 0.15))
    report["models"]["ensemble"] = {"test": scores(y_te, ens)}
    report["gbdt_mlp_agreement_rate"] = round(agreement, 4)

    common = {"dataset": spec.title, "n_train": int(len(X_tr))}
    gbdt.metrics = {**common, **report["models"]["gbdt"]["test"], "partial_data_auc": report["models"]["gbdt"]["test_partial_data"]["roc_auc"]}
    mlp.metrics = {**common, **mlp.metrics, **report["models"]["mlp"]["test"],
                   "partial_data_auc": report["models"]["mlp"]["test_partial_data"]["roc_auc"]}
    gbdt.save()
    mlp.save()
    (settings.model_dir / f"metrics_{name}.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))

    print(f"{'Mô hình':<22}{'AUC':>8}{'PR-AUC':>9}{'Brier':>8}{'AUC thiếu dữ liệu':>20}{'AUC người gốc Á':>18}")
    for key, m in report["models"].items():
        sub = m.get("test_subgroup", {}).get("roc_auc", "")
        partial = m.get("test_partial_data", {}).get("roc_auc", "")
        print(f"{key:<22}{m['test']['roc_auc']:>8}{m['test']['pr_auc']:>9}{m['test']['brier']:>8}{partial:>20}{sub:>18}")
    print(f"GBDT và MLP lệch nhau < 15 điểm % ở {agreement:.1%} ca kiểm tra; MLP dừng sau {mlp.metrics['epochs_trained']} epoch")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=["nhanes", "brfss", "vn", "all"], default="all")
    parser.add_argument("--reload", action="store_true", help="Nạp lại CSV vào DB")
    args = parser.parse_args()
    init_db()
    for name in (["nhanes", "brfss", "vn"] if args.dataset == "all" else [args.dataset]):
        run(name, args.reload)


if __name__ == "__main__":
    main()
