"""Chương 2 — các mô hình học máy kinh điển dưới dạng Pipeline scikit-learn (tiền xử lý + mô hình)."""
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVC, SVR
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor

from .common import SEED

CLF_NAMES = {"baseline": "Baseline (lớp đa số)", "logreg": "Logistic Regression", "knn": "KNN (k = 25)",
             "nb": "Gaussian Naive Bayes", "tree": "Decision Tree", "rf": "Random Forest", "xgb": "XGBoost",
             "lgbm": "LightGBM", "svm": "SVM (RBF)", "mlp": "MLP (PyTorch)"}
REG_NAMES = {"baseline": "Baseline (trung bình)", "linear": "Linear Regression", "ridge": "Ridge (L2)",
             "lasso": "Lasso (L1)", "poly": "Polynomial bậc 2 + Ridge", "knn": "KNN (k = 10)",
             "tree": "Decision Tree", "rf": "Random Forest", "xgb": "XGBoost", "svr": "SVR (RBF)",
             "mlp": "MLP (PyTorch)"}


def classifiers(pos_weight=1.0):
    """Bộ mô hình phân loại cho Diabetes; pos_weight = n_âm / n_dương dùng cho XGBoost khi cân bằng lớp."""
    return {
        "baseline": DummyClassifier(strategy="most_frequent"),
        "logreg": LogisticRegression(C=1.0, max_iter=2000),
        "knn": KNeighborsClassifier(n_neighbors=25, weights="distance", n_jobs=8),
        "nb": GaussianNB(),
        "tree": DecisionTreeClassifier(max_depth=8, min_samples_leaf=20, random_state=SEED),
        "rf": RandomForestClassifier(n_estimators=300, min_samples_leaf=2, n_jobs=8, random_state=SEED),
        "xgb": XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.9,
                             colsample_bytree=0.9, eval_metric="logloss", n_jobs=8, random_state=SEED),
        "lgbm": LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31, n_jobs=8,
                               random_state=SEED, verbose=-1),
        "svm": SVC(C=1.0, kernel="rbf", gamma="scale", probability=True, random_state=SEED),
    }


def clf_pipeline(pre, model):
    """Ghép bộ tiền xử lý (StandardScaler + One-Hot) với một mô hình phân loại."""
    return Pipeline([("pre", pre), ("model", model)])


def regressors():
    """Bộ mô hình hồi quy cho House Price. SVR học trên giá đã chuẩn hóa (TransformedTargetRegressor)."""
    return {
        "baseline": DummyRegressor(strategy="mean"),
        "linear": LinearRegression(),
        "ridge": Ridge(alpha=10.0),
        "lasso": Lasso(alpha=1000.0, max_iter=50000),
        "poly": make_pipeline(PolynomialFeatures(degree=2, include_bias=False), StandardScaler(),
                              Ridge(alpha=30.0)),
        "knn": KNeighborsRegressor(n_neighbors=10, weights="distance"),
        "tree": DecisionTreeRegressor(max_depth=6, min_samples_leaf=10, random_state=SEED),
        "rf": RandomForestRegressor(n_estimators=400, min_samples_leaf=2, n_jobs=8, random_state=SEED),
        "xgb": XGBRegressor(n_estimators=600, max_depth=3, learning_rate=0.03, subsample=0.9,
                            colsample_bytree=0.9, n_jobs=8, random_state=SEED),
        "svr": TransformedTargetRegressor(SVR(C=3.0, epsilon=0.05, kernel="rbf"), transformer=StandardScaler()),
    }


def reg_pipeline(pre, model):
    """Ghép bộ tiền xử lý với một mô hình hồi quy."""
    return Pipeline([("pre", pre), ("model", model)])


__all__ = ["classifiers", "regressors", "clf_pipeline", "reg_pipeline", "CLF_NAMES", "REG_NAMES",
           "LGBMRegressor"]
