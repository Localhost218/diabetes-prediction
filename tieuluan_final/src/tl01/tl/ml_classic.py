"""Mục 2.3: các pipeline scikit-learn của bốn họ thuật toán học máy cơ bản.

Mỗi hàm build_* trả về một Pipeline gồm bộ tiền xử lý (chỉ khớp trên tập huấn luyện khi gọi fit)
và mô hình, nên lưu bằng joblib là có ngay đối tượng nhận dữ liệu thô cho Web App.
"""
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, SVR, LinearSVC

from .common import SEED


def build_linear_regression_pipeline(preprocessor, alpha=None):
    """alpha=None: bình phương tối thiểu thường (OLS); alpha>0: Ridge (phạt L2) chống đa cộng tuyến."""
    reg = LinearRegression() if alpha is None else Ridge(alpha=alpha, random_state=SEED)
    return Pipeline([("pre", preprocessor), ("model", reg)])


def build_logistic_regression_pipeline(preprocessor, balanced=True):
    clf = LogisticRegression(C=1.0, class_weight="balanced" if balanced else None,
                             solver="lbfgs", max_iter=2000, random_state=SEED)
    return Pipeline([("pre", preprocessor), ("model", clf)])


def build_rf_pipeline(preprocessor, task="classification", n_estimators=200, max_depth=None,
                      min_samples_leaf=2, n_jobs=12):
    if task == "regression":
        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                      min_samples_leaf=min_samples_leaf, n_jobs=n_jobs, random_state=SEED)
    else:
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                       min_samples_leaf=min_samples_leaf, class_weight="balanced_subsample",
                                       n_jobs=n_jobs, random_state=SEED)
    return Pipeline([("pre", preprocessor), ("model", model)])


def build_svm_pipeline(preprocessor, task="classification", kernel="rbf", C=1.0):
    """SVM tính khoảng cách nên đặc trưng phải cùng thang đo: bộ tiền xử lý đã có StandardScaler.

    Văn bản TF-IDF thưa nhiều chiều dùng LinearSVC (độ phức tạp tuyến tính theo số mẫu); dữ liệu
    bảng ít chiều dùng nhân RBF.
    """
    if task == "regression":
        model = SVR(kernel=kernel, C=C, epsilon=0.1, cache_size=2000)
    elif kernel == "linear":
        model = LinearSVC(C=C, class_weight="balanced", max_iter=5000, random_state=SEED)
    else:
        model = SVC(kernel=kernel, C=C, gamma="scale", class_weight="balanced", cache_size=4000,
                    random_state=SEED)
    return Pipeline([("pre", preprocessor), ("model", model)])
