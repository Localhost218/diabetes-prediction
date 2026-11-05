"""Chương 2 — nạp, làm sạch, biểu diễn và chia ba bộ dữ liệu: Diabetes (phân loại), House Price (hồi quy),
Women's E-Commerce Reviews (phân loại văn bản)."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .common import SEED, data_dir

D = data_dir("ch2")

DIAB_NUM = ["age", "hypertension", "heart_disease", "bmi", "HbA1c_level", "blood_glucose_level"]
DIAB_CAT = ["gender", "smoking_history"]
DIAB_TARGET = "diabetes"

HOUSE_NUM = ["Area", "Bedrooms", "Bathrooms", "Stories", "Parking", "Age", "Locality Rating"]
HOUSE_CAT = ["City", "Furnishing", "Main Road", "Guest Room", "Basement", "Water Supply", "Air Conditioning",
             "Preferred Tenant"]
HOUSE_TARGET = "Price"


def download():
    """Tải ba bộ dữ liệu Kaggle (ẩn danh qua kagglehub) và chép vào data/ch2/ với tên ngắn."""
    import glob
    import shutil

    import kagglehub
    src = {"ghnshymsaini/diabetes-prediction-dataset": "diabetes.csv",
           "chershi/house-price-prediction-dataset-2000-rows": "house_price.csv",
           "nicapotato/womens-ecommerce-clothing-reviews": "womens_reviews.csv"}
    for slug, name in src.items():
        if (D / name).exists():
            continue
        path = kagglehub.dataset_download(slug)
        csv = glob.glob(f"{path}/**/*.csv", recursive=True)[0]
        shutil.copy(csv, D / name)


# ------------------------------------------------------------------ Diabetes
def load_diabetes_raw():
    return pd.read_csv(D / "diabetes.csv")


def audit(df, numeric):
    """Khảo sát chất lượng: kích thước, giá trị thiếu, bản ghi trùng, ngoại lai theo quy tắc IQR."""
    rep = {"rows": len(df), "n_rows": len(df), "cols": df.shape[1], "missing": int(df.isna().sum().sum()),
           "duplicates": int(df.duplicated().sum()), "outliers": {}}
    for c in numeric:
        q1, q3 = df[c].quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        rep["outliers"][c] = {"q1": float(q1), "q3": float(q3), "lo": float(lo), "hi": float(hi),
                              "n": int(((df[c] < lo) | (df[c] > hi)).sum())}
    return rep


def clean_diabetes(df):
    """Bỏ bản ghi trùng; kiểm tra miền giá trị hợp lệ; giữ ngoại lai (có ý nghĩa lâm sàng);
    'No Info' của smoking_history giữ là một hạng mục riêng thay vì đoán giá trị."""
    out = df.drop_duplicates().reset_index(drop=True)
    valid = (out["age"].between(0, 120) & out["bmi"].between(10, 100) & out["HbA1c_level"].between(3, 15)
             & out["blood_glucose_level"].between(50, 400)
             & out[["hypertension", "heart_disease", DIAB_TARGET]].isin([0, 1]).all(axis=1))
    return out[valid].reset_index(drop=True)


def diabetes_preprocessor():
    """Chuẩn hóa z-score cho 6 thuộc tính số và One-Hot cho 2 thuộc tính phân loại."""
    return ColumnTransformer([
        ("num", StandardScaler(), DIAB_NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), DIAB_CAT),
    ])


def split_3way(X, y, val=0.15, test=0.15, stratify=True, seed=SEED):
    """Chia Train/Validation/Test theo tỷ lệ 70/15/15, phân tầng theo nhãn nếu là bài toán phân loại."""
    strat = y if stratify else None
    X_tmp, X_te, y_tmp, y_te = train_test_split(X, y, test_size=test, stratify=strat, random_state=seed)
    strat = y_tmp if stratify else None
    X_tr, X_va, y_tr, y_va = train_test_split(X_tmp, y_tmp, test_size=val / (1 - test), stratify=strat,
                                              random_state=seed)
    return X_tr, X_va, X_te, y_tr, y_va, y_te


def diabetes_splits():
    df = clean_diabetes(load_diabetes_raw())
    X, y = df[DIAB_NUM + DIAB_CAT], df[DIAB_TARGET].to_numpy()
    return df, split_3way(X, y)


# ------------------------------------------------------------------ House price
def load_house_raw():
    return pd.read_csv(D / "house_price.csv")


def house_preprocessor():
    """Chuẩn hóa thuộc tính số; One-Hot cho 8 thuộc tính phân loại (bỏ cột đầu để tránh đa cộng tuyến hoàn hảo)."""
    return ColumnTransformer([
        ("num", StandardScaler(), HOUSE_NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False), HOUSE_CAT),
    ])


def house_splits():
    df = load_house_raw().drop_duplicates().reset_index(drop=True)
    X, y = df[HOUSE_NUM + HOUSE_CAT], df[HOUSE_TARGET].to_numpy(dtype=float)
    return df, split_3way(X, y, stratify=False)


# ------------------------------------------------------------------ Reviews
def load_reviews():
    """Ghép Title + Review Text thành một văn bản; bỏ dòng không có nội dung nhận xét."""
    df = pd.read_csv(D / "womens_reviews.csv", index_col=0)
    df = df.dropna(subset=["Review Text"]).copy()
    df["text"] = (df["Title"].fillna("") + ". " + df["Review Text"]).str.strip(". ").str.strip()
    df["n_words"] = df["text"].str.split().str.len()
    return df.reset_index(drop=True)


def reviews_splits():
    df = load_reviews()
    X, y = df["text"].to_numpy(), df["Recommended IND"].to_numpy()
    return df, split_3way(X, y)


def feature_names(pre):
    """Tên đặc trưng sau ColumnTransformer (bỏ tiền tố num__/cat__)."""
    return [n.split("__", 1)[1] for n in pre.get_feature_names_out()]


def as_array(x):
    return np.asarray(x)
