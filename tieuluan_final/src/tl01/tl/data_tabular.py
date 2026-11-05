"""Chương 2: ba tập dữ liệu bảng và văn bản, cùng một cách chia 80/20 cố định seed.

1. Diabetes Prediction (100.000 hồ sơ, Kaggle): phân loại nhị phân, lớp dương 8,5 %.
2. Vietnam Housing (30.229 tin đăng nhà ở Việt Nam): hồi quy giá bán (tỷ đồng).
3. Women's E-Commerce Clothing Reviews (22.641 nhận xét có nội dung): cảm xúc 3 lớp suy từ số sao.

Mỗi hàm load_* trả về dict có X_train, X_test, y_train, y_test đã tiền xử lý (bộ tiền xử lý chỉ
khớp trên tập huấn luyện) cùng đối tượng tiền xử lý để Web App dùng lại.
"""
import re

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .common import DATA, SEED

TAB = DATA / "tabular"
DIAB_NUM = ["age", "bmi", "HbA1c_level", "blood_glucose_level"]
DIAB_BIN = ["hypertension", "heart_disease"]
DIAB_CAT = ["gender", "smoking_history"]
HOUSE_NUM = ["log_area", "Frontage", "Access Road", "Floors", "Bedrooms", "Bathrooms"]
HOUSE_CAT = ["province", "district", "Legal status", "Furniture state", "House direction"]
SENT_NAMES = ["Tiêu cực", "Trung tính", "Tích cực"]
TFIDF_VOCAB = 5000


# ------------------------------------------------------------------------------------ Diabetes
def diabetes_frame():
    return pd.read_csv(TAB / "diabetes_prediction_dataset.csv")


def diabetes_preprocessor():
    return ColumnTransformer([
        ("num", StandardScaler(), DIAB_NUM),
        ("bin", "passthrough", DIAB_BIN),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), DIAB_CAT),
    ])


def load_diabetes():
    df = diabetes_frame()
    X, y = df.drop(columns="diabetes"), df.diabetes.to_numpy()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    pre = diabetes_preprocessor().fit(Xtr)
    return {"X_train": pre.transform(Xtr).astype(np.float32), "X_test": pre.transform(Xte).astype(np.float32),
            "y_train": ytr, "y_test": yte, "pre": pre, "raw_train": Xtr, "raw_test": Xte,
            "feature_names": list(pre.get_feature_names_out())}


# ------------------------------------------------------------------------------------ Housing
def _clean_place(s):
    return re.sub(r"[.\s]+$", "", s.strip())


def house_frame():
    """Tách tỉnh và quận/huyện từ địa chỉ; diện tích lệch phải nên dùng log(1 + m²)."""
    df = pd.read_csv(TAB / "vietnam_housing_dataset.csv")
    parts = df.Address.str.split(",")
    df["province"] = parts.str[-1].map(_clean_place)
    df["district"] = parts.str[-2].fillna("").map(_clean_place)
    df["log_area"] = np.log1p(df.Area)
    return df


def house_preprocessor():
    num = Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)), ("sc", StandardScaler())])
    cat = Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="Không rõ")),
                    ("oh", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=40,
                                         sparse_output=False))])
    return ColumnTransformer([("num", num, HOUSE_NUM), ("cat", cat, HOUSE_CAT)])


def load_house():
    df = house_frame()
    X, y = df[HOUSE_NUM + HOUSE_CAT], df.Price.to_numpy(np.float64)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED)
    pre = house_preprocessor().fit(Xtr)
    return {"X_train": pre.transform(Xtr).astype(np.float32), "X_test": pre.transform(Xte).astype(np.float32),
            "y_train": ytr, "y_test": yte, "pre": pre, "raw_train": Xtr, "raw_test": Xte}


# ------------------------------------------------------------------------------------ Reviews
def rating_to_sentiment(r):
    """1-2 sao: tiêu cực (0), 3 sao: trung tính (1), 4-5 sao: tích cực (2)."""
    return np.select([r <= 2, r == 3], [0, 1], 2)


def reviews_frame():
    df = pd.read_csv(TAB / "womens_ecommerce_reviews.csv").dropna(subset=["Review Text"])
    df["text"] = (df.Title.fillna("") + ". " + df["Review Text"]).str.strip(". ")
    df["sentiment"] = rating_to_sentiment(df.Rating.to_numpy())
    df["n_words"] = df["Review Text"].str.split().str.len()
    return df.reset_index(drop=True)


def clean_text(s):
    s = s.lower()
    s = re.sub(r"https?://\S+", " ", s)
    return re.sub(r"[^a-z' ]+", " ", s)


def tfidf_vectorizer():
    return TfidfVectorizer(preprocessor=clean_text, stop_words="english", ngram_range=(1, 2),
                           max_features=TFIDF_VOCAB, min_df=3, sublinear_tf=True)


def load_reviews():
    df = reviews_frame()
    Xtr, Xte, ytr, yte = train_test_split(df.text, df.sentiment.to_numpy(), test_size=0.2,
                                          stratify=df.sentiment, random_state=SEED)
    vec = tfidf_vectorizer().fit(Xtr)
    return {"X_train": vec.transform(Xtr).astype(np.float32), "X_test": vec.transform(Xte).astype(np.float32),
            "y_train": ytr, "y_test": yte, "pre": vec, "raw_train": Xtr, "raw_test": Xte}


LOADERS = {"diabetes": load_diabetes, "house": load_house, "reviews": load_reviews}
