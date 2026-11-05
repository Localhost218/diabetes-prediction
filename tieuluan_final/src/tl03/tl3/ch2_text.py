"""Chương 2 — phân loại văn bản: chuẩn hóa chữ, TF-IDF (uni + bi-gram) và các mô hình tuyến tính/xác suất."""
import re

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

TEXT_NAMES = {"baseline": "Baseline (lớp đa số)", "logreg": "TF-IDF + Logistic Regression",
              "nb": "TF-IDF + Multinomial NB", "cnb": "TF-IDF + Complement NB", "svm": "TF-IDF + Linear SVM",
              "softmax": "Softmax Regression (5 lớp)"}


def normalize(text):
    """Chữ thường, bỏ số và ký tự đặc biệt, giữ dấu nháy trong từ phủ định (don't, isn't)."""
    text = text.lower()
    text = re.sub(r"[^a-z' ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tfidf(max_features=40000):
    """TF-IDF unigram + bigram, bỏ từ quá hiếm (min_df=3), log-scaling tần suất (sublinear_tf)."""
    return TfidfVectorizer(preprocessor=normalize, ngram_range=(1, 2), min_df=3, max_df=0.9,
                           sublinear_tf=True, max_features=max_features)


def text_models():
    """Các Pipeline TF-IDF -> mô hình; class_weight='balanced' bù lớp 'không giới thiệu' chiếm ~18%."""
    return {
        "baseline": Pipeline([("tfidf", tfidf(2000)), ("clf", DummyClassifier(strategy="most_frequent"))]),
        "logreg": Pipeline([("tfidf", tfidf()), ("clf", LogisticRegression(
            C=4.0, class_weight="balanced", max_iter=3000))]),
        "nb": Pipeline([("tfidf", tfidf()), ("clf", MultinomialNB(alpha=0.3))]),
        "cnb": Pipeline([("tfidf", tfidf()), ("clf", ComplementNB(alpha=0.3))]),
        "svm": Pipeline([("tfidf", tfidf()), ("clf", CalibratedClassifierCV(
            LinearSVC(C=0.5, class_weight="balanced"), cv=3))]),
        "softmax": Pipeline([("tfidf", tfidf()), ("clf", LogisticRegression(C=4.0, max_iter=3000))]),
    }


def top_terms(pipe, k=15):
    """k cụm từ có hệ số dương/âm lớn nhất của Logistic Regression = bằng chứng cho từng lớp."""
    vocab = np.array(pipe["tfidf"].get_feature_names_out())
    coef = pipe["clf"].coef_[0]
    order = np.argsort(coef)
    return {"neg": [(vocab[i], float(coef[i])) for i in order[:k]],
            "pos": [(vocab[i], float(coef[i])) for i in order[::-1][:k]]}
