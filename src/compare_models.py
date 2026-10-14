import json
import time

import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.pipeline import make_pipeline

from evaluate import LABELS
from textprep import strip_template

raw = {i["id"]: i for i in json.load(open("data/scikit-learn_scikit-learn.json"))}
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
for df in (train, test):
    df["text_stripped"] = df["id"].map(lambda i: strip_template(raw[i]))

encoder = SentenceTransformer("all-MiniLM-L6-v2")
rows = []


def score(name, inp, pred, ms):
    rec = recall_score(test["label"], pred, labels=LABELS, average=None, zero_division=0)
    rows.append([name, inp, accuracy_score(test["label"], pred),
                 f1_score(test["label"], pred, average="macro"), *rec, ms])


for col, inp in [("text", "full text"), ("text_stripped", "stripped")]:
    # TF-IDF + logistic regression
    tfidf = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
        LogisticRegression(max_iter=1000, class_weight="balanced"))
    tfidf.fit(train[col], train["label"])
    t0 = time.time()
    pred = tfidf.predict(test[col])
    score("TF-IDF + LR", inp, pred, (time.time() - t0) / len(test) * 1000)

    # Embeddings + logistic regression
    X_train = encoder.encode(train[col].tolist(), batch_size=32)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(X_train, train["label"])
    t0 = time.time()
    X_test = encoder.encode(test[col].tolist(), batch_size=32)
    pred = clf.predict(X_test)
    score("Embeddings + LR", inp, pred, (time.time() - t0) / len(test) * 1000)

cols = ["model", "input", "accuracy", "macro_f1",
        *[f"recall_{l}" for l in LABELS], "ms_per_issue"]
table = pd.DataFrame(rows, columns=cols).round(3)
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", None)
print(table.to_string(index=False))
table.to_csv("data/results.csv", index=False)
print("\nSaved data/results.csv")