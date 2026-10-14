import json
import time

import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score

from evaluate import LABELS, evaluate
from textprep import strip_template

raw = {i["id"]: i for i in json.load(open("data/scikit-learn_scikit-learn.json"))}
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
for df in (train, test):
    df["text_stripped"] = df["id"].map(lambda i: strip_template(raw[i]))

encoder = SentenceTransformer("all-MiniLM-L6-v2")   # small, runs on CPU

summary = []
for col, name in [("text", "full text"), ("text_stripped", "template stripped")]:
    X_train = encoder.encode(train[col].tolist(), batch_size=32, show_progress_bar=True)

    t0 = time.time()
    X_test = encoder.encode(test[col].tolist(), batch_size=32)
    ms_per_issue = (time.time() - t0) / len(test) * 1000

    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(X_train, train["label"])
    pred = clf.predict(X_test)

    evaluate(test["label"], pred, name=f"Embeddings + LR, {name}")
    rec = recall_score(test["label"], pred, labels=LABELS, average=None, zero_division=0)
    summary.append([name, accuracy_score(test["label"], pred),
                    f1_score(test["label"], pred, average="macro"),
                    *rec, ms_per_issue])

print("\nSummary")
print(pd.DataFrame(summary, columns=["input", "accuracy", "macro_f1",
                                     *[f"recall_{l}" for l in LABELS],
                                     "ms_per_issue"]).round(2).to_string(index=False))