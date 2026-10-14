import json
import re
from textprep import strip_template

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.pipeline import make_pipeline

from evaluate import LABELS, evaluate

raw = {i["id"]: i for i in json.load(open("data/scikit-learn_scikit-learn.json"))}




train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
for df in (train, test):
    df["text_stripped"] = df["id"].map(lambda i: strip_template(raw[i]))

summary = []
for col, name in [("text", "full text"), ("text_stripped", "template stripped")]:
    model = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
        LogisticRegression(max_iter=1000, class_weight="balanced"),
    )
    model.fit(train[col], train["label"])
    pred = model.predict(test[col])
    evaluate(test["label"], pred, name=f"Balanced LR, {name}")
    rec = recall_score(test["label"], pred, labels=LABELS, average=None, zero_division=0)
    summary.append([name, accuracy_score(test["label"], pred),
                    f1_score(test["label"], pred, average="macro"), *rec])

print("\nSummary")
print(pd.DataFrame(summary, columns=["input", "accuracy", "macro_f1",
                                     *[f"recall_{l}" for l in LABELS]]).round(2).to_string(index=False))

vec, clf = model[0], model[1]
names = np.array(vec.get_feature_names_out())
print("\nTop 10 words per class (stripped):")
for i, c in enumerate(clf.classes_):
    print(f"{c:16s}: {', '.join(names[np.argsort(clf.coef_[i])[-10:][::-1]])}")