import json

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from textprep import strip_template

raw = {i["id"]: i for i in json.load(open("data/scikit-learn_scikit-learn.json"))}
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
for df in (train, test):
    df["text_stripped"] = df["id"].map(lambda i: strip_template(raw[i]))

model = make_pipeline(
    TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
    LogisticRegression(max_iter=1000, class_weight="balanced"))
model.fit(train["text_stripped"], train["label"])

proba = model.predict_proba(test["text_stripped"])
test["pred"] = model.classes_[proba.argmax(axis=1)]
test["confidence"] = proba.max(axis=1).round(2)

errors = test[test["label"] != test["pred"]].copy()
print(f"{len(errors)} errors out of {len(test)}\n")
print("Most common mistakes (true -> predicted):")
print(errors.groupby(["label", "pred"]).size()
      .sort_values(ascending=False).to_string())

errors["title"] = errors["id"].map(lambda i: raw[i]["title"])
errors["url"] = errors["id"].map(
    lambda i: f"https://github.com/scikit-learn/scikit-learn/issues/{i}")
errors["preview"] = errors["text_stripped"].str[:500]
errors["category"] = ""          # you fill this in while reading

cols = ["id", "label", "pred", "confidence", "title", "url", "preview", "category"]
errors.sort_values("confidence", ascending=False)[cols].to_csv(
    "data/errors.csv", index=False)
print("\nSaved data/errors.csv (most confident mistakes first)")