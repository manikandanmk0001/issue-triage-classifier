import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from evaluate import evaluate

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

for name, cw in [("TF-IDF + LR", None),
                 ("TF-IDF + LR (balanced)", "balanced")]:
    model = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
        LogisticRegression(max_iter=1000, class_weight=cw),
    )
    model.fit(train["text"], train["label"])
    pred = model.predict(test["text"])
    evaluate(test["label"], pred, name=name)

# Top words per class from the last model (balanced).
# This is how we spot template leakage.
vec, clf = model[0], model[1]
names = np.array(vec.get_feature_names_out())
print("\nTop 10 words per class:")
for i, c in enumerate(clf.classes_):
    top = names[np.argsort(clf.coef_[i])[-10:][::-1]]
    print(f"{c:16s}: {', '.join(top)}")