"""Pick the 'needs human review' confidence threshold from cross-validation on TRAIN,
then confirm on the test set (never tune on test)."""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from common import SEED, load_splits, make_model

train, test = load_splits()


def table(y, proba, classes, title):
    conf = proba.max(axis=1)
    right = classes[proba.argmax(axis=1)] == y.to_numpy()
    rows = []
    for t in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        flag = conf < t
        rows.append({
            "threshold": t,
            "flagged_%": round(flag.mean() * 100),
            "acc_flagged": round(float(right[flag].mean()), 2) if flag.any() else None,
            "acc_not_flagged": round(float(right[~flag].mean()), 2) if (~flag).any() else None,
        })
    print(f"\n{title}")
    print(pd.DataFrame(rows).to_string(index=False))


classes = np.array(sorted(train["label"].unique()))
model = make_model()
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
p = cross_val_predict(model, train["text"], train["label"], cv=cv, method="predict_proba")
table(train["label"], p, classes, "Train, 5-fold cross-validation (choose the threshold here)")

model.fit(train["text"], train["label"])
table(test["label"], model.predict_proba(test["text"]), classes,
      "Test set (confirm only, do not tune on this)")
print("\nA good threshold flags predictions that are often wrong (low acc_flagged)")
print("and leaves the rest accurate (high acc_not_flagged).")
print("Set it for the API with:  REVIEW_THRESHOLD=0.5 uvicorn api.main:app")