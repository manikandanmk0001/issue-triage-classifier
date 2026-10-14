import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

LABELS = ["bug", "documentation", "feature request", "other"]

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)


def evaluate(y_true, y_pred, name="model"):
    print(f"\n===== {name} =====")
    print(classification_report(y_true, y_pred, labels=LABELS,
                                digits=2, zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    print("Confusion matrix (rows = true, columns = predicted):")
    print(pd.DataFrame(cm, index=LABELS, columns=LABELS))