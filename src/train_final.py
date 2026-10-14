import hashlib
import json
import platform
from datetime import datetime, timezone

import joblib
import sklearn
from sklearn.metrics import accuracy_score, f1_score, recall_score

from common import DATA, LABELS, MODELS, SEED, load_splits, make_model

train, test = load_splits()
model = make_model()
model.fit(train["text"], train["label"])
pred = model.predict(test["text"])

raw = DATA / "scikit-learn_scikit-learn.json"
recalls = recall_score(test["label"], pred, labels=LABELS, average=None, zero_division=0)
meta = {
    "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "model": "TF-IDF (1-2 grams, min_df=2) + LogisticRegression(balanced)",
    "seed": SEED,
    "sklearn_version": sklearn.__version__,
    "python_version": platform.python_version(),
    "raw_data_sha256_12": hashlib.sha256(raw.read_bytes()).hexdigest()[:12],
    "train_size": len(train),
    "test_size": len(test),
    "train_date_range": [str(train["created_at"].min()), str(train["created_at"].max())],
    "test_accuracy": round(accuracy_score(test["label"], pred), 3),
    "test_macro_f1": round(f1_score(test["label"], pred, average="macro"), 3),
    "test_recall": {l: round(float(r), 3) for l, r in zip(LABELS, recalls)},
}

MODELS.mkdir(exist_ok=True)
joblib.dump(model, MODELS / "model.joblib")
(MODELS / "model_meta.json").write_text(json.dumps(meta, indent=2))
print(json.dumps(meta, indent=2))