import json
import urllib.request

import joblib

from common import MODELS
from textprep import build_text

title = "KMeans crash with sparse input"
body = "### Steps/Code to Reproduce import numpy as np ### Actual Results ValueError"

# 1. Ask the running API
req = urllib.request.Request(
    "http://127.0.0.1:8000/predict",
    data=json.dumps({"title": title, "body": body}).encode(),
    headers={"Content-Type": "application/json"},
)
api = json.load(urllib.request.urlopen(req))

# 2. Ask the saved model directly
model = joblib.load(MODELS / "model.joblib")
proba = model.predict_proba([build_text(title, body)])[0]
direct = dict(zip(model.classes_, proba.round(3)))

print("API label:   ", api["label"], api["confidence"])
print("Direct label:", max(direct, key=direct.get), direct[max(direct, key=direct.get)])
same = api["label"] == max(direct, key=direct.get) and abs(api["confidence"] - max(direct.values())) < 0.001
print("OK: the API serves model.joblib" if same else "MISMATCH: investigate")