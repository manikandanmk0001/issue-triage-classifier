import sys
import time

import joblib
import numpy as np
import onnxruntime as ort

from common import LABELS, MODELS, load_splits

_, test = load_splits()
texts = test["text"].tolist()

sk = joblib.load(MODELS / "model.joblib")
onnx_path = sys.argv[1] if len(sys.argv) > 1 else str(MODELS / "model.onnx")
session = ort.InferenceSession(onnx_path)

def onnx_proba(batch):
    arr = np.array(batch, dtype=object).reshape(-1, 1)
    _, probs = session.run(None, {"text": arr})
    return np.asarray(probs, dtype=float)


p_sk = sk.predict_proba(texts)
p_onnx = onnx_proba(texts)

assert list(sk.classes_) == LABELS, "class order changed"
same = (p_sk.argmax(axis=1) == p_onnx.argmax(axis=1))
print(f"Labels identical: {same.sum()} of {len(texts)}")
print(f"Max probability difference: {np.abs(p_sk - p_onnx).max():.6f}")

t0 = time.time()
for t in texts:
    sk.predict_proba([t])
sk_ms = (time.time() - t0) / len(texts) * 1000
t0 = time.time()
for t in texts:
    onnx_proba([t])
onnx_ms = (time.time() - t0) / len(texts) * 1000
print(f"ms per issue: scikit-learn {sk_ms:.2f}, ONNX {onnx_ms:.2f}")

if not same.all():
    print("FAIL: ONNX predictions differ from scikit-learn. Do not ship it.")
    sys.exit(1)
print("OK: ONNX matches scikit-learn on the test set.")