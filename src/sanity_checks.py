"""Try to break the score: look for leakage before trusting it."""
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from common import SEED, load_splits, make_model

train, test = load_splits()
ok = True

# 1. Split really is in time order
tr_end = pd.to_datetime(train["created_at"], utc=True).max()
te_start = pd.to_datetime(test["created_at"], utc=True).min()
print(f"Train ends {tr_end:%Y-%m-%d}, test starts {te_start:%Y-%m-%d}")
if tr_end > te_start:
    print("FAIL: train and test overlap in time")
    ok = False

# 2. Same issue in both splits
shared_ids = set(train["id"]) & set(test["id"])
print("Issue ids in both splits:", len(shared_ids))
ok &= not shared_ids


def norm(s):
    return s.str.lower().str.replace(r"[^a-z0-9 ]", "", regex=True)


# 3. Exact duplicate texts across the split
exact = set(norm(train["text"])) & set(norm(test["text"]))
print("Exact duplicate texts across train/test:", len(exact))
ok &= not exact

# 4. Near-duplicates (same first 60 characters). Reported, not asserted:
#    bot-generated issues such as "CI failed on ..." show up here.
prefixes = set(norm(train["text"]).str[:60])
near = int(norm(test["text"]).str[:60].isin(prefixes).sum())
print(f"Test issues sharing their first 60 characters with a train issue: {near} of {len(test)}")

# 5. Shuffled-label test: random labels must score far below the real model
real = make_model().fit(train["text"], train["label"])
real_acc = accuracy_score(test["label"], real.predict(test["text"]))
rng = np.random.default_rng(SEED)
shuf = make_model().fit(train["text"], rng.permutation(train["label"].to_numpy()))
shuf_acc = accuracy_score(test["label"], shuf.predict(test["text"]))
print(f"Accuracy: real labels {real_acc:.2f}, shuffled training labels {shuf_acc:.2f}")
if shuf_acc > real_acc - 0.15:
    print("FAIL: shuffled labels score too close to real labels, suspect leakage")
    ok = False

if not ok:
    sys.exit(1)
print("Sanity checks passed.")