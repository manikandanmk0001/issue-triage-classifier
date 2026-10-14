import re

import pandas as pd

LABEL_MAP = {
    "bug": "bug", "regression": "bug",
    "new feature": "feature request", "enhancement": "feature request",
    "documentation": "documentation",
    "build / ci": "other", "performance": "other", "rfc": "other",
    "array api": "other", "cython": "other", "api": "other",
}

df = pd.read_json("data/scikit-learn_scikit-learn.json")


def classes_for(labels):
    low = [l.lower() for l in labels]
    if "spam" in low:
        return "spam"
    found = {LABEL_MAP[l] for l in low if l in LABEL_MAP}
    if not found:
        return "unlabeled"
    if len(found) > 1:
        return "ambiguous"
    return found.pop()


df["label"] = df["labels"].apply(classes_for)
print("Before filtering:")
print(df["label"].value_counts(), "\n")

# Keep only issues with one clear class
df = df[df["label"].isin(
    ["bug", "feature request", "documentation", "other"])].copy()

# Combine title + body, collapse whitespace
df["text"] = (df["title"] + "\n" + df["body"]).str.replace(r"\s+", " ", regex=True).str.strip()

# Remove exact duplicates (normalized text)
df["norm"] = df["text"].str.lower().apply(lambda t: re.sub(r"[^a-z0-9 ]", "", t))
before = len(df)
df = df.drop_duplicates(subset="norm")
print(f"Removed {before - len(df)} exact duplicates")

# Time-based split: oldest 80% train, newest 20% test
df = df.sort_values("created_at").reset_index(drop=True)
cut = int(len(df) * 0.8)
train, test = df.iloc[:cut], df.iloc[cut:]

cols = ["id", "created_at", "text", "label"]
train[cols].to_csv("data/train.csv", index=False)
test[cols].to_csv("data/test.csv", index=False)

print(f"\nTrain: {len(train)} ({train.created_at.min():%Y-%m-%d} to {train.created_at.max():%Y-%m-%d})")
print(f"Test:  {len(test)} ({test.created_at.min():%Y-%m-%d} to {test.created_at.max():%Y-%m-%d})")
print("\nTrain label counts:\n", train["label"].value_counts())
print("\nTest label counts:\n", test["label"].value_counts())