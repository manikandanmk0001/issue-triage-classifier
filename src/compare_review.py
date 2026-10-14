import pandas as pd

mine = pd.read_csv("data/review.csv", dtype={"my_label": "string"})
key = pd.read_csv("data/review_key.csv")
df = mine.merge(key, on="id")
df["my_label"] = df["my_label"].fillna("").str.strip().str.lower()
df = df[df["my_label"] != ""]

if df.empty:
    raise SystemExit("No labels found in data/review.csv. Fill in my_label and save as CSV first.")

print(f"Reviewed: {len(df)}")
print(f"Marked unclear: {(df.my_label == 'unclear').sum()}")

clear = df[df.my_label != "unclear"]
agree = (clear.my_label == clear.label).mean()
print(f"\nAgreement with maintainer labels (excluding unclear): {agree:.1%}")

print("\nPer-class agreement:")
print(clear.assign(match=clear.my_label == clear.label)
           .groupby("label")["match"].mean().round(2))

print("\nMaintainer label (rows) vs my label (columns):")
print(pd.crosstab(clear.label, clear.my_label))

# Disagreements to read again later
clear[clear.my_label != clear.label][["id", "label", "my_label"]].to_csv(
    "data/disagreements.csv", index=False)
print("\nSaved disagreements to data/disagreements.csv")