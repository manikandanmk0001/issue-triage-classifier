import pandas as pd

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
df = pd.concat([train, test])

# 40 per class = 160 issues, fixed seed so it's reproducible
sample = (pd.concat([g.sample(40, random_state=42) for _, g in df.groupby("label")])
            .sample(frac=1, random_state=1)        # shuffle
            .reset_index(drop=True))

# Key with the maintainer labels (don't open this one yet)
sample[["id", "label"]].to_csv("data/review_key.csv", index=False)

# Blind file you will fill in
blind = pd.DataFrame({
    "id": sample["id"],
    "text_preview": sample["text"].str[:700],
    "my_label": "",
})
blind.to_csv("data/review.csv", index=False)
print("Wrote data/review.csv (fill in my_label) and data/review_key.csv (don't peek)")