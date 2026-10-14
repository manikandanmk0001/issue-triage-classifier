import pandas as pd
from evaluate import evaluate

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

majority = train["label"].mode()[0]
print(f"Majority class in train: {majority}")
evaluate(test["label"], [majority] * len(test), name="Majority-class baseline")