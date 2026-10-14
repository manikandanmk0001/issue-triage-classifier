from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MODELS = ROOT / "models"
LABELS = ["bug", "documentation", "feature request", "other"]
SEED = 42


def load_splits():
    return pd.read_csv(DATA / "train.csv"), pd.read_csv(DATA / "test.csv")


def make_model(seed=SEED):
    return make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed),
    )