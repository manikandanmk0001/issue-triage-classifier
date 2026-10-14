import pytest
from fastapi.testclient import TestClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

import api.main as main

LABELS = {"bug", "documentation", "feature request", "other"}
TEXTS = ["crash error traceback fails", "exception raised wrong result",
         "docs typo documentation page", "docstring unclear documentation example",
         "add new feature proposed solution", "would be nice to support option",
         "ci build wheel failing", "rfc deprecate discussion proposal"]
Y = ["bug", "bug", "documentation", "documentation",
     "feature request", "feature request", "other", "other"]


@pytest.fixture
def client(monkeypatch):
    tiny = make_pipeline(TfidfVectorizer(), LogisticRegression(max_iter=500))
    tiny.fit(TEXTS, Y)
    monkeypatch.setattr(main, "_model", tiny)
    return TestClient(main.app)


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_predict_shape(client):
    r = client.post("/predict", json={"title": "crash error", "body": "traceback fails"})
    assert r.status_code == 200
    d = r.json()
    assert d["label"] in LABELS
    assert 0 <= d["confidence"] <= 1
    assert set(d["probabilities"]) == LABELS
    assert abs(sum(d["probabilities"].values()) - 1) < 0.01


def test_predict_obvious_bug(client):
    r = client.post("/predict", json={"title": "crash error traceback fails"})
    assert r.json()["label"] == "bug"


def test_empty_title_rejected(client):
    assert client.post("/predict", json={"title": "", "body": "x"}).status_code == 422


def test_oversized_body_rejected(client):
    r = client.post("/predict", json={"title": "x", "body": "a" * 20001})
    assert r.status_code == 422


def test_missing_model_gives_503(monkeypatch):
    monkeypatch.setattr(main, "_model", None)
    monkeypatch.setattr(main, "MODEL_PATH", main.ROOT / "models" / "does_not_exist.joblib")
    r = TestClient(main.app).post("/predict", json={"title": "x"})
    assert r.status_code == 503


def test_build_text_collapses_whitespace():
    assert main.build_text("A  title", "line1\n\n  line2") == "A title line1 line2"


def test_home_page(client):
    r = client.get("/")
    assert r.status_code == 200 and "Issue triage" in r.text