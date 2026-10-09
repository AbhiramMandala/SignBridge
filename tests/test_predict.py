"""predict.py tests with a tiny synthetic-data model (no dataset needed)."""
import json

import joblib
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

import src.predict as P
from src.features import normalize


def _fake_hand(center, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.random((21, 3)) * 0.2 + np.array(center)).astype(np.float32)


@pytest.fixture()
def toy_model(tmp_path, monkeypatch):
    X = np.array([normalize(_fake_hand((c, 0.3, 0), s))
                  for c in (0.1, 0.6) for s in range(30)])
    y = np.array([0] * 30 + [1] * 30)
    clf = LogisticRegression(max_iter=1000).fit(X, y)
    mp, lp = tmp_path / "model.joblib", tmp_path / "labels.json"
    joblib.dump(clf, mp)
    lp.write_text(json.dumps(["A", "B"]))
    monkeypatch.setattr(P, "MODEL_PATH", mp)
    monkeypatch.setattr(P, "LABELS_PATH", lp)
    P._model, P._labels = None, None
    yield clf
    P._model, P._labels = None, None


def test_predict_valid_label_and_confidence(monkeypatch, toy_model):
    monkeypatch.setattr(P, "extract_landmarks_bgr", lambda img: _fake_hand((0.1, 0.3, 0)))
    r = P.predict_array(np.zeros((200, 200, 3), np.uint8))
    assert r["status"] == "ok" and r["letter"] in ("A", "B", "uncertain")
    assert 0.0 <= r["confidence"] <= 1.0
    assert len(r["top3"]) == 2 and abs(sum(p for _, p in r["top3"]) - 1.0) < 1e-5


def test_predict_none_landmarks(monkeypatch, toy_model):
    monkeypatch.setattr(P, "extract_landmarks_bgr", lambda img: None)
    r = P.predict_array(np.zeros((200, 200, 3), np.uint8))
    assert r["status"] == "no_hand" and r["letter"] is None


def test_missing_model_actionable(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "MODEL_PATH", tmp_path / "nope.joblib")
    P._model, P._labels = None, None
    try:
        with pytest.raises(FileNotFoundError, match="python -m src.train"):
            P.load()
    finally:
        P._model, P._labels = None, None


def test_incompatible_artifacts_rejected(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    bad = LogisticRegression(max_iter=500).fit(rng.random((20, 10)), [0, 1] * 10)
    mp, lp = tmp_path / "m.joblib", tmp_path / "l.json"
    joblib.dump(bad, mp)
    lp.write_text(json.dumps(["A", "B"]))
    monkeypatch.setattr(P, "MODEL_PATH", mp)
    monkeypatch.setattr(P, "LABELS_PATH", lp)
    P._model, P._labels = None, None
    try:
        with pytest.raises(ValueError, match="63"):
            P.load()
    finally:
        P._model, P._labels = None, None


def test_predict_unreadable_file(tmp_path):
    r = P.predict_file(str(tmp_path / "missing.jpg"))
    assert r["status"] == "unreadable"


def test_low_confidence_gives_uncertain(monkeypatch, toy_model):
    monkeypatch.setattr(P, "extract_landmarks_bgr",
                        lambda img: np.full((21, 3), 0.5, np.float32))
    r = P.predict_array(np.zeros((200, 200, 3), np.uint8), threshold=0.9999)
    assert r["letter"] == "uncertain"
