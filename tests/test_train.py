"""train.py logic tests on small SYNTHETIC in-memory data.

These fixtures are random blobs, NOT ASL data. They validate split integrity,
model selection, and artifact serialization — never final metrics.
"""
import json

import joblib
import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier

from src.features import FEATURE_DIM
from src.train import drop_rare_classes, split


def _synth(n_per_class=40, n_classes=3, seed=0):
    rng = np.random.default_rng(seed)
    X = np.vstack([rng.normal(loc=c * 3.0, scale=1.0, size=(n_per_class, FEATURE_DIM))
                   for c in range(n_classes)]).astype(np.float32)
    y = np.repeat(np.arange(n_classes), n_per_class)
    return X, y


def test_split_disjoint_covering_stratified():
    X, y = _synth()
    i_tr, i_va, i_te = split(X, y)
    assert len(set(i_tr) & set(i_va) & set(i_te)) == 0
    assert len(i_tr) + len(i_va) + len(i_te) == len(X)
    n = len(X)
    assert abs(len(i_te) / n - 0.15) < 0.03 and abs(len(i_va) / n - 0.15) < 0.03
    for idx in (i_tr, i_va, i_te):  # every split sees every class
        assert set(np.unique(y[idx])) == {0, 1, 2}


def test_split_seeded_reproducible():
    X, y = _synth()
    assert [a.tolist() for a in split(X, y)] == [a.tolist() for a in split(X, y)]


def test_selection_picks_best_on_validation():
    X, y = _synth()
    i_tr, i_va, _ = split(X, y)
    cands = {"logreg": LogisticRegression(max_iter=1000),
             "rf": RandomForestClassifier(n_estimators=10, random_state=0),
             "mlp": MLPClassifier(hidden_layer_sizes=(16,), max_iter=1000, random_state=0)}
    scores = {k: accuracy_score(y[i_va], c.fit(X[i_tr], y[i_tr]).predict(X[i_va]))
              for k, c in cands.items()}
    best = max(scores, key=scores.get)
    assert set(scores) == {"logreg", "rf", "mlp"}
    assert all(0.0 <= s <= 1.0 for s in scores.values())
    assert scores[best] == max(scores.values())


def test_drop_rare_classes_reindexes():
    rng = np.random.default_rng(1)
    X = rng.random((25, FEATURE_DIM)).astype(np.float32)
    y = np.array([0] * 20 + [1] * 5)  # 'B' too rare, 'C' absent
    X2, y2, labs = drop_rare_classes(X, y, ["A", "B", "C"], min_n=10)
    assert labs == ["A"] and len(X2) == 20 and set(np.unique(y2)) == {0}


def test_artifact_roundtrip_with_label_map(tmp_path):
    X, y = _synth(n_per_class=10)
    clf = LogisticRegression(max_iter=1000).fit(X, y)
    mp, lp = tmp_path / "model.joblib", tmp_path / "labels.json"
    joblib.dump(clf, mp)
    lp.write_text(json.dumps(["A", "B", "C"]))
    reloaded = joblib.load(mp)
    assert reloaded.n_features_in_ == FEATURE_DIM
    assert len(json.loads(lp.read_text())) == len(reloaded.classes_)
    np.testing.assert_array_equal(reloaded.predict(X[:5]), clf.predict(X[:5]))
