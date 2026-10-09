"""custom_check.py: missing-model error + missing-dir guidance (no dataset needed)."""
import importlib.util
import json
from pathlib import Path

import joblib
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

import src.predict as P
from src.features import normalize


def _load_mod():
    p = Path(__file__).resolve().parent.parent / "scripts" / "custom_check.py"
    spec = importlib.util.spec_from_file_location("custom_check", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_no_model_actionable(tmp_path, monkeypatch):
    mod = _load_mod()
    monkeypatch.setattr(P, "MODEL_PATH", tmp_path / "nope.joblib")
    monkeypatch.setattr(mod, "CUSTOM_TEST_DIR", tmp_path / "empty")
    P._model, P._labels = None, None
    try:
        with pytest.raises(FileNotFoundError, match="python -m src.train"):
            mod.main()
    finally:
        P._model, P._labels = None, None


def test_missing_dir_guidance_with_model(tmp_path, monkeypatch, capsys):
    rng = np.random.default_rng(0)
    clf = LogisticRegression(max_iter=500).fit(rng.random((40, 63)), [0, 1] * 20)
    mp, lp = tmp_path / "m.joblib", tmp_path / "l.json"
    joblib.dump(clf, mp)
    lp.write_text(json.dumps(["A", "B"]))
    monkeypatch.setattr(P, "MODEL_PATH", mp)
    monkeypatch.setattr(P, "LABELS_PATH", lp)
    mod = _load_mod()
    monkeypatch.setattr(mod, "CUSTOM_TEST_DIR", tmp_path / "nodir")
    P._model, P._labels = None, None
    try:
        assert mod.main() == {}
        assert "Drop labeled webcam photos" in capsys.readouterr().out
    finally:
        P._model, P._labels = None, None
