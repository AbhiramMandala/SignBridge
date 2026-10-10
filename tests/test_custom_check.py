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
        assert mod.main()["n_total"] == 0
        assert "setup_custom_test" in capsys.readouterr().out
    finally:
        P._model, P._labels = None, None


def _load_setup():
    p = Path(__file__).resolve().parent.parent / "scripts" / "setup_custom_test.py"
    spec = importlib.util.spec_from_file_location("setup_custom_test", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _canned(monkeypatch, mod, labels, mapping):
    """Stub the model interface: filename -> (letter, conf) or a skip status."""
    monkeypatch.setattr(mod, "load", lambda: (None, labels))

    def fake(path):
        v = mapping.get(Path(path).name)
        if v is None:
            return {"letter": None, "confidence": 0.0, "top3": [],
                    "annotated": None, "status": "unreadable"}
        if isinstance(v, str):
            return {"letter": None, "confidence": 0.0, "top3": [],
                    "annotated": None, "status": v}
        lab, conf = v
        return {"letter": lab, "confidence": conf, "top3": [(lab, conf)],
                "annotated": None, "status": "ok"}

    monkeypatch.setattr(mod, "predict_file", fake)


def _photos(root: Path, layout: dict):
    for sub, names in layout.items():
        d = root / sub
        d.mkdir(parents=True, exist_ok=True)
        for n in names:
            (d / n).write_bytes(b"img")


def test_scores_only_known_labels(tmp_path, monkeypatch, capsys):
    mod = _load_mod()
    base = tmp_path / "custom_test"
    _photos(base, {"A": ["a1.jpg", "a2.jpg", "a3.jpg", "note.txt", "garbage.jpg"],
                   "Q": ["q1.jpg"],
                   "frobnicate": ["z1.jpg"]})  # unknown label folder
    monkeypatch.setattr(mod, "CUSTOM_TEST_DIR", base)
    _canned(monkeypatch, mod, ["A", "B", "Q"],
            {"a1.jpg": ("A", 0.9), "a2.jpg": ("A", 0.8), "a3.jpg": ("B", 0.7),
             "q1.jpg": "no_hand"})
    r = mod.main()
    assert (r["n_ok"], r["n_total"], r["n_discovered"]) == (2, 3, 5)
    assert r["skipped"] == {"no_hand": 1, "unreadable": 1, "unknown_label": 1}
    assert r["per_class"] == {"A": [2, 3]}
    out = capsys.readouterr().out
    assert "2/3" in out and "unknown label" in out


def test_empty_dir_no_division(tmp_path, monkeypatch, capsys):
    mod = _load_mod()
    base = tmp_path / "custom_test"
    base.mkdir()
    (base / "A").mkdir()  # empty known folder
    monkeypatch.setattr(mod, "CUSTOM_TEST_DIR", base)
    _canned(monkeypatch, mod, ["A", "B"], {})
    r = mod.main()
    assert (r["n_ok"], r["n_total"], r["n_discovered"]) == (0, 0, 0)
    assert "No valid test images" in capsys.readouterr().out


def test_setup_creates_folders_without_overwriting(tmp_path):
    setup = _load_setup()
    base = tmp_path / "custom_test"
    labels = setup.main(base)
    assert len(labels) == 28 and all((base / lab).is_dir() for lab in labels)
    (base / labels[0] / "keep.jpg").write_bytes(b"mine")
    setup.main(base)  # rerun must not delete existing photos
    assert (base / labels[0] / "keep.jpg").read_bytes() == b"mine"


def test_loose_images_warned_not_scored(tmp_path, monkeypatch, capsys):
    mod = _load_mod()
    base = tmp_path / "custom_test"
    _photos(base, {"A": ["a1.jpg"]})
    (base / "stray.jpg").write_bytes(b"img")  # misplaced: no label folder
    (base / "notes.txt").write_bytes(b"txt")  # not an image: ignored silently
    monkeypatch.setattr(mod, "CUSTOM_TEST_DIR", base)
    _canned(monkeypatch, mod, ["A", "B"], {"a1.jpg": ("A", 0.9)})
    r = mod.main()
    assert (r["n_ok"], r["n_total"], r["n_discovered"]) == (1, 1, 1)
    out = capsys.readouterr().out
    assert "directly under custom_test" in out and "stray.jpg" in out
