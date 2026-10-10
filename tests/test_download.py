"""download_data.py: train-subtree filtering, label allowlist, copy counting.

Uses small synthetic archive trees in tmp dirs. No network, no Kaggle creds.
"""
import importlib.util
from pathlib import Path

import pytest


def _load_mod():
    p = Path(__file__).resolve().parent.parent / "scripts" / "download_data.py"
    spec = importlib.util.spec_from_file_location("download_data", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _tree(root: Path, layout: dict):
    for sub, names in layout.items():
        d = root / sub
        d.mkdir(parents=True, exist_ok=True)
        for n in names:
            (d / n).write_bytes(b"fake-jpg")


def test_only_train_subtree_and_known_labels():
    import tempfile
    mod = _load_mod()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _tree(root, {"asl_alphabet_train/A": ["a1.jpg", "a2.jpg"],
                     "asl_alphabet_train/B": ["b1.jpg"],
                     "asl_alphabet_train/junk": ["j.jpg"],   # unknown label: skip
                     "asl_alphabet_test/A": ["t1.jpg"]})      # test subtree: skip
        got = sorted((label, p.name)
                     for label, p in mod.iter_training_images(mod.find_training_root(root)))
        assert got == [("A", "a1.jpg"), ("A", "a2.jpg"), ("B", "b1.jpg")]


def test_missing_train_subtree_errors(tmp_path):
    mod = _load_mod()
    (tmp_path / "something_else").mkdir()
    with pytest.raises(FileNotFoundError, match="asl_alphabet_train"):
        mod.find_training_root(tmp_path)


def test_unexpected_folders_reported():
    import tempfile
    mod = _load_mod()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _tree(root, {"asl_alphabet_train/A": ["a1.jpg"],
                     "asl_alphabet_train/Space": ["s1.jpg"]})  # renamed label
        train_root = mod.find_training_root(root)
        assert mod.unexpected_folders(train_root) == ["Space"]
        got = sorted((label, p.name)
                     for label, p in mod.iter_training_images(train_root))
        assert got == [("A", "a1.jpg")]  # iter signature and filtering unchanged


def test_copy_respects_limit_and_resumes(tmp_path):
    import tempfile
    mod = _load_mod()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _tree(root, {"asl_alphabet_train/A": [f"a{i}.jpg" for i in range(5)]})
        raw = tmp_path / "raw"
        _tree(raw, {"A": ["old0.jpg", "old1.jpg", "old2.jpg"]})  # 3 already there
        n, counts = mod.copy_training_images(mod.find_training_root(root), raw, limit=4)
        assert n == 1 and counts == {"A": 4}  # topped up to the cap, not reset
        n2, counts2 = mod.copy_training_images(mod.find_training_root(root), raw, limit=0)
        assert n2 == 5 and counts2 == {"A": 8}  # no limit: all copied (a0.jpg overwritten)
