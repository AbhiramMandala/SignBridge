import numpy as np

from src.dataset import build
import src.dataset as D


def test_skips_unreadable_without_crashing(tmp_path, monkeypatch):
    raw = tmp_path / "raw";
    raw.mkdir()
    (raw / "A").mkdir()
    (raw / "A" / "bad.jpg").write_bytes(b"not an image")
    (raw / "A" / "empty.png").write_bytes(b"")
    proc = tmp_path / "processed";
    proc.mkdir()
    monkeypatch.setattr(D, "RAW_DIR", raw)
    monkeypatch.setattr(D, "LANDMARKS_CACHE", proc / "landmarks.npz")
    monkeypatch.setattr(D, "SKIPPED_LOG", proc / "skipped.json")
    monkeypatch.setattr(D, "extract_landmarks_bgr", lambda img: (_ for _ in ()).throw(AssertionError("no cv2 read")))
    skipped = build(limit_per_class=None)
    assert skipped["A"]["skipped"] == 2
    assert (proc / "landmarks.npz").exists()


def test_limit_per_class(tmp_path, monkeypatch):
    monkeypatch.setattr(D, "RAW_DIR", tmp_path)  # tmp has no subdirs -> empty labels
    monkeypatch.setattr(D, "LANDMARKS_CACHE", tmp_path / "l.npz")
    monkeypatch.setattr(D, "SKIPPED_LOG", tmp_path / "s.json")
    assert build(limit_per_class=5) == {}
