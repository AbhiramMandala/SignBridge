import numpy as np

from src.dataset import build, sample_files
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


def test_sample_files_capped_without_replacement_reproducible():
    from pathlib import Path
    files = [Path(f"img{i:04d}.jpg") for i in range(3000)]
    a, b = sample_files(files, 1000), sample_files(files, 1000)
    assert len(a) == 1000 and a == b and len(set(a)) == 1000
    assert set(a) != set(files[:1000])  # a spread sample, not the first-N slice


def test_sample_files_short_class_keeps_all():
    from pathlib import Path
    files = [Path(f"img{i}.jpg") for i in range(50)]
    assert sample_files(files, 1000) == files
    assert sample_files(files, None) == files
    assert sample_files(files, 0) == files
