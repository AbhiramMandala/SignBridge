import numpy as np
import pytest

pytest.importorskip("mediapipe", reason="mediapipe not installed")

from src.landmarks import draw_landmarks_bgr, extract_landmarks_bgr


def test_blank_image_returns_none():
    blank = np.zeros((200, 200, 3), dtype=np.uint8)
    assert extract_landmarks_bgr(blank) is None


def test_fixture_hand_image():
    from pathlib import Path
    fixtures = sorted((Path(__file__).parent / "fixtures").glob("*.*"))
    if not fixtures:
        pytest.skip("no fixture hand image in tests/fixtures/")
    import cv2
    lm = extract_landmarks_bgr(cv2.imread(str(fixtures[0])))
    if lm is None:
        pytest.skip("MediaPipe found no hand in fixture image")
    assert lm.shape == (21, 3)


def test_draw_runs_without_mediapipe():
    rng = np.random.default_rng(0)
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    out = draw_landmarks_bgr(img, rng.random((21, 3)).astype(np.float32))
    assert out.shape == img.shape and out.sum() > 0


def test_detector_uses_configured_confidence(monkeypatch):
    from mediapipe.tasks.python import vision

    import src.landmarks as L
    from src.config import MIN_DETECTION_CONFIDENCE

    captured = {}

    def fake_create(opts):
        captured["conf"] = opts.min_hand_detection_confidence

        class Dummy:
            def close(self):
                pass

        return Dummy()

    monkeypatch.setattr(vision.HandLandmarker, "create_from_options", fake_create)
    L._landmarker = None
    try:
        L.get_landmarker()
        assert captured["conf"] == MIN_DETECTION_CONFIDENCE
        monkeypatch.setattr(L, "MIN_DETECTION_CONFIDENCE", 0.3)
        L.close()
        L._landmarker = None
        L.get_landmarker()
        assert captured["conf"] == 0.3  # a config change reaches the detector
    finally:
        L.close()
        L._landmarker = None


def test_detector_calls_serialized(monkeypatch):
    """Concurrent extracts never enter detect() together (mocked detector)."""
    import threading
    import time

    import src.landmarks as L

    active, peak = 0, 0
    peak_lock = threading.Lock()

    class FakeResult:
        hand_landmarks = []

    class FakeLandmarker:
        def detect(self, image):
            nonlocal active, peak
            with peak_lock:
                active += 1
                peak = max(peak, active)
            time.sleep(0.02)
            with peak_lock:
                active -= 1
            return FakeResult()

    monkeypatch.setattr(L, "get_landmarker", lambda: FakeLandmarker())
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    threads = [threading.Thread(target=lambda: L.extract_landmarks_bgr(img)) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert peak == 1
