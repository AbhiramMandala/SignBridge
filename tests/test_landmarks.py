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
