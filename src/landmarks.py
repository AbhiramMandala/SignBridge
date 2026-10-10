"""Hand landmark extraction via MediaPipe Tasks HandLandmarker.

Image (BGR) -> (21, 3) xyz array, or None when no hand is detected.
Left hands are mirrored to a canonical right-hand orientation.
Model bundle (~9 MB) auto-downloads once to models/hand_landmarker.task;
override with SIGNBRIDGE_HAND_MODEL. That file is gitignored.
"""
from __future__ import annotations

import os
import threading
import urllib.request
from pathlib import Path

import numpy as np

from .config import MIN_DETECTION_CONFIDENCE
from .features import mirror_x

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/1/hand_landmarker.task")
MODEL_PATH = Path(os.environ.get("SIGNBRIDGE_HAND_MODEL",
                                 Path(__file__).resolve().parent.parent / "models" / "hand_landmarker.task"))

_landmarker = None
_detect_lock = threading.Lock()  # one shared detector; concurrent Gradio clicks must not enter detect() together


def ensure_model() -> Path:
    if not MODEL_PATH.exists():
        print(f"downloading hand_landmarker.task -> {MODEL_PATH} ...")
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return MODEL_PATH


def get_landmarker():
    """Singleton HandLandmarker (IMAGE running mode, max 1 hand)."""
    global _landmarker
    if _landmarker is None:
        import mediapipe as mp
        from mediapipe.tasks.python import vision
        opts = vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(ensure_model())),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=1,  # multiple hands: first/most-confident only (see README)
            min_hand_detection_confidence=MIN_DETECTION_CONFIDENCE,
        )
        _landmarker = vision.HandLandmarker.create_from_options(opts)
    return _landmarker


def extract_landmarks_bgr(image_bgr: np.ndarray) -> np.ndarray | None:
    import cv2
    import mediapipe as mp

    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    with _detect_lock:
        result = get_landmarker().detect(mp_image)
    if not result.hand_landmarks:
        return None
    lm = np.array([[p.x, p.y, p.z] for p in result.hand_landmarks[0]], dtype=np.float32)
    if result.handedness and result.handedness[0][0].category_name == "Left":
        lm = mirror_x(lm)  # canonicalize to right-hand orientation
    return lm


def draw_landmarks_bgr(image_bgr: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    """Draw 21 joint dots + finger skeleton. Pure OpenCV, no MediaPipe needed."""
    import cv2

    out = image_bgr.copy()
    h, w = out.shape[:2]
    pts = [(int(x * w), int(y * h)) for x, y, _ in landmarks]
    bones = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
             (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
             (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]
    for a, b in bones:
        cv2.line(out, pts[a], pts[b], (0, 255, 0), 2)
    for p in pts:
        cv2.circle(out, p, 4, (0, 0, 255), -1)
    return out


def close():
    global _landmarker
    if _landmarker is not None:
        _landmarker.close()
        _landmarker = None
