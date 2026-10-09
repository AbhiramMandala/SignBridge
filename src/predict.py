"""Load saved model -> predict letter from an image file/array.

Confidence is the raw predict_proba max (uncalibrated); the app treats it as
a rejection threshold, not a calibrated probability.
"""
from __future__ import annotations

import json

import cv2
import joblib
import numpy as np

from .config import CONFIDENCE_THRESHOLD, LABELS_PATH, MODEL_PATH
from .features import FEATURE_DIM, normalize
from .landmarks import draw_landmarks_bgr, extract_landmarks_bgr

_model, _labels = None, None


def load():
    global _model, _labels
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"No trained model at {MODEL_PATH}. Train one with:\n"
                "  python -m src.dataset && python -m src.train"
            )
        _model = joblib.load(MODEL_PATH)
        _labels = json.loads(LABELS_PATH.read_text())
        n_feat = getattr(_model, "n_features_in_", FEATURE_DIM)
        n_cls = len(getattr(_model, "classes_", _labels))
        if n_feat != FEATURE_DIM:
            _model = None
            raise ValueError(
                f"Model expects {n_feat} features, pipeline produces {FEATURE_DIM}. "
                "Retrain with: python -m src.train"
            )
        if len(_labels) != n_cls:
            _model = None
            raise ValueError(
                f"labels.json has {len(_labels)} entries but model has {n_cls} classes. "
                "Retrain with: python -m src.train"
            )
    return _model, _labels


def predict_array(image_bgr: np.ndarray, threshold: float = CONFIDENCE_THRESHOLD) -> dict:
    """-> {letter, confidence, top3, annotated, status} ; never raises on no-hand."""
    lm = extract_landmarks_bgr(image_bgr)
    if lm is None:
        return {"letter": None, "confidence": 0.0, "top3": [],
                "annotated": image_bgr, "status": "no_hand"}
    model, labels = load()
    proba = model.predict_proba(normalize(lm).reshape(1, -1))[0]
    top = np.argsort(proba)[::-1][:3]
    top3 = [(labels[i], float(proba[i])) for i in top]
    letter, conf = top3[0]
    if conf < threshold:
        letter = "uncertain"
    return {"letter": letter, "confidence": conf, "top3": top3,
            "annotated": draw_landmarks_bgr(image_bgr, lm), "status": "ok"}


def predict_file(path: str, threshold: float = CONFIDENCE_THRESHOLD) -> dict:
    img = cv2.imread(path)
    if img is None:
        return {"letter": None, "confidence": 0.0, "top3": [],
                "annotated": None, "status": "unreadable"}
    return predict_array(img, threshold)
