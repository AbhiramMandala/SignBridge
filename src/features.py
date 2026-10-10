"""Normalize 21 hand landmarks -> 63-dim feature vector (pure NumPy)."""
from __future__ import annotations

import numpy as np

FEATURE_DIM = 63  # 21 landmarks * (x, y, z)


def normalize(landmarks: np.ndarray) -> np.ndarray:
    """Translate to wrist origin, scale by max wrist-distance, flatten.

    Translation + uniform-scale invariant; NOT rotation invariant
    (orientation carries letter meaning in ASL, e.g. D vs G-ish tilts).
    """
    lm = np.asarray(landmarks, dtype=np.float64).reshape(21, 3)
    if not np.isfinite(lm).all():
        raise ValueError("landmarks contain NaN/inf")
    lm = lm - lm[0:1]  # wrist origin
    scale = np.linalg.norm(lm, axis=1).max()
    if scale > 1e-9:
        lm = lm / scale
    return lm.reshape(-1).astype(np.float32)


def mirror_x(landmarks: np.ndarray) -> np.ndarray:
    """Mirror x coords (left<->right hand). Call BEFORE normalize."""
    lm = np.asarray(landmarks).reshape(21, 3).copy()
    lm[:, 0] = 1.0 - lm[:, 0]
    return lm


def augment(lm_flat: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Rotation/scale/shift jitter + joint noise on TRAIN split only. Input/output: (63,)."""
    lm = lm_flat.reshape(21, 3).copy()
    theta = rng.uniform(-0.26, 0.26)  # ~+/-15 deg in image plane
    c, s = np.cos(theta), np.sin(theta)
    lm[:, :2] = lm[:, :2] @ np.array([[c, -s], [s, c]])
    lm = lm * rng.uniform(0.85, 1.15)
    lm[:, :2] = lm[:, :2] + rng.uniform(-0.05, 0.05, size=(1, 2))
    lm = lm + rng.normal(0.0, 0.01, size=lm.shape)  # MediaPipe detector jitter
    return lm.reshape(-1).astype(np.float32)
