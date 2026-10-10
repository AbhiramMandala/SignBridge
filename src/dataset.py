"""Walk data/raw/<label>/*.jpg -> landmarks -> data/processed/landmarks.npz.

Usage: python -m src.dataset [--limit N]  (run from signbridge/ dir)
Skips (counts + logs, never crashes): unreadable files, no-hand frames.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from .config import LANDMARKS_CACHE, MAX_PER_CLASS, RANDOM_SEED, RAW_DIR, SKIPPED_LOG
from .features import normalize
from .landmarks import close, extract_landmarks_bgr

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def sample_files(files: list, limit_per_class: int | None, seed: int = RANDOM_SEED) -> list:
    """Deterministic per-class cap: seeded sample without replacement.

    Returns every file when the limit is None/0 or the class is smaller than
    the limit. Same seed + same file list = same selection, every run.
    """
    files = list(files)
    if not limit_per_class or len(files) <= limit_per_class:
        return files
    rng = np.random.default_rng(seed)
    chosen = sorted(rng.choice(len(files), size=limit_per_class, replace=False))
    return [files[i] for i in chosen]


def build(limit_per_class: int | None = MAX_PER_CLASS) -> dict:
    if not RAW_DIR.is_dir():
        sys.exit(f"No dataset found at {RAW_DIR}/. See data/README.md to download it.")
    X, y, labels, skipped = [], [], [], {}
    for label_dir in sorted(p for p in RAW_DIR.iterdir() if p.is_dir()):
        label = label_dir.name
        files = sorted(f for f in label_dir.iterdir() if f.suffix.lower() in IMG_EXTS)
        files = sample_files(files, limit_per_class)
        labels.append(label)
        idx = len(labels) - 1
        n_skip = 0
        for f in files:
            img = cv2.imread(str(f))
            if img is None:
                n_skip += 1
                continue
            try:
                lm = extract_landmarks_bgr(img)
            except Exception:
                n_skip += 1
                continue
            if lm is None:
                n_skip += 1
                continue
            X.append(normalize(lm))
            y.append(idx)
        skipped[label] = {"total": len(files), "skipped": n_skip}
    close()
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    counts = np.array([int((y == i).sum()) for i in range(len(labels))], dtype=np.int64)
    LANDMARKS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(LANDMARKS_CACHE, X=X, y=y, labels=np.array(labels), counts=counts)
    SKIPPED_LOG.write_text(json.dumps(skipped, indent=2))
    total = sum(v["total"] for v in skipped.values())
    nskip = sum(v["skipped"] for v in skipped.values())
    print(f"kept={len(X)} skipped={nskip}/{total} ({100*nskip/max(total,1):.1f}%) -> {LANDMARKS_CACHE}")
    return skipped


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=MAX_PER_CLASS)
    args = ap.parse_args()
    build(args.limit if args.limit > 0 else None)
