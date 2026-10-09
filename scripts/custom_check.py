"""Sanity-check the trained model on YOUR webcam photos.

Reads data/custom_test/<label>/*.jpg, predicts each, reports accuracy.
This is reported SEPARATELY from dataset test metrics — never mixed in.

Usage: python scripts/custom_check.py  (run from signbridge/ dir)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import CUSTOM_TEST_DIR
from src.dataset import IMG_EXTS
from src.predict import load, predict_file


def main() -> dict:
    load()  # actionable FileNotFoundError if no trained model
    if not CUSTOM_TEST_DIR.is_dir():
        print(f"No {CUSTOM_TEST_DIR}/ yet. Drop labeled webcam photos as "
              f"{CUSTOM_TEST_DIR}/<LETTER>/*.jpg and rerun.")
        return {}
    _, labels = load()
    n_ok, n_total, per_image = 0, 0, []
    for label_dir in sorted(p for p in CUSTOM_TEST_DIR.iterdir() if p.is_dir()):
        for f in sorted(label_dir.iterdir()):
            if f.suffix.lower() not in IMG_EXTS:
                continue
            r = predict_file(str(f))
            if r["status"] != "ok":
                print(f"{f.name}: true={label_dir.name} -> {r['status']}")
                continue
            hit = r["letter"] == label_dir.name
            n_ok, n_total = n_ok + hit, n_total + 1
            per_image.append((f.name, label_dir.name, r["letter"], r["confidence"]))
            print(f"{f.name}: true={label_dir.name} pred={r['letter']} "
                  f"conf={r['confidence']:.2f} {'OK' if hit else 'MISS'}")
    if n_total:
        print(f"custom_test accuracy: {n_ok}/{n_total} = {n_ok / n_total:.2f} "
              f"(informal sanity check, NOT comparable to dataset test accuracy)")
    else:
        print("No valid test images found.")
    return {"n_ok": n_ok, "n_total": n_total}


if __name__ == "__main__":
    main()
