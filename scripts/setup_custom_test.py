"""Create data/custom_test/<LABEL>/ folders for independent webcam evaluation.

Creates missing folders only; never deletes or overwrites your photos.
Labels come from models/labels.json (the trained classes), falling back to
src/config.EXPECTED_LABELS when no trained model exists yet.

Usage: python scripts/setup_custom_test.py  (run from signbridge/ dir)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import CUSTOM_TEST_DIR, EXPECTED_LABELS, LABELS_PATH


def custom_labels() -> list:
    try:
        return [str(s) for s in json.loads(LABELS_PATH.read_text())]
    except (FileNotFoundError, ValueError):
        return [lab for lab in EXPECTED_LABELS if lab != "nothing"]


def main(base: Path | str = CUSTOM_TEST_DIR) -> list:
    labels = custom_labels()
    base = Path(base)
    made = []
    for lab in labels:
        d = base / lab
        if not d.is_dir():
            d.mkdir(parents=True, exist_ok=True)
            made.append(lab)
    print(f"custom_test at {base}/: {len(labels)} label folders, {len(made)} created.")
    print(f"Next: drop webcam photos as {base}/<LETTER>/*.jpg "
          f"(jpg/jpeg/png/webp), then run: python scripts/custom_check.py")
    return labels


if __name__ == "__main__":
    main()
