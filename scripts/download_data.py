"""Download the Kaggle ASL Alphabet dataset into data/raw/.

Requires YOUR Kaggle credentials (never commit them):
  1. kaggle.com -> Account -> Create New API Token -> saves kaggle.json
  2. Place it at %USERPROFILE%/.kaggle/kaggle.json  (chmod 600 on Linux/Mac)
  3. pip install kaggle && python scripts/download_data.py [--limit N]

Dataset: "ASL Alphabet" by grassknoted (Akash), https://www.kaggle.com/datasets/grassknoted/asl-alphabet
~87k images, 29 classes (A-Z + space/del/nothing). License: GPL 2 (Kaggle API
metadata, verified 2026-10-09). Do not redistribute the archive; keep data/ out of git).
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import EXPECTED_LABELS, RAW_DIR

DATASET = "grassknoted/asl-alphabet"
TRAIN_SUBDIR = "asl_alphabet_train"  # Kaggle test images live elsewhere; never train on them.


def find_training_root(archive_root: Path | str) -> Path:
    """Locate the training subtree inside the extracted archive.

    Raises FileNotFoundError (listing what was found) when the layout changed.
    """
    train_root = Path(archive_root) / TRAIN_SUBDIR
    if not train_root.is_dir():
        top = sorted(p.name for p in Path(archive_root).iterdir())
        raise FileNotFoundError(
            f"expected training subtree '{TRAIN_SUBDIR}/' under {archive_root} "
            f"(found top-level: {top}). Archive layout changed; update TRAIN_SUBDIR."
        )
    return train_root


def iter_training_images(train_root: Path | str):
    """Yield (label, path) for class images directly under the training subtree.

    Only folders named in EXPECTED_LABELS are accepted; anything else
    (test images, junk folders) is skipped. See unexpected_folders().
    """
    train_root = Path(train_root)
    for label_dir in sorted(p for p in train_root.iterdir() if p.is_dir()):
        if label_dir.name not in EXPECTED_LABELS:
            continue
        for img in sorted(label_dir.glob("*.jpg")):
            yield label_dir.name, img


def unexpected_folders(train_root: Path | str) -> list:
    """Direct child folders of the training root that are not known labels."""
    train_root = Path(train_root)
    return sorted(p.name for p in train_root.iterdir()
                  if p.is_dir() and p.name not in EXPECTED_LABELS)


def copy_training_images(train_root: Path | str, raw_dir: Path | str = RAW_DIR,
                         limit: int = 0) -> tuple[int, dict]:
    """Copy training images to data/raw/<label>/. Returns (n_copied, per-class counts).

    `limit` caps images per class. Destination folders are listed once up front
    (interrupted downloads resume) and counted in memory afterwards.
    """
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    counts = {d.name: sum(1 for _ in d.glob("*.jpg"))
              for d in raw_dir.iterdir() if d.is_dir()}
    n = 0
    for label, img in iter_training_images(train_root):
        if limit and counts.get(label, 0) >= limit:
            continue
        dest = raw_dir / label
        dest.mkdir(exist_ok=True)
        shutil.copy(img, dest / img.name)
        counts[label] = counts.get(label, 0) + 1
        n += 1
    on_disk = {d.name: sum(1 for _ in d.glob("*.jpg"))
               for d in sorted(raw_dir.iterdir()) if d.is_dir()}
    return n, on_disk


def main(limit: int = 0):
    if shutil.which("kaggle") is None:
        sys.exit("kaggle CLI not found. Run: pip install kaggle  (and place kaggle.json first)")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["kaggle", "datasets", "download", "-d", DATASET, "-p", tmp, "--unzip"],
                       check=True)
        train_root = find_training_root(tmp)
        n, counts = copy_training_images(train_root, RAW_DIR, limit)
        skipped_dirs = unexpected_folders(train_root)
        if skipped_dirs:
            print(f"skipped non-label folders (not in EXPECTED_LABELS): {skipped_dirs}")
    print(f"copied {n} images. per-class: {counts}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="max images per class (0 = all)")
    main(ap.parse_args().limit)
