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

DATASET = "grassknoted/asl-alphabet"
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def main(limit: int = 0):
    if shutil.which("kaggle") is None:
        sys.exit("kaggle CLI not found. Run: pip install kaggle  (and place kaggle.json first)")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["kaggle", "datasets", "download", "-d", DATASET, "-p", tmp, "--unzip"],
                       check=True)
        # Flatten: archive nests under asl_alphabet_train/<label>/*.jpg
        n = 0
        for img in sorted(Path(tmp).rglob("*.jpg")):
            label = img.parent.name
            dest = RAW_DIR / label
            dest.mkdir(exist_ok=True)
            if limit and len(list(dest.glob("*.jpg"))) >= limit:
                continue
            shutil.copy(img, dest / img.name)
            n += 1
    counts = {d.name: len(list(d.glob("*.jpg"))) for d in sorted(RAW_DIR.iterdir()) if d.is_dir()}
    print(f"copied {n} images. per-class: {counts}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="max images per class (0 = all)")
    main(ap.parse_args().limit)
