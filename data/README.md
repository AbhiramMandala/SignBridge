# Getting the dataset (data itself is gitignored)

Primary dataset: **Kaggle "ASL Alphabet"** by grassknoted (Akash)
- URL: https://www.kaggle.com/datasets/grassknoted/asl-alphabet
- ~87,000 images, 29 classes (`A`–`Z` + `space`, `del`, `nothing`), 200×200 RGB,
  uniform backgrounds, near-duplicate frames.
- License: **GPL 2** (Kaggle API metadata `licenseName: "GPL 2"`, verified
  2026-10-09, dataset version 1, ~1.1 GB). Do not redistribute the archive;
  keep `data/` out of git. Demo/research use.
- Citation: Akash Nagaraj (2018), "ASL Alphabet",
  https://www.kaggle.com/datasets/grassknoted/asl-alphabet,
  DOI 10.34740/KAGGLE/DSV/29550.

## Steps
1. Create a Kaggle account, go to Account → **Create New API Token** → `kaggle.json`.
2. Put it at `%USERPROFILE%\.kaggle\kaggle.json` (Windows) or `~/.kaggle/kaggle.json`
   with `chmod 600` (Linux/Mac). Never commit this file.
3. `pip install kaggle`
4. From `signbridge/`: `python scripts/download_data.py` (or `--limit 1000` for a subset).
5. Expect `data/raw/<label>/*.jpg`. Then: `python -m src.dataset`.

## No Kaggle access?
Place your own images as `data/raw/<LETTER>/*.jpg` (any image size/format
OpenCV reads) and continue from `python -m src.dataset`. For the webcam sanity
check, drop a few labeled photos in `data/custom_test/<LETTER>/*.jpg`.

## This checkout
Data was provided as `archive.zip` (repo root, ~1.1 GB, kept out of git):
87,000 training images extracted to `data/raw/` (29 labels × 3,000, verified
readable) + 28 Kaggle test images to `data/kaggle_test/` (`del` absent).
Cap used: 1,000/class → 29,000 candidates; kept 25,171, skipped 3,829 (13.2%).
`nothing` contributed 0 landmarks and was excluded from training (28 labels
trained); the app's no-hand path covers that case.
