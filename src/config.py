"""Shared constants for SignBridge MVP."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CUSTOM_TEST_DIR = DATA_DIR / "custom_test"
MODELS_DIR = ROOT / "models"
DOCS_DIR = ROOT / "docs"

# Kaggle ASL Alphabet folders: A-Z plus these three extras.
EXPECTED_LABELS = [chr(c) for c in range(ord("A"), ord("Z") + 1)] + ["space", "del", "nothing"]

IMAGE_SIZE = 200  # MediaPipe Hands `min_detection_confidence` partner: square-ish inputs work best
MIN_DETECTION_CONFIDENCE = 0.5
RANDOM_SEED = 42

# Cap images per class so CPU training stays fast (None = use all).
MAX_PER_CLASS = 1000

# Train/val/test fractions. Split happens BEFORE any augmentation.
TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.70, 0.15, 0.15

# Below this confidence the app shows "uncertain" instead of a letter.
CONFIDENCE_THRESHOLD = 0.50

MODEL_PATH = MODELS_DIR / "model.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"
LABELS_PATH = MODELS_DIR / "labels.json"
LANDMARKS_CACHE = PROCESSED_DIR / "landmarks.npz"
SKIPPED_LOG = PROCESSED_DIR / "skipped.json"
CONFUSION_MATRIX_PNG = DOCS_DIR / "confusion_matrix.png"
