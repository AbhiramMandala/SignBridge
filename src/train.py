"""Train + compare small classifiers on cached landmarks.

Usage: python -m src.train  (run from signbridge/ dir)
Saves models/model.joblib, models/labels.json, models/metrics.json,
docs/confusion_matrix.png. Test set evaluated ONCE on the picked model.
"""
from __future__ import annotations

import json
import sys

import joblib
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neural_network import MLPClassifier

from .config import (CONFUSION_MATRIX_PNG, LABELS_PATH, LANDMARKS_CACHE, METRICS_PATH,
                     MODEL_PATH, RANDOM_SEED, TEST_FRAC, TRAIN_FRAC)
from .features import augment


def split(X, y, seed=RANDOM_SEED):
    s1 = StratifiedShuffleSplit(n_splits=1, test_size=TEST_FRAC, random_state=seed)
    tr_val, te = next(s1.split(X, y))
    val_rel = 1.0 - TRAIN_FRAC / (1.0 - TEST_FRAC)
    s2 = StratifiedShuffleSplit(n_splits=1, test_size=val_rel, random_state=seed + 1)
    tr, va = next(s2.split(X[tr_val], y[tr_val]))
    return tr_val[tr], tr_val[va], te


def drop_rare_classes(X, y, labels, min_n=10):
    """Drop classes with < min_n samples (stratified 3-way split needs >=10).

    Returns (X, y_reindexed, labels). E.g. 'nothing' (hand-free images) yields
    no landmarks; that case is covered at runtime by the no-hand path.
    """
    keep_idx = [i for i in range(len(labels)) if int((y == i).sum()) >= min_n]
    dropped = [lab for i, lab in enumerate(labels) if i not in keep_idx]
    if dropped:
        print(f"dropping {len(dropped)} under-sampled classes (<{min_n}): {dropped}")
    mask = np.isin(y, keep_idx)
    remap = {old: new for new, old in enumerate(keep_idx)}
    return (X[mask],
            np.array([remap[v] for v in y[mask]], dtype=np.int64),
            [labels[i] for i in keep_idx])


def main() -> dict:
    if not LANDMARKS_CACHE.exists():
        sys.exit(f"No cache at {LANDMARKS_CACHE}. Run: python -m src.dataset")
    z = np.load(LANDMARKS_CACHE, allow_pickle=True)
    X, y, labels = z["X"], z["y"], [str(s) for s in z["labels"]]
    print(f"loaded: {X.shape[0]} samples, {len(labels)} classes")

    X, y, labels = drop_rare_classes(X, y, labels)

    i_tr, i_va, i_te = split(X, y)
    rng = np.random.default_rng(RANDOM_SEED)
    # Augment TRAIN only (split happened before): 1 jittered copy per sample.
    X_tr = np.concatenate([X[i_tr], np.array([augment(v, rng) for v in X[i_tr]])])
    y_tr = np.concatenate([y[i_tr], y[i_tr]])

    candidates = {
        "logreg": LogisticRegression(max_iter=2000, C=1.0, random_state=RANDOM_SEED),
        "random_forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_SEED, n_jobs=-1),
        "mlp": MLPClassifier(hidden_layer_sizes=(128,), max_iter=500, random_state=RANDOM_SEED),
    }
    val_scores, fitted = {}, {}
    for name, clf in candidates.items():
        clf.fit(X_tr, y_tr)
        acc = accuracy_score(y[i_va], clf.predict(X[i_va]))
        val_scores[name] = acc
        fitted[name] = clf
        print(f"val acc: {name} = {acc:.4f}")

    best = max(val_scores, key=val_scores.get)
    model = fitted[best]
    test_acc = accuracy_score(y[i_te], model.predict(X[i_te]))
    report = classification_report(y[i_te], model.predict(X[i_te]),
                                   target_names=labels, output_dict=True, zero_division=0)
    print(f"BEST: {best}  test acc = {test_acc:.4f}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    LABELS_PATH.write_text(json.dumps(labels))
    metrics = {"val": val_scores, "best": best, "test_accuracy": test_acc,
               "test_report": report, "n_train": len(X_tr), "n_val": len(i_va),
               "n_test": len(i_te), "classes": labels}
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))

    cm = confusion_matrix(y[i_te], model.predict(X[i_te]))
    CONFUSION_MATRIX_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=90, fontsize=7)
    ax.set_yticks(range(len(labels)), labels, fontsize=7)
    ax.set_xlabel("predicted");
    ax.set_ylabel("true")
    fig.tight_layout()
    fig.savefig(CONFUSION_MATRIX_PNG, dpi=120)
    plt.close(fig)
    print(f"saved model -> {MODEL_PATH}, metrics -> {METRICS_PATH}")
    return metrics


if __name__ == "__main__":
    main()
