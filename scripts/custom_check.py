"""Sanity-check the trained model on YOUR webcam photos.

Reads data/custom_test/<label>/*.jpg, predicts each, reports accuracy.
This is reported SEPARATELY from dataset test metrics — never mixed in.
Only images under known trained labels are scored; anything else is
reported as skipped and never counted as right or wrong.

Usage: python scripts/custom_check.py  (run from signbridge/ dir)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import CUSTOM_TEST_DIR
from src.dataset import IMG_EXTS
from src.predict import load, predict_file

from sklearn.metrics import classification_report


def _empty_result() -> dict:
    return {"n_ok": 0, "n_total": 0, "n_discovered": 0, "skipped": {},
            "per_class": {}, "per_image": []}


def main() -> dict:
    load()  # actionable FileNotFoundError if no trained model; never trains here
    if not CUSTOM_TEST_DIR.is_dir():
        print(f"No {CUSTOM_TEST_DIR}/ yet. Create it with "
              f"python scripts/setup_custom_test.py, drop labeled webcam photos as "
              f"{CUSTOM_TEST_DIR}/<LETTER>/*.jpg and rerun.")
        return _empty_result()
    _, labels = load()
    known = set(labels)
    n_ok = n_total = n_discovered = 0
    skipped: dict[str, int] = {}
    per_class: dict[str, list[int]] = {}
    per_image, y_true, y_pred = [], [], []

    def skip(reason: str) -> None:
        skipped[reason] = skipped.get(reason, 0) + 1

    folders = []
    for label_dir in sorted(p for p in CUSTOM_TEST_DIR.iterdir() if p.is_dir()):
        files = [f for f in sorted(label_dir.iterdir()) if f.suffix.lower() in IMG_EXTS]
        folders.append((label_dir, files))
    loose = sorted(p.name for p in CUSTOM_TEST_DIR.iterdir()
                   if p.is_file() and p.suffix.lower() in IMG_EXTS)
    if loose:
        shown = ", ".join(loose[:5]) + ("..." if len(loose) > 5 else "")
        print(f"note: {len(loose)} image(s) directly under custom_test/ are ignored; "
              f"move them into their <LABEL>/ folders: {shown}")
    if not any(files for _, files in folders):
        print("No valid test images found. Add photos (see "
              "python scripts/setup_custom_test.py) and rerun.")
        print("discovered=0 scored=0")
        return _empty_result()
    for label_dir, files in folders:
        if label_dir.name not in known:
            print(f"skip folder {label_dir.name}/: unknown label "
                  f"({len(files)} image(s) ignored, not scored)")
            for _ in files:
                skip("unknown_label")
            continue
        if not files:
            print(f"note: folder {label_dir.name}/ has no images")
            continue
        for f in files:
            rel = f"{label_dir.name}/{f.name}"
            n_discovered += 1
            r = predict_file(str(f))
            if r["status"] != "ok":
                print(f"{rel}: true={label_dir.name} -> {r['status']} (skipped, not scored)")
                skip(r["status"])  # "unreadable" or "no_hand"
                continue
            hit = r["letter"] == label_dir.name
            n_ok, n_total = n_ok + hit, n_total + 1
            rec = per_class.setdefault(label_dir.name, [0, 0])
            rec[0], rec[1] = rec[0] + hit, rec[1] + 1
            y_true.append(label_dir.name)
            y_pred.append(r["letter"])
            per_image.append((rel, label_dir.name, r["letter"], r["confidence"]))
            print(f"{rel}: true={label_dir.name} pred={r['letter']} "
                  f"conf={r['confidence']:.2f} {'OK' if hit else 'MISS'}")
    if n_total:
        print(f"custom_test accuracy: {n_ok}/{n_total} = {n_ok / n_total:.2f} "
              f"(informal sanity check, NOT comparable to dataset test accuracy)")
        present = sorted(set(y_true) | set(y_pred))
        print(classification_report(y_true, y_pred, labels=present,
                                    target_names=present, zero_division=0))
    else:
        print("No valid test images found.")
    if skipped:
        print("skipped (not scored): " +
              ", ".join(f"{k}={v}" for k, v in sorted(skipped.items())))
    print(f"discovered={n_discovered} scored={n_total}")
    return {"n_ok": n_ok, "n_total": n_total, "n_discovered": n_discovered,
            "skipped": skipped, "per_class": per_class, "per_image": per_image}


if __name__ == "__main__":
    main()
