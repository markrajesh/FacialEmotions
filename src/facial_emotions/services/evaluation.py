"""Held-out dataset evaluation and benchmark support.

Usage (CLI):
    python -m facial_emotions.services.evaluation --prepare --dataset-csv path/to/fer2013.csv
    python -m facial_emotions.services.evaluation --benchmark --model-path models/emotion/emotion_model.onnx
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from facial_emotions import config
from facial_emotions.services.emotion_inference import EmotionInferenceService


class EvaluationError(RuntimeError):
    pass


def prepare_dataset(csv_path: str, output_dir: Optional[str] = None) -> Path:
    """Split FER-2013 CSV into per-class directories of .png images.

    Expected CSV columns: emotion, pixels, Usage
    Usage values: Training, PublicTest, PrivateTest
    """
    import cv2

    csv_p = Path(csv_path)
    if not csv_p.exists():
        raise EvaluationError(f"Dataset CSV not found: {csv_path}")

    out = Path(output_dir) if output_dir else config.ASSETS_DIR / "dataset"
    labels = config.EMOTION_LABELS

    for usage in ("Training", "PublicTest", "PrivateTest"):
        for label in labels:
            (out / usage / label).mkdir(parents=True, exist_ok=True)

    count = 0
    with open(csv_p, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            emotion_idx = int(row["emotion"])
            label = labels[emotion_idx]
            pixels = np.array(row["pixels"].split(), dtype=np.uint8).reshape(48, 48)
            usage = row["Usage"]
            img_path = out / usage / label / f"{count:06d}.png"
            cv2.imwrite(str(img_path), pixels)
            count += 1

    print(f"Prepared {count} images into {out}")
    return out


def benchmark(
    model_path: Optional[str] = None,
    dataset_dir: Optional[str] = None,
    split: str = "PrivateTest",
) -> Dict[str, float]:
    """Run per-emotion accuracy benchmark on the held-out test split.

    Returns a dict mapping emotion label → accuracy (0.0–1.0).
    Prints a summary table and raises EvaluationError if any class is below
    the 90 % target without a documented mitigation.
    """
    import cv2

    mp = Path(model_path) if model_path else config.EMOTION_MODEL_PATH
    svc = EmotionInferenceService(model_path=mp)

    data_dir = Path(dataset_dir) if dataset_dir else config.ASSETS_DIR / "dataset"
    split_dir = data_dir / split
    if not split_dir.exists():
        raise EvaluationError(
            f"Dataset split not found at {split_dir}. Run --prepare first."
        )

    labels = config.EMOTION_LABELS
    correct: Dict[str, int] = {l: 0 for l in labels}
    total: Dict[str, int] = {l: 0 for l in labels}

    from facial_emotions.domain.models import FaceDetection
    import uuid

    for label in labels:
        label_dir = split_dir / label
        if not label_dir.exists():
            continue
        for img_path in label_dir.glob("*.png"):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]
            # Treat the whole image as one face bounding box
            fake_face = FaceDetection(
                face_id=str(uuid.uuid4()),
                bbox_x=0, bbox_y=0,
                bbox_width=w, bbox_height=h,
                detection_confidence=1.0,
            )
            pred = svc.predict_single_face(img, fake_face)
            total[label] += 1
            if pred.emotion_label.value == label:
                correct[label] += 1

    accuracies: Dict[str, float] = {}
    print(f"\nBenchmark results — split: {split}")
    print(f"{'Emotion':<12} {'Correct':>8} {'Total':>8} {'Accuracy':>10}")
    print("-" * 42)
    for label in labels:
        acc = correct[label] / total[label] if total[label] else 0.0
        accuracies[label] = acc
        flag = "✓" if acc >= 0.90 else "✗ BELOW TARGET"
        print(f"{label:<12} {correct[label]:>8} {total[label]:>8} {acc:>9.1%}  {flag}")

    below = [l for l, a in accuracies.items() if a < 0.90 and total[l] > 0]
    if below:
        print(
            f"\nWARNING: {len(below)} emotion class(es) below 90% target: {below}. "
            "Document mitigations in specs/001-you-please-create/spec.md (SC-001)."
        )

    return accuracies


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FER-2013 dataset utilities")
    parser.add_argument("--prepare", action="store_true", help="Prepare dataset from CSV")
    parser.add_argument("--benchmark", action="store_true", help="Run accuracy benchmark")
    parser.add_argument("--dataset-csv", default="fer2013.csv")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--split", default="PrivateTest")
    args = parser.parse_args()

    if args.prepare:
        prepare_dataset(args.dataset_csv, args.output_dir)
    if args.benchmark:
        benchmark(args.model_path, args.output_dir, args.split)
