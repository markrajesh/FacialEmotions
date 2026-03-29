"""Per-emotion accuracy benchmark tests (SC-001).

These tests verify that the evaluation.py benchmark infrastructure is
functional and that when an ONNX model is available, per-emotion accuracy
can be measured.

In the absence of a real model, the tests exercise the evaluation code paths
with stubs and verify output structure.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from facial_emotions import config
from facial_emotions.services.evaluation import EvaluationError, prepare_dataset


class TestEvaluationInfrastructure:
    def test_emotion_labels_cover_all_seven_classes(self):
        expected = {"anger", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"}
        assert set(config.EMOTION_LABELS) == expected

    def test_prepare_dataset_raises_if_csv_missing(self, tmp_path):
        with pytest.raises(EvaluationError, match="not found"):
            prepare_dataset(str(tmp_path / "nonexistent.csv"), str(tmp_path / "out"))

    def test_prepare_dataset_creates_per_class_dirs(self, tmp_path):
        """Smoke-test with a minimal 2-row FER-like CSV."""
        import csv
        import cv2

        csv_path = tmp_path / "fer.csv"
        pixels = " ".join(["128"] * (48 * 48))
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["emotion", "pixels", "Usage"])
            writer.writeheader()
            writer.writerow({"emotion": "3", "pixels": pixels, "Usage": "PrivateTest"})
            writer.writerow({"emotion": "4", "pixels": pixels, "Usage": "PrivateTest"})

        out = prepare_dataset(str(csv_path), str(tmp_path / "out"))
        assert (out / "PrivateTest" / "happiness").exists()
        assert (out / "PrivateTest" / "neutral").exists()


class TestBenchmarkOutputStructure:
    def test_benchmark_returns_dict_keyed_by_emotion(self, tmp_path):
        """Mock the emotion inference to return correct predictions for all images."""
        import csv
        import cv2

        csv_path = tmp_path / "fer.csv"
        pixels = " ".join(["128"] * (48 * 48))
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["emotion", "pixels", "Usage"])
            writer.writeheader()
            writer.writerow({"emotion": "3", "pixels": pixels, "Usage": "PrivateTest"})

        from facial_emotions.services.evaluation import prepare_dataset
        ds_dir = prepare_dataset(str(csv_path), str(tmp_path / "ds"))

        from facial_emotions.domain.enums import EmotionLabel
        from facial_emotions.domain.models import EmotionPrediction

        mock_pred = EmotionPrediction(
            face_id="x",
            emotion_label=EmotionLabel.HAPPINESS,
            confidence=0.9,
            top_k_scores={},
        )
        mock_svc = MagicMock()
        mock_svc.predict_single_face.return_value = mock_pred

        with patch(
            "facial_emotions.services.evaluation.EmotionInferenceService",
            return_value=mock_svc,
        ):
            from facial_emotions.services.evaluation import benchmark
            results = benchmark(
                model_path=str(tmp_path / "fake.onnx"),
                dataset_dir=str(tmp_path / "ds"),
                split="PrivateTest",
            )

        assert isinstance(results, dict)
        assert "happiness" in results
        assert 0.0 <= results["happiness"] <= 1.0

    def test_benchmark_accuracy_is_1_0_when_all_correct(self, tmp_path):
        """Perfect predictions → 1.0 for the tested class."""
        import csv

        csv_path = tmp_path / "fer.csv"
        pixels = " ".join(["128"] * (48 * 48))
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["emotion", "pixels", "Usage"])
            writer.writeheader()
            for _ in range(5):
                writer.writerow({"emotion": "3", "pixels": pixels, "Usage": "PrivateTest"})

        from facial_emotions.services.evaluation import prepare_dataset
        prepare_dataset(str(csv_path), str(tmp_path / "ds"))

        from facial_emotions.domain.enums import EmotionLabel
        from facial_emotions.domain.models import EmotionPrediction

        mock_pred = EmotionPrediction(
            face_id="x",
            emotion_label=EmotionLabel.HAPPINESS,
            confidence=0.9,
            top_k_scores={},
        )
        mock_svc = MagicMock()
        mock_svc.predict_single_face.return_value = mock_pred

        with patch(
            "facial_emotions.services.evaluation.EmotionInferenceService",
            return_value=mock_svc,
        ):
            from facial_emotions.services.evaluation import benchmark
            results = benchmark(
                model_path=str(tmp_path / "fake.onnx"),
                dataset_dir=str(tmp_path / "ds"),
                split="PrivateTest",
            )
        assert results["happiness"] == 1.0
