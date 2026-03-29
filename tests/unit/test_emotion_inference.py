"""Unit tests for emotion inference service behaviour.

Tests mock ONNX Runtime so no model file is required.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from facial_emotions.domain.enums import EmotionLabel
from facial_emotions.domain.models import EmotionPrediction, FaceDetection
from facial_emotions.services.emotion_inference import (
    EmotionInferenceError,
    EmotionInferenceService,
)


@pytest.fixture()
def face_detection():
    return FaceDetection(
        face_id="face-001",
        bbox_x=50, bbox_y=50, bbox_width=100, bbox_height=100,
        detection_confidence=0.9,
    )


@pytest.fixture()
def sample_image():
    return np.full((200, 200, 3), 120, dtype=np.uint8)


def _make_mock_session(probs: list[float]):
    """Return a mock ort.InferenceSession that outputs given probabilities."""
    session = MagicMock()
    input_meta = MagicMock()
    input_meta.name = "input"
    input_meta.shape = [1, 1, 48, 48]
    session.get_inputs.return_value = [input_meta]
    session.run.return_value = [np.array([probs], dtype=np.float32)]
    return session


class TestEmotionInferenceService:
    def test_predict_returns_emotion_prediction(self, face_detection, sample_image):
        probs = [0.1, 0.1, 0.1, 0.5, 0.1, 0.05, 0.05]  # happiness dominant
        with patch("onnxruntime.InferenceSession", return_value=_make_mock_session(probs)):
            svc = EmotionInferenceService(model_path=Path("fake.onnx"))
            svc._session = _make_mock_session(probs)
            svc._input_name = "input"
            svc._input_shape = [1, 1, 48, 48]
            result = svc.predict_single_face(sample_image, face_detection)
        assert isinstance(result, EmotionPrediction)
        assert result.face_id == face_detection.face_id

    def test_predict_top_emotion_matches_highest_prob(self, face_detection, sample_image):
        probs = [0.0, 0.0, 0.0, 0.9, 0.1, 0.0, 0.0]  # happiness = index 3
        svc = EmotionInferenceService(model_path=Path("fake.onnx"))
        svc._session = _make_mock_session(probs)
        svc._input_name = "input"
        svc._input_shape = [1, 1, 48, 48]
        result = svc.predict_single_face(sample_image, face_detection)
        assert result.emotion_label == EmotionLabel.HAPPINESS

    def test_confidence_in_valid_range(self, face_detection, sample_image):
        probs = [1 / 7] * 7
        svc = EmotionInferenceService(model_path=Path("fake.onnx"))
        svc._session = _make_mock_session(probs)
        svc._input_name = "input"
        svc._input_shape = [1, 1, 48, 48]
        result = svc.predict_single_face(sample_image, face_detection)
        assert 0.0 <= result.confidence <= 1.0

    def test_top_k_scores_has_all_labels(self, face_detection, sample_image):
        probs = [0.14, 0.14, 0.14, 0.15, 0.14, 0.14, 0.15]
        svc = EmotionInferenceService(model_path=Path("fake.onnx"))
        svc._session = _make_mock_session(probs)
        svc._input_name = "input"
        svc._input_shape = [1, 1, 48, 48]
        result = svc.predict_single_face(sample_image, face_detection)
        from facial_emotions import config
        assert all(label in result.top_k_scores for label in config.EMOTION_LABELS)

    def test_empty_crop_returns_neutral(self, face_detection, sample_image):
        """Face bbox outside image bounds returns safe neutral result."""
        svc = EmotionInferenceService(model_path=Path("fake.onnx"))
        svc._session = _make_mock_session([1 / 7] * 7)
        svc._input_name = "input"
        svc._input_shape = [1, 1, 48, 48]
        # Face bbox outside image → empty crop
        oob_face = FaceDetection(
            face_id="oob",
            bbox_x=500, bbox_y=500, bbox_width=100, bbox_height=100,
            detection_confidence=0.9,
        )
        result = svc.predict_single_face(sample_image, oob_face)
        assert result.emotion_label == EmotionLabel.NEUTRAL
        assert result.confidence == 0.0

    def test_predict_list_returns_one_per_face(self, face_detection, sample_image):
        probs = [1 / 7] * 7
        svc = EmotionInferenceService(model_path=Path("fake.onnx"))
        svc._session = _make_mock_session(probs)
        svc._input_name = "input"
        svc._input_shape = [1, 1, 48, 48]
        face2 = FaceDetection(
            face_id="face-002",
            bbox_x=10, bbox_y=10, bbox_width=50, bbox_height=50,
            detection_confidence=0.8,
        )
        results = svc.predict(sample_image, [face_detection, face2])
        assert len(results) == 2

    def test_model_not_found_raises_error(self, face_detection, sample_image):
        svc = EmotionInferenceService(model_path=Path("/nonexistent/path.onnx"))
        with pytest.raises(EmotionInferenceError, match="not found"):
            svc.predict_single_face(sample_image, face_detection)
