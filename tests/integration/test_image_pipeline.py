"""Integration tests for single-image emotion detection pipeline.

These tests verify that the full pipeline (face detection → landmarks →
emotion inference → genuineness assessment) produces aligned, well-formed
results for a single image with one or more faces.

Tests use mocked services to avoid requiring a real ONNX model on CI.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from facial_emotions.contracts.results import FaceResult, ImageAnalysisResult
from facial_emotions.domain.enums import EmotionLabel, GenuinenessState
from facial_emotions.domain.models import (
    EmotionPrediction,
    FaceDetection,
    GenuinenessAssessment,
    LandmarkSet,
)


@pytest.fixture()
def image_pipeline(mock_face_detector, mock_emotion_service):
    from facial_emotions.pipelines.image_pipeline import ImageAnalysisPipeline
    return ImageAnalysisPipeline(
        face_detector=mock_face_detector,
        emotion_service=mock_emotion_service,
    )


class TestImagePipelineSingleFace:
    def test_returns_image_analysis_result(self, image_pipeline, sample_bgr_image):
        result = image_pipeline.analyse(sample_bgr_image, request_id="test-001")
        assert isinstance(result, ImageAnalysisResult)

    def test_result_contains_one_face(self, image_pipeline, sample_bgr_image):
        result = image_pipeline.analyse(sample_bgr_image, request_id="test-002")
        assert len(result.faces) == 1

    def test_face_has_valid_emotion_label(self, image_pipeline, sample_bgr_image):
        result = image_pipeline.analyse(sample_bgr_image, request_id="test-003")
        face = result.faces[0]
        valid_emotions = {e.value for e in EmotionLabel}
        assert face.emotion in valid_emotions

    def test_face_emotion_confidence_in_range(self, image_pipeline, sample_bgr_image):
        result = image_pipeline.analyse(sample_bgr_image, request_id="test-004")
        face = result.faces[0]
        assert 0.0 <= face.emotion_confidence <= 1.0

    def test_face_has_valid_genuineness_state(self, image_pipeline, sample_bgr_image):
        result = image_pipeline.analyse(sample_bgr_image, request_id="test-005")
        face = result.faces[0]
        assert face.genuineness_state in ("genuine", "posed", "uncertain")

    def test_face_has_bbox_with_expected_keys(self, image_pipeline, sample_bgr_image):
        result = image_pipeline.analyse(sample_bgr_image, request_id="test-006")
        face = result.faces[0]
        assert all(k in face.bbox for k in ("x", "y", "width", "height"))

    def test_result_image_dimensions_populated(self, image_pipeline, sample_bgr_image):
        result = image_pipeline.analyse(sample_bgr_image, request_id="test-007")
        assert result.image_width == sample_bgr_image.shape[1]
        assert result.image_height == sample_bgr_image.shape[0]

    def test_no_face_returns_empty_list(self, sample_bgr_image):
        from facial_emotions.pipelines.image_pipeline import ImageAnalysisPipeline
        no_face_detector = MagicMock()
        no_face_detector.detect.return_value = []
        emotion_service = MagicMock()
        pipeline = ImageAnalysisPipeline(
            face_detector=no_face_detector,
            emotion_service=emotion_service,
        )
        result = pipeline.analyse(sample_bgr_image, request_id="test-no-face")
        assert result.faces == []

    def test_face_id_consistent(self, image_pipeline, sample_bgr_image):
        result = image_pipeline.analyse(sample_bgr_image, request_id="test-id")
        face = result.faces[0]
        assert isinstance(face.face_id, str)
        assert len(face.face_id) > 0

    def test_top_k_emotions_present(self, image_pipeline, sample_bgr_image):
        result = image_pipeline.analyse(sample_bgr_image, request_id="test-topk")
        face = result.faces[0]
        assert isinstance(face.top_k_emotions, dict)
