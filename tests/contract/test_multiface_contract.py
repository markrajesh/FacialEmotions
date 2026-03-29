"""Contract tests for multi-face image analysis responses (US2).

These tests verify that when multiple faces are detected, the
ImageAnalysisResult contains one FaceResult per face with aligned IDs.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from facial_emotions.contracts.results import FaceResult, ImageAnalysisResult
from facial_emotions.domain.enums import EmotionLabel
from facial_emotions.domain.models import EmotionPrediction, FaceDetection


def _make_multi_face_pipeline(n_faces: int):
    from facial_emotions.pipelines.image_pipeline import ImageAnalysisPipeline

    faces = [
        FaceDetection(
            face_id=f"face-{i:03d}",
            bbox_x=i * 120, bbox_y=50, bbox_width=100, bbox_height=100,
            detection_confidence=0.9,
        )
        for i in range(n_faces)
    ]
    predictions = [
        EmotionPrediction(
            face_id=f"face-{i:03d}",
            emotion_label=list(EmotionLabel)[i % 7],
            confidence=0.7,
            top_k_scores={},
        )
        for i in range(n_faces)
    ]

    detector = MagicMock()
    detector.detect.return_value = faces
    emotion_svc = MagicMock()
    emotion_svc.predict.return_value = predictions

    return ImageAnalysisPipeline(face_detector=detector, emotion_service=emotion_svc)


class TestMultiFaceContract:
    def test_two_face_result_has_two_faces(self):
        pipeline = _make_multi_face_pipeline(2)
        result = pipeline.analyse(np.zeros((480, 640, 3), dtype=np.uint8))
        assert len(result.faces) == 2

    def test_five_face_result_has_five_faces(self):
        pipeline = _make_multi_face_pipeline(5)
        result = pipeline.analyse(np.zeros((480, 640, 3), dtype=np.uint8))
        assert len(result.faces) == 5

    def test_face_ids_are_unique(self):
        pipeline = _make_multi_face_pipeline(3)
        result = pipeline.analyse(np.zeros((480, 640, 3), dtype=np.uint8))
        ids = [f.face_id for f in result.faces]
        assert len(ids) == len(set(ids))

    def test_each_face_has_valid_emotion(self):
        pipeline = _make_multi_face_pipeline(3)
        result = pipeline.analyse(np.zeros((480, 640, 3), dtype=np.uint8))
        valid_emotions = {e.value for e in EmotionLabel}
        for face in result.faces:
            assert face.emotion in valid_emotions

    def test_each_face_has_bbox(self):
        pipeline = _make_multi_face_pipeline(2)
        result = pipeline.analyse(np.zeros((480, 640, 3), dtype=np.uint8))
        for face in result.faces:
            assert all(k in face.bbox for k in ("x", "y", "width", "height"))

    def test_each_face_has_genuineness_state(self):
        pipeline = _make_multi_face_pipeline(2)
        result = pipeline.analyse(np.zeros((480, 640, 3), dtype=np.uint8))
        for face in result.faces:
            assert face.genuineness_state in ("genuine", "posed", "uncertain")

    def test_result_is_image_analysis_result_type(self):
        pipeline = _make_multi_face_pipeline(2)
        result = pipeline.analyse(np.zeros((480, 640, 3), dtype=np.uint8))
        assert isinstance(result, ImageAnalysisResult)
