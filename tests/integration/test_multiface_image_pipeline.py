"""Integration tests for multi-face image analysis (US2)."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from facial_emotions.domain.enums import EmotionLabel
from facial_emotions.domain.models import EmotionPrediction, FaceDetection


def _make_pipeline(faces, predictions):
    from facial_emotions.pipelines.image_pipeline import ImageAnalysisPipeline
    detector = MagicMock()
    detector.detect.return_value = faces
    emotion_svc = MagicMock()
    emotion_svc.predict.return_value = predictions
    return ImageAnalysisPipeline(face_detector=detector, emotion_service=emotion_svc)


@pytest.fixture()
def two_face_pipeline():
    faces = [
        FaceDetection(face_id="f1", bbox_x=0, bbox_y=0, bbox_width=80, bbox_height=80, detection_confidence=0.9),
        FaceDetection(face_id="f2", bbox_x=150, bbox_y=0, bbox_width=80, bbox_height=80, detection_confidence=0.85),
    ]
    preds = [
        EmotionPrediction(face_id="f1", emotion_label=EmotionLabel.HAPPINESS, confidence=0.8, top_k_scores={}),
        EmotionPrediction(face_id="f2", emotion_label=EmotionLabel.SADNESS, confidence=0.7, top_k_scores={}),
    ]
    return _make_pipeline(faces, preds)


class TestMultiFaceImagePipeline:
    def test_two_faces_detected(self, two_face_pipeline):
        result = two_face_pipeline.analyse(np.zeros((480, 640, 3), dtype=np.uint8))
        assert len(result.faces) == 2

    def test_face_emotions_differ(self, two_face_pipeline):
        result = two_face_pipeline.analyse(np.zeros((480, 640, 3), dtype=np.uint8))
        emotions = [f.emotion for f in result.faces]
        assert "happiness" in emotions
        assert "sadness" in emotions

    def test_face_ids_preserved(self, two_face_pipeline):
        result = two_face_pipeline.analyse(np.zeros((480, 640, 3), dtype=np.uint8))
        ids = {f.face_id for f in result.faces}
        assert "f1" in ids
        assert "f2" in ids

    def test_bounding_boxes_do_not_overlap_for_distinct_faces(self, two_face_pipeline):
        result = two_face_pipeline.analyse(np.zeros((480, 640, 3), dtype=np.uint8))
        f1, f2 = result.faces[0], result.faces[1]
        # The two bounding boxes we injected do not overlap
        f1_right = f1.bbox["x"] + f1.bbox["width"]
        assert f1_right <= f2.bbox["x"] or f2.bbox["x"] + f2.bbox["width"] <= f1.bbox["x"]

    def test_five_faces_within_max(self):
        faces = [
            FaceDetection(face_id=f"f{i}", bbox_x=i*100, bbox_y=0, bbox_width=80, bbox_height=80, detection_confidence=0.9)
            for i in range(5)
        ]
        preds = [
            EmotionPrediction(face_id=f"f{i}", emotion_label=EmotionLabel.NEUTRAL, confidence=0.6, top_k_scores={})
            for i in range(5)
        ]
        pipeline = _make_pipeline(faces, preds)
        result = pipeline.analyse(np.zeros((480, 640, 3), dtype=np.uint8))
        assert len(result.faces) == 5

    def test_result_image_dimensions_correct(self, two_face_pipeline):
        img = np.zeros((720, 1280, 3), dtype=np.uint8)
        result = two_face_pipeline.analyse(img)
        assert result.image_width == 1280
        assert result.image_height == 720
