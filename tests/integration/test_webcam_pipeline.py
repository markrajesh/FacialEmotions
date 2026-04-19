"""Integration tests for webcam session processing with recorded fixtures (US3).

Uses synthetic frames (numpy arrays) instead of real camera hardware.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from facial_emotions.contracts.results import WebcamFrameResult
from facial_emotions.domain.enums import EmotionLabel, GenuinenessState
from facial_emotions.domain.models import (
    EmotionPrediction,
    FaceDetection,
    GenuinenessAssessment,
)


def _face(face_id: str = "f0") -> FaceDetection:
    return FaceDetection(face_id=face_id, bbox_x=10, bbox_y=10, bbox_width=80, bbox_height=80, detection_confidence=0.9)


def _emotion(face_id: str = "f0") -> EmotionPrediction:
    return EmotionPrediction(face_id=face_id, emotion_label=EmotionLabel.HAPPINESS, confidence=0.75, top_k_scores={})


def _genuine(face_id: str = "f0") -> GenuinenessAssessment:
    return GenuinenessAssessment(face_id=face_id, predicted_state=GenuinenessState.GENUINE, confidence=0.7, reasoning_flags=[])


def _make_webcam_pipeline(num_faces: int = 1):
    from facial_emotions.pipelines.webcam_pipeline import WebcamAnalysisPipeline

    mock_detector = MagicMock()
    mock_emotion = MagicMock()
    mock_landmark = MagicMock()
    mock_genuine = MagicMock()
    mock_renderer = MagicMock(side_effect=lambda img, faces, emotions, genuineness: img)

    faces = [_face(f"f{i}") for i in range(num_faces)]
    mock_detector.detect.return_value = faces
    mock_emotion.predict_single_face.side_effect = lambda img, face: _emotion(face.face_id)
    mock_landmark.extract.side_effect = lambda img, face_list: [
        _face(f.face_id) for f in face_list
    ] if face_list else []
    mock_genuine.assess.side_effect = lambda lm_list, emo_list: [
        _genuine(emo.face_id) for emo in emo_list
    ]

    return WebcamAnalysisPipeline(
        face_detector=mock_detector,
        emotion_service=mock_emotion,
        landmark_extractor=mock_landmark,
        genuineness_service=mock_genuine,
        overlay_renderer=mock_renderer,
    )


class TestWebcamPipeline:
    def test_analyse_frame_returns_webcam_frame_result(self):
        pipeline = _make_webcam_pipeline()
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        result = pipeline.analyse_frame(frame)
        assert isinstance(result, WebcamFrameResult)

    def test_result_has_request_id(self):
        pipeline = _make_webcam_pipeline()
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        result = pipeline.analyse_frame(frame, request_id="req-cam-01")
        assert result.request_id == "req-cam-01"

    def test_result_has_faces(self):
        pipeline = _make_webcam_pipeline(num_faces=2)
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        result = pipeline.analyse_frame(frame)
        assert len(result.faces) == 2

    def test_result_face_emotion_is_happiness(self):
        pipeline = _make_webcam_pipeline()
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        result = pipeline.analyse_frame(frame)
        assert result.faces[0].emotion == "happiness"

    def test_no_faces_returns_empty_list(self):
        from facial_emotions.pipelines.webcam_pipeline import WebcamAnalysisPipeline

        mock_detector = MagicMock()
        mock_detector.detect.return_value = []
        pipeline = WebcamAnalysisPipeline(
            face_detector=mock_detector,
            emotion_service=MagicMock(),
            landmark_extractor=MagicMock(),
            genuineness_service=MagicMock(),
            overlay_renderer=MagicMock(side_effect=lambda img, faces, emotions, genuineness: img),
        )
        result = pipeline.analyse_frame(np.zeros((240, 320, 3), dtype=np.uint8))
        assert result.faces == []

    def test_frame_index_increments_across_calls(self):
        pipeline = _make_webcam_pipeline()
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        r1 = pipeline.analyse_frame(frame)
        r2 = pipeline.analyse_frame(frame)
        assert r2.frame_index > r1.frame_index

    def test_annotated_frame_is_populated(self):
        pipeline = _make_webcam_pipeline()
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        result = pipeline.analyse_frame(frame)
        assert result.annotated_frame is not None
