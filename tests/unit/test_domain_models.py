"""Unit tests for shared domain models and their validation logic (T060)."""

from __future__ import annotations

import pytest

from facial_emotions.domain.enums import (
    EmotionLabel,
    GenuinenessState,
    InputMode,
    MediaFormat,
    MediaType,
)
from facial_emotions.domain.models import (
    AnalysisRequest,
    EmotionPrediction,
    FaceDetection,
    FrameAnalysis,
    GenuinenessAssessment,
    InputAsset,
    LandmarkSet,
)


class TestAnalysisRequest:
    def test_image_request_requires_source_path(self):
        with pytest.raises(ValueError, match="source_path"):
            AnalysisRequest(request_id="r1", input_mode=InputMode.IMAGE)

    def test_webcam_request_requires_stream_id(self):
        with pytest.raises(ValueError, match="stream_id"):
            AnalysisRequest(request_id="r2", input_mode=InputMode.WEBCAM)

    def test_valid_image_request(self):
        r = AnalysisRequest(request_id="r3", input_mode=InputMode.IMAGE, source_path="/img.jpg")
        assert r.source_path == "/img.jpg"
        assert r.max_faces == 5

    def test_max_faces_below_1_raises(self):
        with pytest.raises(ValueError, match="max_faces"):
            AnalysisRequest(request_id="r4", input_mode=InputMode.IMAGE, source_path="/f", max_faces=0)

    def test_max_faces_above_10_raises(self):
        with pytest.raises(ValueError, match="max_faces"):
            AnalysisRequest(request_id="r5", input_mode=InputMode.IMAGE, source_path="/f", max_faces=11)


class TestInputAsset:
    def test_zero_width_raises(self):
        with pytest.raises(ValueError, match="dimensions"):
            InputAsset(asset_id="a1", media_type=MediaType.IMAGE, format=MediaFormat.JPEG, width=0, height=100)

    def test_negative_height_raises(self):
        with pytest.raises(ValueError, match="dimensions"):
            InputAsset(asset_id="a2", media_type=MediaType.IMAGE, format=MediaFormat.PNG, width=100, height=-1)

    def test_valid_asset(self):
        a = InputAsset(asset_id="a3", media_type=MediaType.IMAGE, format=MediaFormat.JPEG, width=640, height=480)
        assert a.width == 640


class TestFaceDetection:
    def test_confidence_above_1_raises(self):
        with pytest.raises(ValueError, match="detection_confidence"):
            FaceDetection(face_id="f1", bbox_x=0, bbox_y=0, bbox_width=80, bbox_height=80, detection_confidence=1.1)

    def test_confidence_below_0_raises(self):
        with pytest.raises(ValueError, match="detection_confidence"):
            FaceDetection(face_id="f2", bbox_x=0, bbox_y=0, bbox_width=80, bbox_height=80, detection_confidence=-0.1)

    def test_valid_face_detection(self):
        f = FaceDetection(face_id="f3", bbox_x=10, bbox_y=20, bbox_width=80, bbox_height=80, detection_confidence=0.95)
        assert f.face_id == "f3"


class TestEmotionPrediction:
    def test_confidence_above_1_raises(self):
        with pytest.raises(ValueError, match="confidence"):
            EmotionPrediction(face_id="f1", emotion_label=EmotionLabel.HAPPINESS, confidence=1.5, top_k_scores={})

    def test_valid_emotion_prediction(self):
        e = EmotionPrediction(face_id="f2", emotion_label=EmotionLabel.SADNESS, confidence=0.7, top_k_scores={})
        assert e.emotion_label == EmotionLabel.SADNESS


class TestLandmarkSet:
    def test_empty_landmarks_raises(self):
        with pytest.raises(ValueError, match="landmark_points"):
            LandmarkSet(
                face_id="f1",
                landmark_points=[],
                eye_aspect_ratio=0.3,
                mouth_curvature=0.1,
                smile_symmetry_score=0.8,
                cheek_raise_score=0.5,
            )

    def test_valid_landmark_set(self):
        lm = LandmarkSet(
            face_id="f2",
            landmark_points=[(0, 0), (10, 10)],
            eye_aspect_ratio=0.3,
            mouth_curvature=0.0,
            smile_symmetry_score=0.9,
            cheek_raise_score=0.4,
        )
        assert len(lm.landmark_points) == 2


class TestFrameAnalysis:
    def test_default_lists_are_empty(self):
        frame = FrameAnalysis(frame_id="frm-001", timestamp_ms=0)
        assert frame.faces == []
        assert frame.emotions == []
        assert frame.genuineness == []

    def test_faces_and_emotions_populated(self):
        face = FaceDetection(face_id="f1", bbox_x=0, bbox_y=0, bbox_width=80, bbox_height=80, detection_confidence=0.9)
        emo = EmotionPrediction(face_id="f1", emotion_label=EmotionLabel.NEUTRAL, confidence=0.6, top_k_scores={})
        frame = FrameAnalysis(frame_id="frm-002", timestamp_ms=500, faces=[face], emotions=[emo])
        assert len(frame.faces) == 1
        assert frame.emotions[0].emotion_label == EmotionLabel.NEUTRAL


class TestEnums:
    def test_emotion_label_values(self):
        expected = {"anger", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"}
        actual = {e.value for e in EmotionLabel}
        assert actual == expected

    def test_genuineness_state_values(self):
        values = {s.value for s in GenuinenessState}
        assert "genuine" in values
        assert "posed" in values
        assert "uncertain" in values

    def test_input_mode_values(self):
        assert InputMode.IMAGE.value == "image"
        assert InputMode.VIDEO.value == "video"
        assert InputMode.WEBCAM.value == "webcam"
