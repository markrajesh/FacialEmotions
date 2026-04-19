"""Integration tests for the video analysis pipeline (US4).

Uses mocked sub-services so no real video file or ONNX model is needed.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from facial_emotions.contracts.results import VideoAnalysisResult, FrameResult
from facial_emotions.domain.enums import EmotionLabel, GenuinenessState
from facial_emotions.domain.models import (
    EmotionPrediction,
    FaceDetection,
    GenuinenessAssessment,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_face(face_id: str) -> FaceDetection:
    return FaceDetection(face_id=face_id, bbox_x=10, bbox_y=10, bbox_width=80, bbox_height=80, detection_confidence=0.9)


def _make_emotion(face_id: str) -> EmotionPrediction:
    return EmotionPrediction(face_id=face_id, emotion_label=EmotionLabel.HAPPINESS, confidence=0.75, top_k_scores={})


def _make_genuineness(face_id: str) -> GenuinenessAssessment:
    return GenuinenessAssessment(face_id=face_id, predicted_state=GenuinenessState.GENUINE, confidence=0.7, reasoning_flags=[])


def _make_pipeline(num_faces: int = 1, num_sampled_frames: int = 3):
    """Return a VideoAnalysisPipeline with all sub-services mocked."""
    from facial_emotions.pipelines.video_pipeline import VideoAnalysisPipeline

    mock_detector = MagicMock()
    mock_emotion = MagicMock()
    mock_landmark = MagicMock()
    mock_genuine = MagicMock()
    mock_renderer = MagicMock(side_effect=lambda img, faces: img)

    faces = [_make_face(f"f{i}") for i in range(num_faces)]
    mock_detector.detect.return_value = faces
    mock_emotion.predict_single_face.side_effect = lambda img, face: _make_emotion(face.face_id)
    mock_landmark.extract.side_effect = lambda img, face_list: [
        _make_face(f.face_id) for f in face_list  # return a LandmarkSet stub per face
    ] if face_list else []
    mock_genuine.assess.side_effect = lambda lm_list, emo_list: [
        _make_genuineness(emo.face_id) for emo in emo_list
    ]

    pipeline = VideoAnalysisPipeline(
        face_detector=mock_detector,
        emotion_service=mock_emotion,
        landmark_extractor=mock_landmark,
        genuineness_service=mock_genuine,
        overlay_renderer=mock_renderer,
    )

    # Patch the internal method that reads a video to return synthetic frames
    blank_frame = np.zeros((240, 320, 3), dtype=np.uint8)
    timestamps_ms = [i * 500 for i in range(num_sampled_frames)]

    with patch.object(pipeline, "_read_sampled_frames", return_value=list(zip(timestamps_ms, [blank_frame] * num_sampled_frames))):
        result = pipeline.analyse(
            video_path="fake_video.mp4",
            request_id="req-vid-test",
        )

    return result


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestVideoAnalysisPipeline:
    def test_result_is_video_analysis_result(self):
        result = _make_pipeline()
        assert isinstance(result, VideoAnalysisResult)

    def test_result_request_id_matches(self):
        result = _make_pipeline()
        assert result.request_id == "req-vid-test"

    def test_sampled_frames_count_matches_input(self):
        result = _make_pipeline(num_sampled_frames=5)
        assert len(result.sampled_frames) == 5

    def test_each_frame_has_correct_timestamp(self):
        result = _make_pipeline(num_sampled_frames=3)
        timestamps = [f.timestamp_ms for f in result.sampled_frames]
        assert timestamps == [0, 500, 1000]

    def test_each_frame_has_face_results(self):
        result = _make_pipeline(num_faces=2, num_sampled_frames=2)
        for frame in result.sampled_frames:
            assert len(frame.faces) == 2

    def test_face_emotion_assigned_to_correct_face_id(self):
        result = _make_pipeline(num_faces=1, num_sampled_frames=1)
        face = result.sampled_frames[0].faces[0]
        assert face.emotion == "happiness"
        assert face.genuineness_state == "genuine"

    def test_empty_video_returns_empty_frames(self):
        result = _make_pipeline(num_sampled_frames=0)
        assert result.sampled_frames == []

    def test_frame_with_no_detected_faces(self):
        from facial_emotions.pipelines.video_pipeline import VideoAnalysisPipeline

        mock_detector = MagicMock()
        mock_emotion = MagicMock()
        mock_landmark = MagicMock()
        mock_genuine = MagicMock()
        mock_renderer = MagicMock(side_effect=lambda img, faces: img)

        mock_detector.detect.return_value = []  # no faces

        pipeline = VideoAnalysisPipeline(
            face_detector=mock_detector,
            emotion_service=mock_emotion,
            landmark_extractor=mock_landmark,
            genuineness_service=mock_genuine,
            overlay_renderer=mock_renderer,
        )
        blank = np.zeros((240, 320, 3), dtype=np.uint8)
        with patch.object(pipeline, "_read_sampled_frames", return_value=[(0, blank)]):
            result = pipeline.analyse(video_path="fake.mp4", request_id="req-noface")

        assert len(result.sampled_frames) == 1
        assert result.sampled_frames[0].faces == []
