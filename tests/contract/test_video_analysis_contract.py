"""Contract tests for video analysis output schema (US4).

Verifies that VideoAnalysisResult satisfies the structural contract regardless
of how the pipeline is implemented internally.
"""

from __future__ import annotations

import pytest

from facial_emotions.contracts.results import (
    FaceResult,
    FrameResult,
    VideoAnalysisResult,
)


def _make_face_result(face_id: str = "f0") -> FaceResult:
    return FaceResult(
        face_id=face_id,
        bbox={"x": 10, "y": 10, "width": 80, "height": 80},
        detection_confidence=0.9,
        emotion="happiness",
        emotion_confidence=0.8,
        top_k_emotions={"happiness": 0.8, "neutral": 0.2},
        genuineness_state="genuine",
        genuineness_confidence=0.7,
        genuineness_flags=["symmetric_smile"],
    )


def _make_video_result(num_frames: int = 3) -> VideoAnalysisResult:
    frames = [
        FrameResult(
            frame_id=f"frame-{i:04d}",
            timestamp_ms=i * 500,
            faces=[_make_face_result()],
        )
        for i in range(num_frames)
    ]
    return VideoAnalysisResult(
        request_id="req-video-001",
        video_path="/fixtures/sample.mp4",
        duration_seconds=1.5,
        sampled_frames=frames,
    )


class TestVideoAnalysisResultSchema:
    def test_result_has_request_id(self):
        result = _make_video_result()
        assert result.request_id == "req-video-001"

    def test_result_has_video_path(self):
        result = _make_video_result()
        assert result.video_path == "/fixtures/sample.mp4"

    def test_result_has_duration_seconds(self):
        result = _make_video_result()
        assert result.duration_seconds == pytest.approx(1.5)

    def test_result_has_sampled_frames(self):
        result = _make_video_result(num_frames=3)
        assert len(result.sampled_frames) == 3

    def test_each_frame_has_timestamp_ms(self):
        result = _make_video_result(num_frames=3)
        timestamps = [f.timestamp_ms for f in result.sampled_frames]
        assert timestamps == [0, 500, 1000]

    def test_each_frame_has_faces(self):
        result = _make_video_result(num_frames=2)
        for frame in result.sampled_frames:
            assert len(frame.faces) == 1
            assert frame.faces[0].face_id == "f0"

    def test_to_dict_contains_all_fields(self):
        result = _make_video_result(num_frames=1)
        d = result.to_dict()
        assert set(d.keys()) >= {"request_id", "video_path", "duration_seconds", "sampled_frames"}

    def test_to_dict_serialises_nested_frames(self):
        result = _make_video_result(num_frames=2)
        d = result.to_dict()
        assert len(d["sampled_frames"]) == 2
        assert d["sampled_frames"][0]["timestamp_ms"] == 0

    def test_empty_video_result_is_valid(self):
        result = VideoAnalysisResult(
            request_id="req-empty",
            video_path="/fixtures/empty.mp4",
            duration_seconds=0.0,
            sampled_frames=[],
        )
        assert result.sampled_frames == []
        assert result.to_dict()["sampled_frames"] == []

    def test_frame_with_no_faces_is_valid(self):
        frame = FrameResult(frame_id="frame-0000", timestamp_ms=0, faces=[])
        result = VideoAnalysisResult(
            request_id="req-nofaces",
            video_path="/fixtures/nofaces.mp4",
            duration_seconds=0.5,
            sampled_frames=[frame],
        )
        d = result.to_dict()
        assert d["sampled_frames"][0]["faces"] == []
