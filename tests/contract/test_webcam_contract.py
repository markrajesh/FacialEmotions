"""Contract tests for webcam frame analysis output schema (US3).

Verifies that WebcamFrameResult satisfies the structural contract.
"""

from __future__ import annotations

import pytest

from facial_emotions.contracts.results import FaceResult, WebcamFrameResult


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


def _make_webcam_result(num_faces: int = 1) -> WebcamFrameResult:
    return WebcamFrameResult(
        request_id="req-cam-001",
        frame_index=5,
        timestamp_ms=500,
        faces=[_make_face_result(f"f{i}") for i in range(num_faces)],
    )


class TestWebcamFrameResultSchema:
    def test_has_request_id(self):
        r = _make_webcam_result()
        assert r.request_id == "req-cam-001"

    def test_has_frame_index(self):
        r = _make_webcam_result()
        assert r.frame_index == 5

    def test_has_timestamp_ms(self):
        r = _make_webcam_result()
        assert r.timestamp_ms == 500

    def test_has_faces_list(self):
        r = _make_webcam_result(num_faces=2)
        assert len(r.faces) == 2

    def test_each_face_has_emotion(self):
        r = _make_webcam_result(num_faces=2)
        for face in r.faces:
            assert face.emotion == "happiness"

    def test_to_dict_excludes_annotated_frame(self):
        import numpy as np
        r = _make_webcam_result()
        r.annotated_frame = np.zeros((10, 10, 3), dtype=np.uint8)
        d = r.to_dict()
        assert "annotated_frame" not in d

    def test_to_dict_contains_required_keys(self):
        r = _make_webcam_result()
        d = r.to_dict()
        assert set(d.keys()) >= {"request_id", "frame_index", "timestamp_ms", "faces"}

    def test_empty_faces_is_valid(self):
        r = WebcamFrameResult(
            request_id="empty",
            frame_index=0,
            timestamp_ms=0,
            faces=[],
        )
        assert r.to_dict()["faces"] == []
