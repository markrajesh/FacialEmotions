"""Unit tests for face-to-result alignment (US2).

Verifies that the base_pipeline helpers correctly align face detections with
their corresponding emotion and genuineness results by face_id.
"""

from __future__ import annotations

import pytest

from facial_emotions.domain.enums import EmotionLabel, GenuinenessState
from facial_emotions.domain.models import (
    EmotionPrediction,
    FaceDetection,
    GenuinenessAssessment,
)
from facial_emotions.pipelines.base_pipeline import build_face_result


class TestFaceResultAlignment:
    def _face(self, face_id: str, x: int = 0) -> FaceDetection:
        return FaceDetection(face_id=face_id, bbox_x=x, bbox_y=0, bbox_width=80, bbox_height=80, detection_confidence=0.9)

    def _emotion(self, face_id: str, label=EmotionLabel.HAPPINESS) -> EmotionPrediction:
        return EmotionPrediction(face_id=face_id, emotion_label=label, confidence=0.75, top_k_scores={})

    def _genuine(self, face_id: str, state=GenuinenessState.GENUINE) -> GenuinenessAssessment:
        return GenuinenessAssessment(face_id=face_id, predicted_state=state, confidence=0.7, reasoning_flags=["symmetric_smile"])

    def test_face_result_face_id_matches(self):
        face = self._face("f1")
        emo = self._emotion("f1")
        result = build_face_result(face, emo, None)
        assert result.face_id == "f1"

    def test_face_result_emotion_matches(self):
        face = self._face("f2")
        emo = self._emotion("f2", EmotionLabel.SADNESS)
        result = build_face_result(face, emo, None)
        assert result.emotion == "sadness"

    def test_face_result_bbox_matches(self):
        face = self._face("f3", x=100)
        emo = self._emotion("f3")
        result = build_face_result(face, emo, None)
        assert result.bbox["x"] == 100

    def test_genuineness_populated_when_provided(self):
        face = self._face("f4")
        emo = self._emotion("f4")
        gen = self._genuine("f4", GenuinenessState.POSED)
        result = build_face_result(face, emo, gen)
        assert result.genuineness_state == "posed"
        assert "symmetric_smile" in result.genuineness_flags

    def test_genuineness_defaults_to_uncertain_when_absent(self):
        face = self._face("f5")
        emo = self._emotion("f5")
        result = build_face_result(face, emo, None)
        assert result.genuineness_state == "uncertain"

    def test_multiple_faces_produce_separate_results(self):
        from facial_emotions.pipelines.base_pipeline import build_frame_result
        from facial_emotions.domain.models import FrameAnalysis

        faces = [self._face(f"f{i}", x=i*100) for i in range(3)]
        emotions = [self._emotion(f"f{i}", list(EmotionLabel)[i]) for i in range(3)]
        frame = FrameAnalysis(
            frame_id="frame-001",
            timestamp_ms=0,
            faces=faces,
            landmarks=[],
            emotions=emotions,
            genuineness=[],
        )
        result = build_frame_result(frame)
        assert len(result.faces) == 3
        assert {f.face_id for f in result.faces} == {"f0", "f1", "f2"}

    def test_face_not_in_emotion_map_is_excluded(self):
        from facial_emotions.pipelines.base_pipeline import build_frame_result
        from facial_emotions.domain.models import FrameAnalysis

        face_with_no_emotion = self._face("orphan")
        face_with_emotion = self._face("matched")
        emo = self._emotion("matched")
        frame = FrameAnalysis(
            frame_id="frame-002",
            timestamp_ms=0,
            faces=[face_with_no_emotion, face_with_emotion],
            landmarks=[],
            emotions=[emo],
            genuineness=[],
        )
        result = build_frame_result(frame)
        assert len(result.faces) == 1
        assert result.faces[0].face_id == "matched"
