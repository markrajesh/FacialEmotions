"""Unit tests for genuineness assessment heuristics.

These cover the Duchenne-style landmark feature rules defined in
genuineness_assessment.py: mouth curvature, smile symmetry, eye narrowing,
and cheek raise.
"""

from __future__ import annotations

import pytest

from facial_emotions.domain.enums import EmotionLabel, GenuinenessState
from facial_emotions.domain.models import EmotionPrediction, LandmarkSet
from facial_emotions.services.genuineness_assessment import GenuinenessAssessmentService


def _make_landmark(
    face_id: str = "face-001",
    ear: float = 0.28,
    mouth_curve: float = -0.05,
    symmetry: float = 0.85,
    cheek_raise: float = 0.15,
) -> LandmarkSet:
    pts = [(float(i), float(i), 0.0) for i in range(478)]
    return LandmarkSet(
        face_id=face_id,
        landmark_points=pts,
        eye_aspect_ratio=ear,
        mouth_curvature=mouth_curve,
        smile_symmetry_score=symmetry,
        cheek_raise_score=cheek_raise,
    )


def _make_emotion(face_id: str, label: EmotionLabel) -> EmotionPrediction:
    return EmotionPrediction(
        face_id=face_id,
        emotion_label=label,
        confidence=0.8,
        top_k_scores={},
    )


class TestGenuinenessHeuristics:
    def test_genuine_smile_scores_genuine(self):
        svc = GenuinenessAssessmentService()
        lm = _make_landmark(ear=0.25, mouth_curve=-0.1, symmetry=0.9, cheek_raise=0.2)
        emo = _make_emotion("face-001", EmotionLabel.HAPPINESS)
        results = svc.assess([lm], [emo])
        assert len(results) == 1
        assert results[0].predicted_state == GenuinenessState.GENUINE

    def test_posed_smile_wide_eyes_scores_lower(self):
        svc = GenuinenessAssessmentService()
        # Wide eyes + no cheek raise + low symmetry = posed-ish
        lm = _make_landmark(ear=0.42, mouth_curve=0.05, symmetry=0.4, cheek_raise=0.0)
        emo = _make_emotion("face-001", EmotionLabel.HAPPINESS)
        results = svc.assess([lm], [emo])
        assert results[0].predicted_state in (GenuinenessState.POSED, GenuinenessState.UNCERTAIN)

    def test_reasoning_flags_populated(self):
        svc = GenuinenessAssessmentService()
        lm = _make_landmark(ear=0.25, mouth_curve=-0.1, symmetry=0.9, cheek_raise=0.2)
        emo = _make_emotion("face-001", EmotionLabel.HAPPINESS)
        results = svc.assess([lm], [emo])
        assert len(results[0].reasoning_flags) > 0

    def test_symmetric_smile_flag_triggered(self):
        svc = GenuinenessAssessmentService()
        lm = _make_landmark(symmetry=0.9)
        emo = _make_emotion("face-001", EmotionLabel.HAPPINESS)
        results = svc.assess([lm], [emo])
        assert "symmetric_smile" in results[0].reasoning_flags

    def test_eye_narrowing_flag_triggered(self):
        svc = GenuinenessAssessmentService()
        lm = _make_landmark(ear=0.22)
        emo = _make_emotion("face-001", EmotionLabel.HAPPINESS)
        results = svc.assess([lm], [emo])
        assert "eye_narrowing_detected" in results[0].reasoning_flags

    def test_cheek_raise_flag_triggered(self):
        svc = GenuinenessAssessmentService()
        lm = _make_landmark(cheek_raise=0.2)
        emo = _make_emotion("face-001", EmotionLabel.HAPPINESS)
        results = svc.assess([lm], [emo])
        assert "cheek_raise_detected" in results[0].reasoning_flags

    def test_classifier_version_set(self):
        svc = GenuinenessAssessmentService()
        lm = _make_landmark()
        emo = _make_emotion("face-001", EmotionLabel.NEUTRAL)
        results = svc.assess([lm], [emo])
        assert results[0].classifier_version == "heuristic-v1"

    def test_missing_face_in_emotion_map_defaults_neutral(self):
        """Face with no matching emotion entry defaults to neutral scoring."""
        svc = GenuinenessAssessmentService()
        lm = _make_landmark(face_id="face-unknown")
        emo = _make_emotion("face-other", EmotionLabel.HAPPINESS)
        results = svc.assess([lm], [emo])
        assert len(results) == 1
        assert results[0].face_id == "face-unknown"

    def test_multiple_faces_produces_multiple_results(self):
        svc = GenuinenessAssessmentService()
        lm1 = _make_landmark(face_id="f1")
        lm2 = _make_landmark(face_id="f2")
        emo1 = _make_emotion("f1", EmotionLabel.HAPPINESS)
        emo2 = _make_emotion("f2", EmotionLabel.SADNESS)
        results = svc.assess([lm1, lm2], [emo1, emo2])
        assert len(results) == 2
        assert {r.face_id for r in results} == {"f1", "f2"}

    def test_confidence_in_valid_range(self):
        svc = GenuinenessAssessmentService()
        lm = _make_landmark()
        emo = _make_emotion("face-001", EmotionLabel.HAPPINESS)
        results = svc.assess([lm], [emo])
        assert 0.0 <= results[0].confidence <= 1.0
