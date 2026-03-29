"""Genuine-versus-posed expression assessment.

Uses Duchenne-style landmark heuristics:
- Smile symmetry  (genuine smiles are symmetric)
- Mouth curvature (corners raised = genuine happiness)
- Eye Aspect Ratio (eyes narrow slightly in a genuine smile)
- Cheek raise score (orbital muscle engagement)

A score above GENUINE_THRESHOLD is labelled "genuine"; below
POSED_THRESHOLD is labelled "posed"; in between is "uncertain".
"""

from __future__ import annotations

from typing import List

from facial_emotions.domain.enums import GenuinenessState
from facial_emotions.domain.models import (
    EmotionPrediction,
    GenuinenessAssessment,
    LandmarkSet,
)

# Tunable thresholds
GENUINE_THRESHOLD: float = 0.55
POSED_THRESHOLD: float = 0.35

# EAR: genuine smiles cause slight eye narrowing
EAR_SMILE_MAX: float = 0.30          # below this = eyes narrowed (good sign for genuine)
EAR_NEUTRAL_MAX: float = 0.38        # above this = eyes wide open

# Mouth curvature: negative = corners down, positive = corners up
MOUTH_CURVE_POSITIVE: float = 0.0    # any positive value means smile

# Symmetry: high value = symmetric
SYMMETRY_FLOOR: float = 0.70


def _score_landmarks(lm: LandmarkSet, emotion: str) -> tuple[float, list[str]]:
    """Return (0.0–1.0 score, list of triggered heuristic flags)."""
    score = 0.0
    flags: list[str] = []

    is_happy = emotion in ("happiness",)
    is_negative = emotion in ("sadness", "anger", "disgust", "fear")

    # Smile symmetry
    if lm.smile_symmetry_score >= SYMMETRY_FLOOR:
        score += 0.25
        flags.append("symmetric_smile")

    # Mouth curvature
    if is_happy and lm.mouth_curvature < MOUTH_CURVE_POSITIVE:
        score += 0.25
        flags.append("mouth_corners_raised")
    elif is_negative and lm.mouth_curvature > 0:
        # Corners are raised yet emotion is negative → posed signal
        score -= 0.15
        flags.append("incongruent_mouth_corner_raise")

    # Eye narrowing (Duchenne marker)
    if lm.eye_aspect_ratio < EAR_SMILE_MAX:
        score += 0.25
        flags.append("eye_narrowing_detected")
    elif lm.eye_aspect_ratio > EAR_NEUTRAL_MAX and is_happy:
        # Eyes wide open during a "happiness" prediction → likely posed
        score -= 0.10
        flags.append("eyes_wide_during_smile")

    # Cheek raise
    if lm.cheek_raise_score > 0.1:
        score += 0.25
        flags.append("cheek_raise_detected")

    return max(0.0, min(1.0, score)), flags


class GenuinenessAssessmentService:
    """Assigns genuine/posed/uncertain labels to detected faces."""

    def assess(
        self,
        faces_landmarks: List[LandmarkSet],
        emotions: List[EmotionPrediction],
    ) -> List[GenuinenessAssessment]:
        # Build a quick lookup: face_id → emotion label
        emotion_map = {e.face_id: e.emotion_label.value for e in emotions}

        results: List[GenuinenessAssessment] = []
        for lm in faces_landmarks:
            emotion_str = emotion_map.get(lm.face_id, "neutral")
            raw_score, flags = _score_landmarks(lm, emotion_str)

            if raw_score >= GENUINE_THRESHOLD:
                state = GenuinenessState.GENUINE
            elif raw_score <= POSED_THRESHOLD:
                state = GenuinenessState.POSED
            else:
                state = GenuinenessState.UNCERTAIN

            results.append(
                GenuinenessAssessment(
                    face_id=lm.face_id,
                    predicted_state=state,
                    confidence=round(raw_score, 4),
                    reasoning_flags=flags,
                    classifier_version="heuristic-v1",
                )
            )
        return results
