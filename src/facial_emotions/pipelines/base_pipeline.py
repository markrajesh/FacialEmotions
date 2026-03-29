"""Shared image and frame assembly logic."""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from facial_emotions.domain.models import (
    EmotionPrediction,
    FaceDetection,
    FrameAnalysis,
    GenuinenessAssessment,
    LandmarkSet,
)
from facial_emotions.contracts.results import FaceResult, FrameResult


def build_face_result(
    face: FaceDetection,
    emotion: EmotionPrediction,
    genuineness: Optional[GenuinenessAssessment],
) -> FaceResult:
    """Assemble a serialisable FaceResult from domain objects."""
    return FaceResult(
        face_id=face.face_id,
        bbox={
            "x": face.bbox_x,
            "y": face.bbox_y,
            "width": face.bbox_width,
            "height": face.bbox_height,
        },
        detection_confidence=face.detection_confidence,
        emotion=emotion.emotion_label.value,
        emotion_confidence=emotion.confidence,
        top_k_emotions=emotion.top_k_scores,
        genuineness_state=genuineness.predicted_state.value if genuineness else "uncertain",
        genuineness_confidence=genuineness.confidence if genuineness else 0.0,
        genuineness_flags=genuineness.reasoning_flags if genuineness else [],
    )


def build_frame_result(frame: FrameAnalysis) -> FrameResult:
    """Convert a FrameAnalysis domain object to a contract FrameResult."""
    emotion_map = {e.face_id: e for e in frame.emotions}
    genuine_map = {g.face_id: g for g in frame.genuineness}

    face_results = [
        build_face_result(
            face,
            emotion_map[face.face_id],
            genuine_map.get(face.face_id),
        )
        for face in frame.faces
        if face.face_id in emotion_map
    ]
    return FrameResult(
        frame_id=frame.frame_id,
        timestamp_ms=frame.timestamp_ms,
        faces=face_results,
    )
