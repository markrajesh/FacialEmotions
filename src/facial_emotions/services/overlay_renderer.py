"""Analysis result rendering helpers.

Draws bounding boxes, emotion labels, confidence bars, and
genuineness state indicators onto BGR frames.
"""

from __future__ import annotations

from typing import List, Optional

import cv2
import numpy as np

from facial_emotions.domain.models import (
    EmotionPrediction,
    FaceDetection,
    GenuinenessAssessment,
)

# Colour palette by genuineness state
_COLOUR_GENUINE = (0, 220, 0)      # green
_COLOUR_POSED = (0, 80, 255)       # orange-red
_COLOUR_UNCERTAIN = (180, 180, 0)  # teal/yellow
_COLOUR_DEFAULT = (200, 200, 200)  # grey

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.55
_THICKNESS = 2


def _box_colour(genuineness: Optional[GenuinenessAssessment]) -> tuple:
    if genuineness is None:
        return _COLOUR_DEFAULT
    state = genuineness.predicted_state.value
    return {
        "genuine": _COLOUR_GENUINE,
        "posed": _COLOUR_POSED,
        "uncertain": _COLOUR_UNCERTAIN,
    }.get(state, _COLOUR_DEFAULT)


def draw_results(
    bgr_image: np.ndarray,
    face_detections: List[FaceDetection],
    emotion_predictions: List[EmotionPrediction],
    genuineness_assessments: List[GenuinenessAssessment],
) -> np.ndarray:
    """Return a copy of bgr_image with annotations drawn."""
    canvas = bgr_image.copy()
    emotion_map = {e.face_id: e for e in emotion_predictions}
    genuine_map = {g.face_id: g for g in genuineness_assessments}

    for face in face_detections:
        emo = emotion_map.get(face.face_id)
        gen = genuine_map.get(face.face_id)
        colour = _box_colour(gen)

        x, y, bw, bh = face.bbox_x, face.bbox_y, face.bbox_width, face.bbox_height
        cv2.rectangle(canvas, (x, y), (x + bw, y + bh), colour, _THICKNESS)

        if emo is not None:
            label = f"{emo.emotion_label.value} {emo.confidence:.0%}"
            cv2.putText(
                canvas, label, (x, max(y - 8, 0)),
                _FONT, _FONT_SCALE, colour, _THICKNESS, cv2.LINE_AA,
            )

        if gen is not None:
            g_label = gen.predicted_state.value
            cv2.putText(
                canvas, g_label, (x, y + bh + 18),
                _FONT, _FONT_SCALE * 0.85, colour, 1, cv2.LINE_AA,
            )

    return canvas
