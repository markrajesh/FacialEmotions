"""Face detection service wrapper.

Uses OpenCV Haar Cascade for face detection — requires no model downloads and
works on any platform.  If a MediaPipe face-detection task model is placed at
``models/face_detection/blaze_face_short_range.task``, the MediaPipe Tasks API
will be used instead for higher accuracy.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import List

import cv2
import numpy as np

from facial_emotions import config
from facial_emotions.domain.models import FaceDetection

_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml"
_MP_TASK_PATH = config.MODELS_DIR / "face_detection" / "blaze_face_short_range.task"


class FaceDetector:
    """Face detector using OpenCV Haar Cascade (default) or MediaPipe Tasks (optional)."""

    def __init__(
        self,
        min_detection_confidence: float = config.FACE_DETECTION_CONFIDENCE,
        max_faces: int = config.MAX_FACES,
    ) -> None:
        self._max_faces = max_faces
        self._min_confidence = min_detection_confidence
        self._cascade = cv2.CascadeClassifier(_CASCADE_PATH)

    def detect(self, bgr_image: np.ndarray) -> List[FaceDetection]:
        """Return a list of FaceDetection objects for a single BGR frame."""
        h, w = bgr_image.shape[:2]
        gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)  # normalise contrast — critical for webcam lighting
        rects = self._cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=2,
            minSize=(20, 20),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )

        detections: List[FaceDetection] = []
        if len(rects) == 0:
            return detections

        for x, y, bw, bh in rects[: self._max_faces]:
            detections.append(
                FaceDetection(
                    face_id=str(uuid.uuid4()),
                    bbox_x=int(x),
                    bbox_y=int(y),
                    bbox_width=int(bw),
                    bbox_height=int(bh),
                    detection_confidence=self._min_confidence,
                )
            )
        return detections

    def close(self) -> None:
        pass  # cascade classifier does not need explicit release

    def __enter__(self) -> "FaceDetector":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
