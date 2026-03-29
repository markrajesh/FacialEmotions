"""Static-image latency validation — must complete within 2 seconds (SC-002).

Uses mocked services so the test validates pipeline overhead, not model I/O.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

IMAGE_ANALYSIS_BUDGET_S = 2.0


class TestImageAnalysisLatency:
    def test_single_face_analysis_under_2_seconds(
        self, mock_face_detector, mock_emotion_service
    ):
        from facial_emotions.pipelines.image_pipeline import ImageAnalysisPipeline

        pipeline = ImageAnalysisPipeline(
            face_detector=mock_face_detector,
            emotion_service=mock_emotion_service,
        )
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        start = time.perf_counter()
        pipeline.analyse(image, request_id="lat-001")
        elapsed = time.perf_counter() - start
        assert elapsed < IMAGE_ANALYSIS_BUDGET_S, (
            f"Image analysis took {elapsed:.3f}s, exceeding {IMAGE_ANALYSIS_BUDGET_S}s budget"
        )

    def test_five_face_analysis_under_2_seconds(self, mock_emotion_service):
        """Pipeline with 5 mocked faces must still complete within budget."""
        from unittest.mock import MagicMock

        from facial_emotions.domain.models import FaceDetection
        from facial_emotions.pipelines.image_pipeline import ImageAnalysisPipeline

        faces = [
            FaceDetection(
                face_id=f"face-{i:03d}",
                bbox_x=i * 10, bbox_y=0, bbox_width=80, bbox_height=80,
                detection_confidence=0.9,
            )
            for i in range(5)
        ]
        multi_detector = MagicMock()
        multi_detector.detect.return_value = faces

        from facial_emotions.domain.enums import EmotionLabel
        from facial_emotions.domain.models import EmotionPrediction

        def make_pred(face):
            return EmotionPrediction(
                face_id=face.face_id,
                emotion_label=EmotionLabel.HAPPINESS,
                confidence=0.75,
                top_k_scores={},
            )

        multi_emotion = MagicMock()
        multi_emotion.predict.side_effect = lambda img, fs: [make_pred(f) for f in fs]

        pipeline = ImageAnalysisPipeline(
            face_detector=multi_detector,
            emotion_service=multi_emotion,
        )
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        start = time.perf_counter()
        pipeline.analyse(image, request_id="lat-5face")
        elapsed = time.perf_counter() - start
        assert elapsed < IMAGE_ANALYSIS_BUDGET_S
