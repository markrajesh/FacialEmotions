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


class TestDownscalePerformance:
    """T014 — downscale and full-pipeline timing for large images (US3)."""

    def test_downscale_ceiling_completes_under_200ms(self):
        """cv2.INTER_AREA downscale of 8192×6144 must complete in under 200 ms."""
        from facial_emotions.services.media_io import downscale_to_processing_ceiling

        img = np.zeros((6144, 8192, 3), dtype=np.uint8)
        start = time.perf_counter()
        _frame, scale_factor = downscale_to_processing_ceiling(img)
        elapsed = time.perf_counter() - start
        assert scale_factor < 1.0, "Pre-condition: image should have been downscaled"
        assert elapsed < 0.2, (
            f"downscale_to_processing_ceiling took {elapsed:.3f}s, exceeding 200 ms budget"
        )

    def test_full_pipeline_4k_image_under_2s(self, mock_face_detector, mock_emotion_service):
        """End-to-end ImageAnalysisService on a 3840×2160 image must stay under 2 s."""
        from facial_emotions.pipelines.image_pipeline import ImageAnalysisPipeline
        from facial_emotions.services.image_analysis_service import ImageAnalysisService

        pipeline = ImageAnalysisPipeline(
            face_detector=mock_face_detector,
            emotion_service=mock_emotion_service,
        )
        service = ImageAnalysisService(pipeline=pipeline)
        img = np.zeros((2160, 3840, 3), dtype=np.uint8)

        start = time.perf_counter()
        service.analyse_array(img)
        elapsed = time.perf_counter() - start
        assert elapsed < IMAGE_ANALYSIS_BUDGET_S, (
            f"Full 4K pipeline took {elapsed:.3f}s, exceeding {IMAGE_ANALYSIS_BUDGET_S}s budget"
        )
