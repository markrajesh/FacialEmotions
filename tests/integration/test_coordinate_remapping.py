"""Integration tests: coordinate remapping when pipeline operates on a downscaled frame.

These tests verify that bounding-box coordinates are mapped back to the original
image's coordinate space when ``ImageAnalysisPipeline.analyse`` receives an
``original_image`` argument.

Tests are written *before* the pipeline supports the ``original_image`` parameter
(TDD — they fail until T008 is implemented).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from facial_emotions.domain.models import FaceDetection, EmotionPrediction
from facial_emotions.domain.enums import EmotionLabel
from facial_emotions.pipelines.image_pipeline import ImageAnalysisPipeline
from facial_emotions.services.media_io import downscale_to_processing_ceiling


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bgr(h: int, w: int) -> np.ndarray:
    """Return a random BGR frame of the requested size."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


def _stub_emotion() -> EmotionPrediction:
    return EmotionPrediction(
        face_id="f1",
        emotion_label=EmotionLabel.NEUTRAL,
        confidence=0.9,
        top_k_scores={"neutral": 0.9},
    )


def _make_mock_detector(face: FaceDetection):
    """Return a FaceDetector stub that always returns [face]."""
    detector = MagicMock()
    detector.detect.return_value = [face]
    return detector


def _make_face_at(x: int, y: int, w: int, h: int) -> FaceDetection:
    return FaceDetection(
        face_id="f1",
        bbox_x=x,
        bbox_y=y,
        bbox_width=w,
        bbox_height=h,
        detection_confidence=0.99,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestOriginalDimensionsReturned:
    """Result dimensions always reflect the *original* image, not the processing frame."""

    def test_analyse_returns_original_dimensions_when_downscaled(self):
        """8000×6000 image → after downscale at 4096 ceiling, result must report 8000×6000."""
        original = _make_bgr(6000, 8000)
        frame, scale_factor = downscale_to_processing_ceiling(original)
        assert scale_factor < 1.0, "Pre-condition: image should have been downscaled"

        pipeline = ImageAnalysisPipeline(render_overlay=False)
        result = pipeline.analyse(frame, original_image=original)

        assert result.image_width == 8000
        assert result.image_height == 6000

    def test_no_downscale_path_result_dimensions_match_input(self):
        """Images at or below 4096 px: original_image=None, dimensions match processing frame."""
        image = _make_bgr(1080, 1920)
        pipeline = ImageAnalysisPipeline(render_overlay=False)
        result = pipeline.analyse(image, original_image=None)

        assert result.image_width == 1920
        assert result.image_height == 1080


class TestBboxCoordinateRemapping:
    """Bounding-box coordinates must be expressed in original image space."""

    def test_bbox_remapped_to_original_coordinates(self):
        """Mock detector returns face at (100, 100, 200, 200) on a 4096-wide frame.

        Original image is 8192 wide × 8192 tall → inv_scale = 2.0.
        Expected remapped bbox: x=200, y=200, width=400, height=400.
        """
        face_on_frame = _make_face_at(x=100, y=100, w=200, h=200)
        detector = _make_mock_detector(face_on_frame)

        # Build mock emotion service so we get a FaceResult back
        emo_service = MagicMock()
        emo_service.predict.return_value = [_stub_emotion()]

        pipeline = ImageAnalysisPipeline(
            face_detector=detector,
            emotion_service=emo_service,
            render_overlay=False,
        )

        # processing frame: 4096×4096; original: 8192×8192 (2× in each axis)
        frame = _make_bgr(4096, 4096)
        original = _make_bgr(8192, 8192)

        result = pipeline.analyse(frame, original_image=original)

        assert len(result.faces) == 1, "Expected exactly one face in result"
        bbox = result.faces[0].bbox
        assert bbox["x"] == 200
        assert bbox["y"] == 200
        assert bbox["width"] == 400

    def test_bbox_remap_accuracy_within_2px(self):
        """For non-integer scale factors, remapped coordinates must be within ±2 px of expected."""
        # Processing frame 4096 wide; original 6000 wide → inv_scale_x ≈ 1.4648
        frame_w, frame_h = 4096, 3000
        orig_w, orig_h = 6000, 4400

        face_on_frame = _make_face_at(x=150, y=200, w=300, h=400)
        detector = _make_mock_detector(face_on_frame)

        emo_service = MagicMock()
        emo_service.predict.return_value = [_stub_emotion()]

        pipeline = ImageAnalysisPipeline(
            face_detector=detector,
            emotion_service=emo_service,
            render_overlay=False,
        )

        frame = _make_bgr(frame_h, frame_w)
        original = _make_bgr(orig_h, orig_w)

        result = pipeline.analyse(frame, original_image=original)

        inv_scale_x = orig_w / frame_w
        inv_scale_y = orig_h / frame_h

        expected_x = round(150 * inv_scale_x)
        expected_y = round(200 * inv_scale_y)
        expected_w = round(300 * inv_scale_x)

        bbox = result.faces[0].bbox
        assert abs(bbox["x"] - expected_x) <= 2
        assert abs(bbox["y"] - expected_y) <= 2
        assert abs(bbox["width"] - expected_w) <= 2


class TestAnnotatedImageDimensions:
    """Annotated image returned in the result must have original-image dimensions."""

    def test_annotated_image_has_original_dimensions(self):
        """When original_image is provided, annotated output must match original shape."""
        face_on_frame = _make_face_at(x=50, y=50, w=100, h=100)
        detector = _make_mock_detector(face_on_frame)

        emo_service = MagicMock()
        emo_service.predict.return_value = [_stub_emotion()]

        pipeline = ImageAnalysisPipeline(
            face_detector=detector,
            emotion_service=emo_service,
            render_overlay=True,
        )

        frame = _make_bgr(2048, 4096)
        original = _make_bgr(4032, 8000)  # distinct shape

        result = pipeline.analyse(frame, original_image=original)

        assert result.annotated_image is not None
        assert result.annotated_image.shape[:2] == (4032, 8000)
