"""Unit tests for the Gradio image-analysis handler (analyse_image).

Tests cover:
 (a) Rejection error message propagation when the image is too large.
 (b) Info-notice inserted into summary when a large image is auto-downscaled.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from facial_emotions.contracts.results import ImageAnalysisResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pil(h: int, w: int) -> Image.Image:
    """Return a plain black PIL image of the requested size."""
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _empty_result(w: int = 100, h: int = 100) -> ImageAnalysisResult:
    return ImageAnalysisResult(
        request_id="test",
        image_width=w,
        image_height=h,
        faces=[],
        annotated_image=None,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRejectionError:
    def test_rejection_error_message_contains_too_large(self):
        """Validation fires before the pipeline — no mock needed; 9000×9000 exceeds 8192."""
        from facial_emotions.ui.gradio_app import analyse_image

        img = _pil(9000, 9000)
        returned_image, text = analyse_image(img)

        assert "too large" in text.lower(), f"Expected 'too large' in: {text!r}"
        assert returned_image is not None


class TestDownscaleInfoNotice:
    def test_info_notice_present_when_downscaled(self):
        """A 5000×4000 image exceeds 4096 ceiling → info notice contains 'auto-resized'."""
        from facial_emotions.ui.gradio_app import analyse_image

        img = _pil(4000, 5000)

        mock_result = _empty_result(w=5000, h=4000)

        with patch("facial_emotions.ui.gradio_app._get_image_pipeline") as mock_get:
            mock_pipeline = MagicMock()
            mock_pipeline.analyse.return_value = mock_result
            mock_get.return_value = mock_pipeline

            _returned_image, text = analyse_image(img)

        assert "auto-resized" in text.lower(), f"Expected 'auto-resized' in: {text!r}"

    def test_no_info_notice_for_small_image(self):
        """A 1280×720 image is within ceiling — no downscale notice expected."""
        from facial_emotions.ui.gradio_app import analyse_image

        img = _pil(720, 1280)

        mock_result = _empty_result(w=1280, h=720)

        with patch("facial_emotions.ui.gradio_app._get_image_pipeline") as mock_get:
            mock_pipeline = MagicMock()
            mock_pipeline.analyse.return_value = mock_result
            mock_get.return_value = mock_pipeline

            _returned_image, text = analyse_image(img)

        assert "auto-resized" not in text.lower(), (
            f"Unexpected 'auto-resized' in text for small image: {text!r}"
        )
