"""Unit tests for downscale_to_processing_ceiling (FR-002, FR-003, FR-004, FR-011).

Written TDD-style: these tests are written before the implementation exists.
They will fail until T006 (add downscale_to_processing_ceiling) is complete.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from facial_emotions.services.media_io import downscale_to_processing_ceiling


class TestDownscaleToProcessingCeiling:

    def test_image_within_ceiling_returns_same_array_and_scale_1(self):
        """Images with longest side ≤ 4096 are returned unchanged (no copy)."""
        img = np.zeros((1500, 2000, 3), dtype=np.uint8)
        frame, scale = downscale_to_processing_ceiling(img)
        assert scale == 1.0
        assert frame is img  # same object, no copy

    def test_image_at_ceiling_returns_same_array_and_scale_1(self):
        """Image with longest side exactly == 4096 is returned unchanged."""
        img = np.zeros((3072, 4096, 3), dtype=np.uint8)
        frame, scale = downscale_to_processing_ceiling(img)
        assert scale == 1.0
        assert frame is img

    def test_landscape_image_downscaled(self):
        """8192×6144 image (longest=8192) should produce frame with longest side == 4096."""
        img = np.zeros((6144, 8192, 3), dtype=np.uint8)
        frame, scale = downscale_to_processing_ceiling(img)
        assert max(frame.shape[1], frame.shape[0]) == 4096
        assert pytest.approx(scale, rel=1e-3) == 0.5

    def test_portrait_image_downscaled(self):
        """6048×8064 portrait (longest=8064 > 4096) should produce frame with longest side == 4096."""
        img = np.zeros((8064, 6048, 3), dtype=np.uint8)
        frame, scale = downscale_to_processing_ceiling(img)
        assert max(frame.shape[1], frame.shape[0]) == 4096

    def test_scale_factor_is_float_between_0_and_1(self):
        """Scale factor for a downscaled image is strictly in (0, 1)."""
        img = np.zeros((6000, 8000, 3), dtype=np.uint8)
        _, scale = downscale_to_processing_ceiling(img)
        assert 0.0 < scale < 1.0

    def test_downscale_preserves_aspect_ratio_landscape(self):
        """After downscale, aspect ratio is preserved to within 1%."""
        orig_w, orig_h = 8000, 5000
        img = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
        frame, _ = downscale_to_processing_ceiling(img)
        frame_h, frame_w = frame.shape[:2]
        expected_ratio = orig_w / orig_h
        actual_ratio = frame_w / frame_h
        assert abs(actual_ratio - expected_ratio) / expected_ratio < 0.01

    def test_downscale_preserves_aspect_ratio_portrait(self):
        """Portrait downscale preserves aspect ratio to within 1%."""
        orig_w, orig_h = 3000, 6000
        img = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
        frame, _ = downscale_to_processing_ceiling(img)
        frame_h, frame_w = frame.shape[:2]
        expected_ratio = orig_w / orig_h
        actual_ratio = frame_w / frame_h
        assert abs(actual_ratio - expected_ratio) / expected_ratio < 0.01

    def test_debug_log_emitted_on_downscale(self):
        """A DEBUG log is emitted with dimensions and scale factor when downscaling."""
        img = np.zeros((6144, 8192, 3), dtype=np.uint8)
        with patch("facial_emotions.services.media_io._logger") as mock_logger:
            downscale_to_processing_ceiling(img)
            mock_logger.debug.assert_called_once()
            call_args = mock_logger.debug.call_args
            # Message should contain dimension info
            assert "8192" in str(call_args) or "4096" in str(call_args)

    def test_no_debug_log_when_no_downscale(self):
        """No DEBUG log when image is within the ceiling."""
        img = np.zeros((1000, 2000, 3), dtype=np.uint8)
        with patch("facial_emotions.services.media_io._logger") as mock_logger:
            downscale_to_processing_ceiling(img)
            mock_logger.debug.assert_not_called()

    def test_returned_frame_dtype_preserved(self):
        """The downscaled frame preserves the uint8 dtype."""
        img = np.zeros((5000, 7000, 3), dtype=np.uint8)
        frame, _ = downscale_to_processing_ceiling(img)
        assert frame.dtype == np.uint8

    def test_portrait_aspect_ratio_preserved(self):
        """2048×4096 (2:1 portrait ratio w/h) maintains ratio within 1%."""
        orig_w, orig_h = 2048, 4096
        img = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
        frame, _ = downscale_to_processing_ceiling(img)
        frame_h, frame_w = frame.shape[:2]
        expected_ratio = orig_w / orig_h  # ≈ 0.5
        actual_ratio = frame_w / frame_h
        assert abs(actual_ratio - expected_ratio) / expected_ratio < 0.01

    def test_panoramic_aspect_ratio_preserved(self):
        """8192×2048 (4:1 landscape) maintains ratio within 1%."""
        orig_w, orig_h = 8192, 2048
        img = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
        frame, _ = downscale_to_processing_ceiling(img)
        frame_h, frame_w = frame.shape[:2]
        expected_ratio = orig_w / orig_h  # ≈ 4.0
        actual_ratio = frame_w / frame_h
        assert abs(actual_ratio - expected_ratio) / expected_ratio < 0.01
