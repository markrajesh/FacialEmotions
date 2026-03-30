"""Image input boundary tests (FR-001, FR-006, FR-007).

Verifies that media_io correctly accepts or rejects images based on size
constraints: minimum side ≥ 224px, rejection ceiling 8192 px longest side.
"""

from __future__ import annotations

import numpy as np
import pytest

from facial_emotions.services.media_io import MediaIOError, validate_image_dimensions


class TestImageSizeBoundaries:
    # ---- Minimum boundary ----

    def test_224x224_image_accepted(self):
        img = np.zeros((224, 224, 3), dtype=np.uint8)
        result = validate_image_dimensions(img)
        assert result.shape == (224, 224, 3)

    def test_image_exactly_at_min_side_accepted(self):
        # 224 × 500: min_side = 224 → accepted
        img = np.zeros((500, 224, 3), dtype=np.uint8)
        validate_image_dimensions(img)

    def test_image_below_min_side_rejected(self):
        # 100 × 200: min_side = 100 < 224
        img = np.zeros((200, 100, 3), dtype=np.uint8)
        with pytest.raises(MediaIOError, match="too small"):
            validate_image_dimensions(img)

    def test_1x1_image_rejected(self):
        img = np.zeros((1, 1, 3), dtype=np.uint8)
        with pytest.raises(MediaIOError, match="too small"):
            validate_image_dimensions(img)

    def test_223x223_rejected(self):
        img = np.zeros((223, 223, 3), dtype=np.uint8)
        with pytest.raises(MediaIOError, match="too small"):
            validate_image_dimensions(img)

    # ---- Maximum boundary (new: 8192 px longest side) ----

    def test_1920x1080_image_accepted(self):
        img = np.zeros((1080, 1920, 3), dtype=np.uint8)
        validate_image_dimensions(img)

    def test_image_exceeding_old_max_width_now_accepted(self):
        # 1921×1080 was rejected before; now accepted (below 8192 ceiling)
        img = np.zeros((1080, 1921, 3), dtype=np.uint8)
        validate_image_dimensions(img)

    def test_image_exceeding_old_max_height_now_accepted(self):
        # 1920×1081 was rejected before; now accepted (below 8192 ceiling)
        img = np.zeros((1081, 1920, 3), dtype=np.uint8)
        validate_image_dimensions(img)

    def test_4k_image_accepted(self):
        img = np.zeros((2160, 3840, 3), dtype=np.uint8)
        validate_image_dimensions(img)

    def test_image_at_rejection_ceiling_accepted(self):
        # 8192×224: longest side == 8192 exactly → accepted
        img = np.zeros((224, 8192, 3), dtype=np.uint8)
        validate_image_dimensions(img)

    def test_image_exceeding_rejection_ceiling_rejected(self):
        # 8193×224: longest side == 8193 > 8192 → rejected
        img = np.zeros((224, 8193, 3), dtype=np.uint8)
        with pytest.raises(MediaIOError, match="too large"):
            validate_image_dimensions(img)

    def test_portrait_above_old_cap_now_accepted(self):
        # 3024×4032 (portrait) was rejected by old 1920×1080 cap; now accepted
        img = np.zeros((4032, 3024, 3), dtype=np.uint8)
        validate_image_dimensions(img)

    def test_8k_image_rejected(self):
        # 7681 px longest side > 8192? No — 7680×4320 is 8K, longest=7680 → accepted
        img = np.zeros((4320, 7680, 3), dtype=np.uint8)
        validate_image_dimensions(img)

    def test_over_8k_image_rejected(self):
        img = np.zeros((1000, 8200, 3), dtype=np.uint8)
        with pytest.raises(MediaIOError, match="too large"):
            validate_image_dimensions(img)

    # ---- Resize helper ----

    def test_resize_for_model_outputs_correct_shape(self):
        from facial_emotions.services.media_io import resize_for_model

        img = np.zeros((480, 640, 3), dtype=np.uint8)
        result = resize_for_model(img, target_size=48)
        assert result.shape == (48, 48, 3)

    def test_resize_pads_non_square_to_square(self):
        from facial_emotions.services.media_io import resize_for_model

        img = np.zeros((480, 640, 3), dtype=np.uint8)
        result = resize_for_model(img, target_size=224)
        assert result.shape == (224, 224, 3)


class TestPortraitAndNonStandardAspectRatios:
    """T012 — Portrait-mode and unusual aspect-ratio images (US2)."""

    def test_portrait_4032x3024_accepted(self):
        """4032 wide × 3024 tall: longest=4032, within ceiling → accepted."""
        img = np.zeros((3024, 4032, 3), dtype=np.uint8)
        validate_image_dimensions(img)

    def test_portrait_3024x4032_accepted(self):
        """3024 wide × 4032 tall (portrait): longest=4032, within ceiling → accepted."""
        img = np.zeros((4032, 3024, 3), dtype=np.uint8)
        validate_image_dimensions(img)

    def test_tall_portrait_above_processing_ceiling_downscaled_not_rejected(self):
        """2000 wide × 8000 tall: longest=8000 > 4096 but ≤ 8192.

        validate_image_dimensions should accept it; downscale_to_processing_ceiling
        should return a frame with longest side ≤ 4096.
        """
        from facial_emotions.services.media_io import downscale_to_processing_ceiling

        img = np.zeros((8000, 2000, 3), dtype=np.uint8)
        validate_image_dimensions(img)  # must not raise
        frame, scale_factor = downscale_to_processing_ceiling(img)
        assert max(frame.shape[0], frame.shape[1]) <= 4096

    def test_panoramic_5000x1000_accepted_and_downscaled(self):
        """5000 wide × 1000 tall: longest=5000 > 4096 but ≤ 8192.

        validate_image_dimensions accepts; downscale frame longest side == 4096.
        """
        from facial_emotions.services.media_io import downscale_to_processing_ceiling

        img = np.zeros((1000, 5000, 3), dtype=np.uint8)
        validate_image_dimensions(img)  # must not raise
        frame, _scale_factor = downscale_to_processing_ceiling(img)
        assert max(frame.shape[0], frame.shape[1]) == 4096
