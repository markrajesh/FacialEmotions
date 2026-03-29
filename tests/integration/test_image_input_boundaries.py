"""Image input boundary tests (FR-006).

Verifies that media_io correctly accepts or rejects images based on size
constraints: minimum side ≥ 224px, maximum 1920×1080.
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

    # ---- Maximum boundary ----

    def test_1920x1080_image_accepted(self):
        img = np.zeros((1080, 1920, 3), dtype=np.uint8)
        validate_image_dimensions(img)

    def test_image_exceeding_max_width_rejected(self):
        img = np.zeros((1080, 1921, 3), dtype=np.uint8)
        with pytest.raises(MediaIOError, match="too large"):
            validate_image_dimensions(img)

    def test_image_exceeding_max_height_rejected(self):
        img = np.zeros((1081, 1920, 3), dtype=np.uint8)
        with pytest.raises(MediaIOError, match="too large"):
            validate_image_dimensions(img)

    def test_large_portrait_within_bounds_accepted(self):
        # 720 × 1080 — within bounds
        img = np.zeros((1080, 720, 3), dtype=np.uint8)
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
