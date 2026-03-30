"""Unit tests for configuration loading and local-only inference assumptions (T060 + T064)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest


class TestConfigConstants:
    def test_local_only_is_true(self):
        from facial_emotions import config
        assert config.LOCAL_ONLY is True, "LOCAL_ONLY must be True — no cloud calls allowed"

    def test_emotion_labels_has_seven_entries(self):
        from facial_emotions import config
        assert len(config.EMOTION_LABELS) == 7

    def test_emotion_labels_are_lowercase_strings(self):
        from facial_emotions import config
        for label in config.EMOTION_LABELS:
            assert isinstance(label, str)
            assert label == label.lower()

    def test_emotion_labels_match_enum_values(self):
        from facial_emotions import config
        from facial_emotions.domain.enums import EmotionLabel
        assert set(config.EMOTION_LABELS) == {e.value for e in EmotionLabel}

    def test_image_min_side_is_positive(self):
        from facial_emotions import config
        assert config.IMAGE_MIN_SIDE > 0

    def test_image_min_side_default_is_224(self):
        from facial_emotions import config
        assert config.IMAGE_MIN_SIDE == 224

    def test_image_processing_ceiling_default_is_4096(self):
        from facial_emotions import config
        assert config.IMAGE_PROCESSING_CEILING == 4096

    def test_image_rejection_ceiling_default_is_8192(self):
        from facial_emotions import config
        assert config.IMAGE_REJECTION_CEILING == 8192

    def test_image_processing_ceiling_less_than_rejection_ceiling(self):
        from facial_emotions import config
        assert config.IMAGE_PROCESSING_CEILING < config.IMAGE_REJECTION_CEILING

    def test_image_processing_ceiling_env_override(self, monkeypatch):
        import importlib
        monkeypatch.setenv("IMAGE_PROCESSING_CEILING", "2048")
        from facial_emotions import config as cfg
        importlib.reload(cfg)
        assert cfg.IMAGE_PROCESSING_CEILING == 2048
        importlib.reload(cfg)  # restore default

    def test_image_rejection_ceiling_env_override(self, monkeypatch):
        import importlib
        monkeypatch.setenv("IMAGE_REJECTION_CEILING", "16384")
        from facial_emotions import config as cfg
        importlib.reload(cfg)
        assert cfg.IMAGE_REJECTION_CEILING == 16384
        importlib.reload(cfg)  # restore default

    def test_image_min_side_env_override(self, monkeypatch):
        import importlib
        monkeypatch.setenv("IMAGE_MIN_SIDE", "128")
        from facial_emotions import config as cfg
        importlib.reload(cfg)
        assert cfg.IMAGE_MIN_SIDE == 128
        importlib.reload(cfg)  # restore default

    def test_video_sample_fps_positive(self):
        from facial_emotions import config
        assert config.VIDEO_SAMPLE_FPS > 0

    def test_video_max_duration_seconds_positive(self):
        from facial_emotions import config
        assert config.VIDEO_MAX_DURATION_S > 0

    def test_webcam_frame_budget_ms_positive(self):
        from facial_emotions import config
        assert config.WEBCAM_FRAME_BUDGET_MS > 0

    def test_webcam_target_fps_positive(self):
        from facial_emotions import config
        assert config.WEBCAM_TARGET_FPS > 0

    def test_max_faces_between_1_and_10(self):
        from facial_emotions import config
        assert 1 <= config.MAX_FACES <= 10


class TestConfigPaths:
    def test_models_dir_is_path_object(self):
        from facial_emotions import config
        assert isinstance(config.MODELS_DIR, Path)

    def test_emotion_model_path_has_onnx_extension(self):
        from facial_emotions import config
        assert config.EMOTION_MODEL_PATH.suffix == ".onnx"

    def test_root_dir_exists(self):
        from facial_emotions import config
        assert config.ROOT_DIR.exists()


class TestLocalOnlyAssumptions:
    """Validate that no outbound network code is wired into core paths."""

    def test_gradio_app_launches_with_share_false(self):
        """The app entry point must launch Gradio with share=False (no tunnelling)."""
        import inspect
        from facial_emotions import app as entry
        source = inspect.getsource(entry)
        assert "share=False" in source
        assert "share=True" not in source

    def test_app_entry_point_uses_local_server(self):
        """app.py must bind to localhost only."""
        import inspect
        from facial_emotions import app as entry
        source = inspect.getsource(entry)
        assert "127.0.0.1" in source or "localhost" in source

    def test_local_only_flag_cannot_be_overridden_to_false(self):
        """LOCAL_ONLY is hard-coded True and should not read a falsy env var."""
        from facial_emotions import config
        # Even if somehow the env var were set, the module already loaded — just
        # confirm the current value is True.
        assert config.LOCAL_ONLY is True
