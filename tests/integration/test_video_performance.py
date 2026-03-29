"""Video latency and timestamp-alignment validation tests (US4).

These tests verify:
- The sampling logic produces timestamps that are evenly spaced within tolerance.
- Frame timestamps increase monotonically.
- The sampling function handles edge cases without hanging.

No real video file is needed — synthetic frames are injected via mocking.
"""

from __future__ import annotations

import time
from unittest.mock import patch

import numpy as np
import pytest

from facial_emotions.services.video_analysis_service import sample_frame_timestamps


class TestTimestampAlignment:
    """Verify that sampled timestamps are correctly spaced."""

    def test_timestamps_evenly_spaced_within_tolerance(self):
        """At 2 FPS sampling from 25 FPS, each step should be ~500 ms."""
        result = sample_frame_timestamps(total_frames=250, video_fps=25.0, sample_fps=2.0)
        assert len(result) >= 2
        gaps = [result[i + 1] - result[i] for i in range(len(result) - 1)]
        # Allow ±50 ms tolerance due to rounding
        for gap in gaps:
            assert 400 <= gap <= 600, f"Unexpected gap: {gap} ms"

    def test_timestamps_monotonically_increasing(self):
        result = sample_frame_timestamps(total_frames=300, video_fps=30.0, sample_fps=3.0)
        assert result == sorted(result)
        assert result == list(dict.fromkeys(result)), "Duplicate timestamps found"

    def test_first_timestamp_is_zero(self):
        result = sample_frame_timestamps(total_frames=100, video_fps=25.0, sample_fps=2.0)
        assert result[0] == 0

    def test_last_timestamp_within_video_duration(self):
        total_frames = 250
        video_fps = 25.0
        duration_ms = int(total_frames / video_fps * 1000)
        result = sample_frame_timestamps(total_frames=total_frames, video_fps=video_fps, sample_fps=2.0)
        assert result[-1] <= duration_ms

    def test_1fps_sample_step_is_approx_1000ms(self):
        result = sample_frame_timestamps(total_frames=300, video_fps=30.0, sample_fps=1.0)
        gaps = [result[i + 1] - result[i] for i in range(len(result) - 1)]
        for gap in gaps:
            assert 900 <= gap <= 1100, f"Unexpected gap: {gap} ms"


class TestVideoPipelineLatency:
    """Verify the pipeline processes a synthetic video within time budgets."""

    def test_pipeline_processes_short_video_quickly(self):
        """A 10-frame synthetic video should be processed in under 5 seconds."""
        from facial_emotions.pipelines.video_pipeline import VideoAnalysisPipeline
        from unittest.mock import MagicMock

        mock_detector = MagicMock()
        mock_detector.detect.return_value = []
        mock_emotion = MagicMock()
        mock_landmark = MagicMock()
        mock_genuine = MagicMock()
        mock_renderer = MagicMock(side_effect=lambda img, faces: img)

        pipeline = VideoAnalysisPipeline(
            face_detector=mock_detector,
            emotion_service=mock_emotion,
            landmark_extractor=mock_landmark,
            genuineness_service=mock_genuine,
            overlay_renderer=mock_renderer,
        )

        blank_frame = np.zeros((240, 320, 3), dtype=np.uint8)
        timestamps = [i * 500 for i in range(10)]
        frames = list(zip(timestamps, [blank_frame] * 10))

        start = time.perf_counter()
        with patch.object(pipeline, "_read_sampled_frames", return_value=frames):
            result = pipeline.analyse(video_path="fake.mp4", request_id="perf-test")
        elapsed = time.perf_counter() - start

        assert elapsed < 5.0, f"Pipeline too slow: {elapsed:.2f}s"
        assert len(result.sampled_frames) == 10

    def test_empty_video_returns_immediately(self):
        from facial_emotions.pipelines.video_pipeline import VideoAnalysisPipeline
        from unittest.mock import MagicMock

        pipeline = VideoAnalysisPipeline(
            face_detector=MagicMock(),
            emotion_service=MagicMock(),
            landmark_extractor=MagicMock(),
            genuineness_service=MagicMock(),
            overlay_renderer=MagicMock(),
        )
        with patch.object(pipeline, "_read_sampled_frames", return_value=[]):
            with patch.object(pipeline, "_get_duration_seconds", return_value=0.0):
                result = pipeline.analyse(video_path="empty.mp4", request_id="empty-test")

        assert result.sampled_frames == []
        assert result.duration_seconds == 0.0
