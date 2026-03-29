"""Unit tests for video frame sampling and timestamp generation (US4).

Tests the pure sampling logic independently of I/O or OpenCV.
"""

from __future__ import annotations

import pytest

from facial_emotions.services.video_analysis_service import sample_frame_timestamps


class TestSampleFrameTimestamps:
    def test_returns_empty_for_zero_total_frames(self):
        assert sample_frame_timestamps(total_frames=0, video_fps=25.0, sample_fps=2.0) == []

    def test_returns_empty_for_zero_video_fps(self):
        assert sample_frame_timestamps(total_frames=100, video_fps=0.0, sample_fps=2.0) == []

    def test_single_frame_video(self):
        result = sample_frame_timestamps(total_frames=1, video_fps=25.0, sample_fps=2.0)
        assert result == [0]

    def test_2fps_sample_from_25fps_5s_video(self):
        # 5 seconds * 25 fps = 125 frames; sample every 12 frames ≈ 2 FPS
        result = sample_frame_timestamps(total_frames=125, video_fps=25.0, sample_fps=2.0)
        # Expect ~10 or 11 timestamps (step = round(25/2) = 12)
        assert len(result) >= 9
        assert result[0] == 0

    def test_timestamps_are_monotonically_increasing(self):
        result = sample_frame_timestamps(total_frames=50, video_fps=25.0, sample_fps=2.0)
        assert result == sorted(result)

    def test_timestamps_are_non_negative_integers(self):
        result = sample_frame_timestamps(total_frames=50, video_fps=25.0, sample_fps=2.0)
        assert all(isinstance(t, int) and t >= 0 for t in result)

    def test_sample_fps_exceeding_video_fps_returns_all_frames(self):
        # sample_fps > video_fps means step=1, every frame is sampled
        result = sample_frame_timestamps(total_frames=10, video_fps=5.0, sample_fps=30.0)
        assert len(result) == 10

    def test_1fps_sample_from_30fps_3s_video(self):
        # 3 s * 30 fps = 90 frames; step = round(30/1) = 30
        result = sample_frame_timestamps(total_frames=90, video_fps=30.0, sample_fps=1.0)
        assert len(result) == 3
        assert result[0] == 0

    def test_last_timestamp_does_not_exceed_video_duration(self):
        video_fps = 25.0
        duration_s = 10.0
        total_frames = int(video_fps * duration_s)
        result = sample_frame_timestamps(total_frames=total_frames, video_fps=video_fps, sample_fps=2.0)
        assert result[-1] <= int(duration_s * 1000)
