"""Webcam latency validation against target FPS and per-frame thresholds (US3)."""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import numpy as np
import pytest

from facial_emotions import config
from facial_emotions.pipelines.webcam_pipeline import WebcamAnalysisPipeline
from facial_emotions.services.webcam_session import should_process_frame, WebcamSession


class TestWebcamLatency:
    def test_single_frame_processes_under_budget(self):
        """A single frame with mocked sub-services should complete in <250 ms."""
        mock_detector = MagicMock()
        mock_detector.detect.return_value = []
        pipeline = WebcamAnalysisPipeline(
            face_detector=mock_detector,
            emotion_service=MagicMock(),
            landmark_extractor=MagicMock(),
            genuineness_service=MagicMock(),
            overlay_renderer=MagicMock(side_effect=lambda img, faces: img),
        )
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        start = time.perf_counter()
        pipeline.analyse_frame(frame)
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert elapsed_ms < 250.0, f"Frame processing too slow: {elapsed_ms:.1f} ms"

    def test_ten_consecutive_frames_process_quickly(self):
        """Process 10 sequential frames in under 500 ms total."""
        mock_detector = MagicMock()
        mock_detector.detect.return_value = []
        pipeline = WebcamAnalysisPipeline(
            face_detector=mock_detector,
            emotion_service=MagicMock(),
            landmark_extractor=MagicMock(),
            genuineness_service=MagicMock(),
            overlay_renderer=MagicMock(side_effect=lambda img, faces: img),
        )
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        start = time.perf_counter()
        for _ in range(10):
            pipeline.analyse_frame(frame)
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert elapsed_ms < 500.0, f"10 frames too slow: {elapsed_ms:.1f} ms"


class TestWebcamSessionThrottling:
    def test_session_skips_frames_to_meet_target_fps(self):
        """Rapidly calling tick() should skip most frames at 10 FPS."""
        session = WebcamSession(target_fps=10.0)
        results = [session.tick() for _ in range(50)]
        # First frame always processed, then most others skipped
        assert results[0] is True
        assert results.count(False) > 0

    def test_session_frame_count_increments_regardless_of_skip(self):
        session = WebcamSession(target_fps=10.0)
        for _ in range(5):
            session.tick()
        assert session.frame_count == 5
