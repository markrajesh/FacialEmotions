"""Unit tests for webcam frame throttling and refresh behavior (US3).

Tests the pure throttling logic without any camera hardware.
"""

from __future__ import annotations

import time

import pytest

from facial_emotions.services.webcam_session import should_process_frame


class TestFrameThrottling:
    """Verify the frame-gate/throttle logic."""

    def test_first_frame_is_always_processed(self):
        assert should_process_frame(frame_index=0, last_processed_ms=None, target_fps=10.0)

    def test_frame_within_budget_is_skipped(self):
        # target_fps=10 → 100 ms budget; last processed 40 ms ago → skip
        now_ms = int(time.monotonic() * 1000)
        assert not should_process_frame(
            frame_index=1,
            last_processed_ms=now_ms - 40,
            target_fps=10.0,
        )

    def test_frame_after_budget_is_processed(self):
        # target_fps=10 → 100 ms budget; last processed 150 ms ago → process
        now_ms = int(time.monotonic() * 1000)
        assert should_process_frame(
            frame_index=5,
            last_processed_ms=now_ms - 150,
            target_fps=10.0,
        )

    def test_zero_target_fps_processes_every_frame(self):
        now_ms = int(time.monotonic() * 1000)
        assert should_process_frame(frame_index=1, last_processed_ms=now_ms, target_fps=0.0)

    def test_very_high_fps_skips_most_frames_in_rapid_succession(self):
        # Simulate frames arriving at ~1000 FPS (1 ms apart) with target 10 FPS
        now_ms = int(time.monotonic() * 1000)
        results = [
            should_process_frame(
                frame_index=i,
                last_processed_ms=now_ms - i,  # i ms old = recent
                target_fps=10.0,
            )
            for i in range(1, 20)
        ]
        # Most frames should be skipped
        assert results.count(False) > results.count(True)
