"""Webcam capture orchestration, frame throttling, and session management."""

from __future__ import annotations

import time
from typing import Optional

from facial_emotions import config


def should_process_frame(
    frame_index: int,
    last_processed_ms: Optional[int],
    target_fps: float = config.WEBCAM_TARGET_FPS,
) -> bool:
    """Return True if *frame_index* should be processed given current timing.

    Always processes the first frame (frame_index == 0 or last_processed_ms is
    None).  After that, only processes when the elapsed time since the last
    processed frame meets the target FPS budget.  If *target_fps* is zero or
    negative, every frame is processed.
    """
    if frame_index == 0 or last_processed_ms is None:
        return True
    if target_fps <= 0:
        return True
    budget_ms = 1000.0 / target_fps
    elapsed_ms = int(time.monotonic() * 1000) - last_processed_ms
    return elapsed_ms >= budget_ms


class WebcamSession:
    """Manages state across successive webcam frames."""

    def __init__(self, target_fps: float = config.WEBCAM_TARGET_FPS) -> None:
        self._target_fps = target_fps
        self._last_processed_ms: Optional[int] = None
        self._frame_count = 0

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def tick(self) -> bool:
        """Advance the internal counter and return True if this frame should be processed."""
        process = should_process_frame(
            frame_index=self._frame_count,
            last_processed_ms=self._last_processed_ms,
            target_fps=self._target_fps,
        )
        if process:
            self._last_processed_ms = int(time.monotonic() * 1000)
        self._frame_count += 1
        return process
