"""Video decoding, frame sampling, and timestamp extraction utilities."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from facial_emotions import config
from facial_emotions.contracts.results import VideoAnalysisResult
from facial_emotions.pipelines.video_pipeline import VideoAnalysisPipeline


# ---------------------------------------------------------------------------
# Pure sampling logic — can be tested without I/O
# ---------------------------------------------------------------------------

def sample_frame_timestamps(
    total_frames: int,
    video_fps: float,
    sample_fps: float = config.VIDEO_SAMPLE_FPS,
) -> list[int]:
    """Compute timestamps (ms) for uniformly sampled frames.

    Returns an empty list when *total_frames* or *video_fps* is zero or
    negative so callers never need to guard against division-by-zero.
    """
    if total_frames <= 0 or video_fps <= 0:
        return []
    step = max(1, round(video_fps / sample_fps))
    indices = range(0, total_frames, step)
    return [round(idx / video_fps * 1000) for idx in indices]


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class VideoAnalysisService:
    """Wraps VideoAnalysisPipeline with file validation and path handling."""

    def __init__(self, pipeline: Optional[VideoAnalysisPipeline] = None) -> None:
        self._pipeline = pipeline or VideoAnalysisPipeline()

    def analyse_path(self, video_path: str | Path, request_id: Optional[str] = None) -> VideoAnalysisResult:
        path = Path(video_path)
        if not path.exists():
            raise FileNotFoundError(f"Video not found: {path}")
        if path.suffix.lower() not in {".mp4", ".avi", ".mov", ".mkv", ".webm"}:
            raise ValueError(f"Unsupported video format: {path.suffix}")
        return self._pipeline.analyse(
            video_path=str(path),
            request_id=request_id or str(uuid.uuid4()),
        )
