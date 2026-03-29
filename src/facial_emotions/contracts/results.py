"""Contract-layer result schemas.

These are plain serialisable structures returned from service and pipeline
boundaries.  They are intentionally independent of any UI framework so that
Gradio, tests, and future integrations can all consume them without coupling
to internal domain models.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional


@dataclass
class FaceResult:
    face_id: str
    bbox: dict[str, int]  # {"x": int, "y": int, "width": int, "height": int}
    detection_confidence: float
    emotion: str
    emotion_confidence: float
    top_k_emotions: dict[str, float]
    genuineness_state: str
    genuineness_confidence: float
    genuineness_flags: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FrameResult:
    frame_id: str
    timestamp_ms: int
    faces: list[FaceResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ImageAnalysisResult:
    request_id: str
    image_width: int
    image_height: int
    faces: list[FaceResult] = field(default_factory=list)
    annotated_image: Optional[Any] = None  # numpy array or PIL Image

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("annotated_image", None)  # not serialisable
        return d


@dataclass
class VideoAnalysisResult:
    request_id: str
    video_path: str
    duration_seconds: float
    sampled_frames: list[FrameResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class WebcamFrameResult:
    request_id: str
    frame_index: int
    timestamp_ms: int
    faces: list[FaceResult] = field(default_factory=list)
    annotated_frame: Optional[Any] = None  # numpy array

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("annotated_frame", None)
        return d
