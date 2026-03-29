"""Domain dataclasses mirroring the data model specification."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from facial_emotions.domain.enums import (
    EmotionLabel,
    GenuinenessState,
    InputMode,
    MediaFormat,
    MediaType,
)


@dataclass
class AnalysisRequest:
    request_id: str
    input_mode: InputMode
    enable_genuineness_check: bool = True
    source_path: Optional[str] = None
    stream_id: Optional[str] = None
    sampling_rate_fps: Optional[float] = None
    max_faces: int = 5

    def __post_init__(self) -> None:
        if self.input_mode in (InputMode.IMAGE, InputMode.VIDEO) and not self.source_path:
            raise ValueError("source_path is required for image and video modes")
        if self.input_mode == InputMode.WEBCAM and not self.stream_id:
            raise ValueError("stream_id is required for webcam mode")
        if not (1 <= self.max_faces <= 10):
            raise ValueError("max_faces must be between 1 and 10")


@dataclass
class InputAsset:
    asset_id: str
    media_type: MediaType
    format: MediaFormat
    width: int
    height: int
    duration_seconds: Optional[float] = None
    frame_rate: Optional[float] = None

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("dimensions must be positive integers")


@dataclass
class FaceDetection:
    face_id: str
    bbox_x: int
    bbox_y: int
    bbox_width: int
    bbox_height: int
    detection_confidence: float

    def __post_init__(self) -> None:
        if not (0.0 <= self.detection_confidence <= 1.0):
            raise ValueError("detection_confidence must be between 0.0 and 1.0")


@dataclass
class LandmarkSet:
    face_id: str
    landmark_points: list  # list of (x, y) or (x, y, z) tuples
    eye_aspect_ratio: float
    mouth_curvature: float
    smile_symmetry_score: float
    cheek_raise_score: float

    def __post_init__(self) -> None:
        if not self.landmark_points:
            raise ValueError("landmark_points cannot be empty")


@dataclass
class EmotionPrediction:
    face_id: str
    emotion_label: EmotionLabel
    confidence: float
    top_k_scores: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("confidence must be between 0.0 and 1.0")


@dataclass
class GenuinenessAssessment:
    face_id: str
    predicted_state: GenuinenessState
    confidence: float
    reasoning_flags: list[str] = field(default_factory=list)
    classifier_version: str = "heuristic-v1"


@dataclass
class FrameAnalysis:
    frame_id: str
    timestamp_ms: int
    faces: list[FaceDetection] = field(default_factory=list)
    landmarks: list[LandmarkSet] = field(default_factory=list)
    emotions: list[EmotionPrediction] = field(default_factory=list)
    genuineness: list[GenuinenessAssessment] = field(default_factory=list)


@dataclass
class AnalysisSession:
    request: AnalysisRequest
    asset: InputAsset
    analyzed_frames: list[FrameAnalysis] = field(default_factory=list)
