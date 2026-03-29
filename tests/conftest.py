"""Shared pytest fixtures."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import List
from unittest.mock import MagicMock

import numpy as np
import pytest

from facial_emotions.domain.enums import (
    EmotionLabel,
    GenuinenessState,
    InputMode,
    MediaFormat,
    MediaType,
)
from facial_emotions.domain.models import (
    AnalysisRequest,
    EmotionPrediction,
    FaceDetection,
    FrameAnalysis,
    GenuinenessAssessment,
    InputAsset,
    LandmarkSet,
)

FIXTURES_DIR = Path(__file__).parent.parent / "assets" / "fixtures"
IMAGES_DIR = FIXTURES_DIR / "images"
VIDEOS_DIR = FIXTURES_DIR / "videos"


# ---------------------------------------------------------------------------
# Image fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def blank_rgb_image() -> np.ndarray:
    """224×224 black RGB frame — zero-content placeholder."""
    import cv2
    return np.zeros((224, 224, 3), dtype=np.uint8)


@pytest.fixture()
def sample_bgr_image() -> np.ndarray:
    """640×480 synthetic BGR image with a face-coloured ellipse."""
    import cv2
    img = np.full((480, 640, 3), 60, dtype=np.uint8)
    cv2.ellipse(img, (320, 240), (80, 100), 0, 0, 360, (200, 180, 160), -1)
    return img


# ---------------------------------------------------------------------------
# Domain object fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def sample_face_detection() -> FaceDetection:
    return FaceDetection(
        face_id="face-001",
        bbox_x=100,
        bbox_y=80,
        bbox_width=120,
        bbox_height=140,
        detection_confidence=0.92,
    )


@pytest.fixture()
def sample_landmark_set(sample_face_detection: FaceDetection) -> LandmarkSet:
    pts = [(float(i), float(i), 0.0) for i in range(478)]
    return LandmarkSet(
        face_id=sample_face_detection.face_id,
        landmark_points=pts,
        eye_aspect_ratio=0.28,
        mouth_curvature=-0.05,
        smile_symmetry_score=0.85,
        cheek_raise_score=0.15,
    )


@pytest.fixture()
def sample_emotion_prediction(sample_face_detection: FaceDetection) -> EmotionPrediction:
    top_k = {e.value: 1 / 7 for e in EmotionLabel}
    top_k[EmotionLabel.HAPPINESS.value] = 0.75
    return EmotionPrediction(
        face_id=sample_face_detection.face_id,
        emotion_label=EmotionLabel.HAPPINESS,
        confidence=0.75,
        top_k_scores=top_k,
    )


@pytest.fixture()
def sample_genuineness() -> GenuinenessAssessment:
    return GenuinenessAssessment(
        face_id="face-001",
        predicted_state=GenuinenessState.GENUINE,
        confidence=0.72,
        reasoning_flags=["symmetric_smile", "cheek_raise_detected"],
        classifier_version="heuristic-v1",
    )


@pytest.fixture()
def sample_frame_analysis(
    sample_face_detection: FaceDetection,
    sample_landmark_set: LandmarkSet,
    sample_emotion_prediction: EmotionPrediction,
    sample_genuineness: GenuinenessAssessment,
) -> FrameAnalysis:
    return FrameAnalysis(
        frame_id="frame-001",
        timestamp_ms=0,
        faces=[sample_face_detection],
        landmarks=[sample_landmark_set],
        emotions=[sample_emotion_prediction],
        genuineness=[sample_genuineness],
    )


@pytest.fixture()
def sample_analysis_request() -> AnalysisRequest:
    return AnalysisRequest(
        request_id=str(uuid.uuid4()),
        input_mode=InputMode.IMAGE,
        source_path=str(IMAGES_DIR / "single_face.jpg"),
        enable_genuineness_check=True,
        max_faces=5,
    )


@pytest.fixture()
def sample_input_asset() -> InputAsset:
    return InputAsset(
        asset_id=str(uuid.uuid4()),
        media_type=MediaType.IMAGE,
        format=MediaFormat.JPEG,
        width=640,
        height=480,
    )


# ---------------------------------------------------------------------------
# Mock emotion inference service (avoids ONNX model requirement in unit tests)
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_emotion_service(sample_emotion_prediction: EmotionPrediction):
    svc = MagicMock()
    svc.predict.return_value = [sample_emotion_prediction]
    svc.predict_single_face.return_value = sample_emotion_prediction
    return svc


@pytest.fixture()
def mock_face_detector(sample_face_detection: FaceDetection):
    det = MagicMock()
    det.detect.return_value = [sample_face_detection]
    return det
