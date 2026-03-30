"""Application configuration loaded from environment variables / .env file."""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT_DIR / "models"
EMOTION_MODEL_PATH = MODELS_DIR / "emotion" / os.getenv("EMOTION_MODEL_FILE", "emotion_model.onnx")
ASSETS_DIR = ROOT_DIR / "assets"
FIXTURES_DIR = ASSETS_DIR / "fixtures"

# ---------------------------------------------------------------------------
# Inference settings
# ---------------------------------------------------------------------------
MAX_FACES: int = int(os.getenv("MAX_FACES", "5"))
FACE_DETECTION_CONFIDENCE: float = float(os.getenv("FACE_DETECTION_CONFIDENCE", "0.5"))
FACE_TRACKING_CONFIDENCE: float = float(os.getenv("FACE_TRACKING_CONFIDENCE", "0.5"))

# Emotion class labels ordered to match the model's output vector
EMOTION_LABELS: list[str] = [
    "anger",
    "disgust",
    "fear",
    "happiness",
    "neutral",
    "sadness",
    "surprise",
]

# ---------------------------------------------------------------------------
# Performance thresholds
# ---------------------------------------------------------------------------
IMAGE_ANALYSIS_TIMEOUT_S: float = float(os.getenv("IMAGE_ANALYSIS_TIMEOUT_S", "2.0"))
WEBCAM_FRAME_BUDGET_MS: float = float(os.getenv("WEBCAM_FRAME_BUDGET_MS", "250.0"))
WEBCAM_TARGET_FPS: float = float(os.getenv("WEBCAM_TARGET_FPS", "10.0"))

# ---------------------------------------------------------------------------
# Video sampling
# ---------------------------------------------------------------------------
VIDEO_SAMPLE_FPS: float = float(os.getenv("VIDEO_SAMPLE_FPS", "2.0"))
VIDEO_MAX_DURATION_S: float = float(os.getenv("VIDEO_MAX_DURATION_S", "600.0"))

# ---------------------------------------------------------------------------
# Input image size constraints
# ---------------------------------------------------------------------------
IMAGE_MIN_SIDE: int = int(os.getenv("IMAGE_MIN_SIDE", "224"))
IMAGE_PROCESSING_CEILING: int = int(os.getenv("IMAGE_PROCESSING_CEILING", "4096"))
IMAGE_REJECTION_CEILING: int = int(os.getenv("IMAGE_REJECTION_CEILING", "8192"))

# ---------------------------------------------------------------------------
# Local-only processing enforcement
# No cloud calls are made.  Analysis always happens on the local device.
# ---------------------------------------------------------------------------
LOCAL_ONLY: bool = True
