"""Media input validation and path handling utilities.

All analysis happens locally — no files are uploaded to any external server.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

from facial_emotions import config

_logger = logging.getLogger(__name__)

SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
SUPPORTED_VIDEO_EXTS = {".mp4", ".avi", ".mov"}


class MediaIOError(ValueError):
    """Raised when media cannot be read or does not meet constraints."""


def validate_image_path(path: str) -> Path:
    p = Path(path)
    if not p.exists():
        raise MediaIOError(f"File not found: {path}")
    if p.suffix.lower() not in SUPPORTED_IMAGE_EXTS:
        raise MediaIOError(
            f"Unsupported image format '{p.suffix}'. Supported: {SUPPORTED_IMAGE_EXTS}"
        )
    return p


def validate_video_path(path: str) -> Path:
    p = Path(path)
    if not p.exists():
        raise MediaIOError(f"File not found: {path}")
    if p.suffix.lower() not in SUPPORTED_VIDEO_EXTS:
        raise MediaIOError(
            f"Unsupported video format '{p.suffix}'. Supported: {SUPPORTED_VIDEO_EXTS}"
        )
    return p


def load_image(path: str) -> np.ndarray:
    """Load an image from disk and return a BGR numpy array."""
    validate_image_path(path)
    img = cv2.imread(str(path))
    if img is None:
        raise MediaIOError(f"OpenCV could not decode image: {path}")
    return img


def get_image_dimensions(img: np.ndarray) -> Tuple[int, int]:
    """Return (width, height) from a BGR image array."""
    h, w = img.shape[:2]
    return w, h


def validate_image_dimensions(img: np.ndarray) -> np.ndarray:
    """Raise if image is outside the supported size range; otherwise return it unchanged.

    Supported range: IMAGE_MIN_SIDE px minimum on the shorter side,
    up to IMAGE_REJECTION_CEILING px on the longest side.
    """
    h, w = img.shape[:2]
    min_side = min(w, h)
    if min_side < config.IMAGE_MIN_SIDE:
        raise MediaIOError(
            f"Image is too small ({w}x{h}). Minimum side must be ≥ {config.IMAGE_MIN_SIDE}px."
        )
    longest = max(w, h)
    if longest > config.IMAGE_REJECTION_CEILING:
        raise MediaIOError(
            f"Image is too large ({w}x{h}). "
            f"Maximum accepted longest side is {config.IMAGE_REJECTION_CEILING}px."
        )
    return img


def downscale_to_processing_ceiling(img: np.ndarray) -> tuple[np.ndarray, float]:
    """Proportionally downscale an image if its longest side exceeds the processing ceiling.

    Returns (frame, scale_factor) where scale_factor is in (0, 1] and equals 1.0
    when no downscaling was performed (the original array is returned unchanged).
    """
    h, w = img.shape[:2]
    longest = max(w, h)
    if longest <= config.IMAGE_PROCESSING_CEILING:
        return img, 1.0
    scale = config.IMAGE_PROCESSING_CEILING / longest
    new_w = round(w * scale)
    new_h = round(h * scale)
    frame = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    _logger.debug(
        "Image downscaled for processing: original=%dx%d → frame=%dx%d (scale=%.4f)",
        w, h, new_w, new_h, scale,
    )
    return frame, scale


def resize_for_model(img: np.ndarray, target_size: int = 224) -> np.ndarray:
    """Resize image to target_size × target_size with aspect-ratio-preserving letterbox."""
    h, w = img.shape[:2]
    scale = target_size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((target_size, target_size, img.shape[2]), dtype=img.dtype)
    pad_top = (target_size - new_h) // 2
    pad_left = (target_size - new_w) // 2
    canvas[pad_top : pad_top + new_h, pad_left : pad_left + new_w] = resized
    return canvas


def open_video_capture(path: str) -> cv2.VideoCapture:
    validate_video_path(path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise MediaIOError(f"OpenCV could not open video: {path}")
    return cap
