"""Domain enumerations."""

from __future__ import annotations

from enum import Enum


class InputMode(str, Enum):
    IMAGE = "image"
    VIDEO = "video"
    WEBCAM = "webcam"


class MediaType(str, Enum):
    IMAGE = "image"
    VIDEO = "video"
    WEBCAM_FRAME = "webcam_frame"


class MediaFormat(str, Enum):
    JPEG = "jpeg"
    PNG = "png"
    BMP = "bmp"
    MP4 = "mp4"
    AVI = "avi"
    MOV = "mov"
    LIVE = "live"


class EmotionLabel(str, Enum):
    ANGER = "anger"
    DISGUST = "disgust"
    FEAR = "fear"
    HAPPINESS = "happiness"
    NEUTRAL = "neutral"
    SADNESS = "sadness"
    SURPRISE = "surprise"


class GenuinenessState(str, Enum):
    GENUINE = "genuine"
    POSED = "posed"
    UNCERTAIN = "uncertain"
