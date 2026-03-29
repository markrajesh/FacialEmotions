"""Webcam analysis pipeline — processes a single frame and returns results."""

from __future__ import annotations

import time
import uuid
from typing import Optional

import numpy as np

from facial_emotions import config
from facial_emotions.contracts.results import WebcamFrameResult
from facial_emotions.domain.models import FrameAnalysis
from facial_emotions.pipelines.base_pipeline import build_frame_result


class WebcamAnalysisPipeline:
    """Process individual webcam frames and return WebcamFrameResult objects."""

    def __init__(
        self,
        face_detector=None,
        emotion_service=None,
        landmark_extractor=None,
        genuineness_service=None,
        overlay_renderer=None,
    ) -> None:
        self._face_detector = face_detector
        self._emotion_service = emotion_service
        self._landmark_extractor = landmark_extractor
        self._genuineness_service = genuineness_service
        self._overlay_renderer = overlay_renderer
        self._frame_index: int = 0
        self._start_time_ms: int = int(time.monotonic() * 1000)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyse_frame(
        self,
        bgr_frame: np.ndarray,
        request_id: Optional[str] = None,
    ) -> WebcamFrameResult:
        req_id = request_id or str(uuid.uuid4())
        timestamp_ms = int(time.monotonic() * 1000) - self._start_time_ms
        frame_idx = self._frame_index
        self._frame_index += 1

        face_detector = self._get_face_detector()
        emotion_service = self._get_emotion_service()
        landmark_extractor = self._get_landmark_extractor()
        genuineness_service = self._get_genuineness_service()
        overlay_renderer = self._get_overlay_renderer()

        face_detections = face_detector.detect(bgr_frame)
        emotions = []
        landmarks = []
        genuineness_results = []

        for face in face_detections:
            emotion = emotion_service.predict_single_face(bgr_frame, face)
            emotions.append(emotion)

            lm = landmark_extractor.extract(bgr_frame, face)
            if lm:
                landmarks.append(lm)
            gen = genuineness_service.assess(lm, emotion)
            if gen:
                genuineness_results.append(gen)

        frame_domain = FrameAnalysis(
            frame_id=f"webcam-{frame_idx:06d}",
            timestamp_ms=timestamp_ms,
            faces=face_detections,
            landmarks=landmarks,
            emotions=emotions,
            genuineness=genuineness_results,
        )
        frame_result = build_frame_result(frame_domain)

        annotated = overlay_renderer(bgr_frame, face_detections)

        return WebcamFrameResult(
            request_id=req_id,
            frame_index=frame_idx,
            timestamp_ms=timestamp_ms,
            faces=frame_result.faces,
            annotated_frame=annotated,
        )

    # ------------------------------------------------------------------
    # Lazy service accessors
    # ------------------------------------------------------------------

    def _get_face_detector(self):
        if self._face_detector is None:
            from facial_emotions.services.face_detection import FaceDetector
            self._face_detector = FaceDetector()
        return self._face_detector

    def _get_emotion_service(self):
        if self._emotion_service is None:
            from facial_emotions.services.emotion_inference import EmotionInferenceService
            self._emotion_service = EmotionInferenceService()
        return self._emotion_service

    def _get_landmark_extractor(self):
        if self._landmark_extractor is None:
            from facial_emotions.services.landmark_features import LandmarkExtractor
            self._landmark_extractor = LandmarkExtractor()
        return self._landmark_extractor

    def _get_genuineness_service(self):
        if self._genuineness_service is None:
            from facial_emotions.services.genuineness_assessment import GenuinenessAssessmentService
            self._genuineness_service = GenuinenessAssessmentService()
        return self._genuineness_service

    def _get_overlay_renderer(self):
        if self._overlay_renderer is None:
            from facial_emotions.services.overlay_renderer import draw_results
            self._overlay_renderer = draw_results
        return self._overlay_renderer
