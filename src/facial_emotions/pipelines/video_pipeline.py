"""Video analysis pipeline — decodes video and processes sampled frames."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from facial_emotions import config
from facial_emotions.contracts.results import FrameResult, VideoAnalysisResult
from facial_emotions.pipelines.base_pipeline import build_frame_result
from facial_emotions.domain.models import FrameAnalysis


class VideoAnalysisPipeline:
    """Decode a video, sample frames, and return a VideoAnalysisResult."""

    def __init__(
        self,
        face_detector=None,
        emotion_service=None,
        landmark_extractor=None,
        genuineness_service=None,
        overlay_renderer=None,
    ) -> None:
        # Lazy-init real services only when not injected (keeps tests fast)
        self._face_detector = face_detector
        self._emotion_service = emotion_service
        self._landmark_extractor = landmark_extractor
        self._genuineness_service = genuineness_service
        self._overlay_renderer = overlay_renderer

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyse(
        self,
        video_path: str,
        request_id: Optional[str] = None,
        sample_fps: float = config.VIDEO_SAMPLE_FPS,
    ) -> VideoAnalysisResult:
        req_id = request_id or str(uuid.uuid4())
        sampled = self._read_sampled_frames(video_path, sample_fps)

        duration_s = self._get_duration_seconds(video_path)
        frame_results: list[FrameResult] = []

        for frame_idx, (timestamp_ms, bgr_frame) in enumerate(sampled):
            frame_result = self._process_frame(bgr_frame, frame_idx, timestamp_ms)
            frame_results.append(frame_result)

        return VideoAnalysisResult(
            request_id=req_id,
            video_path=video_path,
            duration_seconds=duration_s,
            sampled_frames=frame_results,
        )

    # ------------------------------------------------------------------
    # Internal helpers — separated for easy patching in tests
    # ------------------------------------------------------------------

    def _read_sampled_frames(
        self,
        video_path: str,
        sample_fps: float = config.VIDEO_SAMPLE_FPS,
    ) -> list[tuple[int, np.ndarray]]:
        """Return list of (timestamp_ms, bgr_frame) for sampled frames."""
        from facial_emotions.services.video_analysis_service import sample_frame_timestamps

        cap = cv2.VideoCapture(video_path)
        try:
            video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            timestamps = sample_frame_timestamps(total_frames, video_fps, sample_fps)

            step = max(1, round(video_fps / sample_fps))
            frames: list[tuple[int, np.ndarray]] = []
            frame_idx = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx in range(0, total_frames, step):
                    ts_ms = round(frame_idx / video_fps * 1000)
                    frames.append((ts_ms, frame))
                frame_idx += 1

            return frames
        finally:
            cap.release()

    def _get_duration_seconds(self, video_path: str) -> float:
        cap = cv2.VideoCapture(video_path)
        try:
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            return total / fps if fps > 0 else 0.0
        finally:
            cap.release()

    def _process_frame(
        self,
        bgr_frame: np.ndarray,
        frame_idx: int,
        timestamp_ms: int,
    ) -> FrameResult:
        """Run detection + emotion + genuineness on a single frame."""
        face_detector = self._get_face_detector()
        emotion_service = self._get_emotion_service()
        landmark_extractor = self._get_landmark_extractor()
        genuineness_service = self._get_genuineness_service()

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
            frame_id=f"frame-{frame_idx:04d}",
            timestamp_ms=timestamp_ms,
            faces=face_detections,
            landmarks=landmarks,
            emotions=emotions,
            genuineness=genuineness_results,
        )
        return build_frame_result(frame_domain)

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
