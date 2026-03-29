"""Image analysis pipeline — US1 and US2."""

from __future__ import annotations

import uuid
from typing import Optional

import numpy as np

from facial_emotions.contracts.results import ImageAnalysisResult
from facial_emotions.pipelines.base_pipeline import build_face_result
from facial_emotions.services.face_detection import FaceDetector
from facial_emotions.services.emotion_inference import EmotionInferenceService
from facial_emotions.services.genuineness_assessment import GenuinenessAssessmentService
from facial_emotions.services.landmark_features import LandmarkExtractor
from facial_emotions.services.overlay_renderer import draw_results


class ImageAnalysisPipeline:
    """Orchestrates the full single-image analysis flow."""

    def __init__(
        self,
        face_detector: Optional[FaceDetector] = None,
        emotion_service: Optional[EmotionInferenceService] = None,
        landmark_extractor: Optional[LandmarkExtractor] = None,
        genuineness_service: Optional[GenuinenessAssessmentService] = None,
        render_overlay: bool = True,
    ) -> None:
        self._detector = face_detector or FaceDetector()
        self._emotion = emotion_service or EmotionInferenceService()
        self._landmarks = landmark_extractor or LandmarkExtractor()
        self._genuineness = genuineness_service or GenuinenessAssessmentService()
        self._render_overlay = render_overlay

    def analyse(
        self,
        bgr_image: np.ndarray,
        request_id: Optional[str] = None,
        enable_genuineness: bool = True,
    ) -> ImageAnalysisResult:
        h, w = bgr_image.shape[:2]
        req_id = request_id or str(uuid.uuid4())

        # 1. Detect faces
        face_detections = self._detector.detect(bgr_image)
        if not face_detections:
            return ImageAnalysisResult(
                request_id=req_id,
                image_width=w,
                image_height=h,
                faces=[],
                annotated_image=bgr_image.copy() if self._render_overlay else None,
            )

        # 2. Predict emotions
        emotions = self._emotion.predict(bgr_image, face_detections)
        emotion_map = {e.face_id: e for e in emotions}

        # 3. Extract landmarks (best-effort; skip if extractor fails)
        try:
            landmarks = self._landmarks.extract(bgr_image, face_detections)
        except Exception:
            landmarks = []
        landmark_map = {lm.face_id: lm for lm in landmarks}

        # 4. Genuineness assessment (only when landmarks available)
        if enable_genuineness and landmarks:
            genuineness_list = self._genuineness.assess(landmarks, emotions)
        else:
            genuineness_list = []
        genuine_map = {g.face_id: g for g in genuineness_list}

        # 5. Build contract results
        face_results = []
        for face in face_detections:
            emo = emotion_map.get(face.face_id)
            if emo is None:
                continue
            face_results.append(
                build_face_result(face, emo, genuine_map.get(face.face_id))
            )

        # 6. Render overlay
        annotated = None
        if self._render_overlay:
            annotated = draw_results(
                bgr_image, face_detections, emotions, genuineness_list
            )

        return ImageAnalysisResult(
            request_id=req_id,
            image_width=w,
            image_height=h,
            faces=face_results,
            annotated_image=annotated,
        )
