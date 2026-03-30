"""Image analysis pipeline — US1 and US2."""

from __future__ import annotations

import dataclasses
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
        original_image: Optional[np.ndarray] = None,
    ) -> ImageAnalysisResult:
        req_id = request_id or str(uuid.uuid4())

        # Determine which image will carry the final annotation / dimensions
        annotation_img = bgr_image if original_image is None else original_image
        out_h, out_w = annotation_img.shape[:2]

        # 1. Detect faces (always on the processing frame)
        face_detections = self._detector.detect(bgr_image)
        if not face_detections:
            return ImageAnalysisResult(
                request_id=req_id,
                image_width=out_w,
                image_height=out_h,
                faces=[],
                annotated_image=annotation_img.copy() if self._render_overlay else None,
            )

        # 2. Predict emotions (processing frame + processing-frame coords)
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

        # 4b. Remap bbox coordinates to original image space (if downscaled)
        if original_image is not None:
            inv_scale_x = original_image.shape[1] / bgr_image.shape[1]
            inv_scale_y = original_image.shape[0] / bgr_image.shape[0]
            face_detections = self._remap_detections(
                face_detections, inv_scale_x, inv_scale_y
            )

        # 5. Build contract results
        face_results = []
        for face in face_detections:
            emo = emotion_map.get(face.face_id)
            if emo is None:
                continue
            face_results.append(
                build_face_result(face, emo, genuine_map.get(face.face_id))
            )

        # 6. Render overlay on the annotation image (original dimensions)
        annotated = None
        if self._render_overlay:
            annotated = draw_results(
                annotation_img, face_detections, emotions, genuineness_list
            )

        return ImageAnalysisResult(
            request_id=req_id,
            image_width=out_w,
            image_height=out_h,
            faces=face_results,
            annotated_image=annotated,
        )

    @staticmethod
    def _remap_detections(
        faces: list,
        inv_scale_x: float,
        inv_scale_y: float,
    ) -> list:
        """Return new FaceDetection objects with bbox coords mapped to original image space."""
        from facial_emotions.domain.models import FaceDetection  # avoid circular at module level
        return [
            dataclasses.replace(
                f,
                bbox_x=round(f.bbox_x * inv_scale_x),
                bbox_y=round(f.bbox_y * inv_scale_y),
                bbox_width=round(f.bbox_width * inv_scale_x),
                bbox_height=round(f.bbox_height * inv_scale_y),
            )
            for f in faces
        ]
