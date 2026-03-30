"""Image-specific request assembly and response mapping."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Optional

import numpy as np

from facial_emotions.contracts.results import ImageAnalysisResult
from facial_emotions.domain.enums import InputMode, MediaFormat, MediaType
from facial_emotions.domain.models import AnalysisRequest, InputAsset
from facial_emotions.pipelines.image_pipeline import ImageAnalysisPipeline
from facial_emotions.services.media_io import (
    downscale_to_processing_ceiling,
    load_image,
    validate_image_dimensions,
    get_image_dimensions,
)


class ImageAnalysisService:
    """Ties request validation, loading, and pipeline execution together."""

    def __init__(self, pipeline: Optional[ImageAnalysisPipeline] = None) -> None:
        self._pipeline = pipeline or ImageAnalysisPipeline()

    def analyse_path(self, image_path: str) -> ImageAnalysisResult:
        """Load an image from disk, validate it, and run the analysis pipeline."""
        img = load_image(image_path)
        validate_image_dimensions(img)
        frame, scale_factor = downscale_to_processing_ceiling(img)
        request_id = str(uuid.uuid4())
        return self._pipeline.analyse(
            frame,
            request_id=request_id,
            original_image=img if scale_factor < 1.0 else None,
        )

    def analyse_array(
        self,
        bgr_image: np.ndarray,
        request_id: Optional[str] = None,
    ) -> ImageAnalysisResult:
        """Run the analysis pipeline on an already-loaded BGR image array."""
        validate_image_dimensions(bgr_image)
        frame, scale_factor = downscale_to_processing_ceiling(bgr_image)
        return self._pipeline.analyse(
            frame,
            request_id=request_id,
            original_image=bgr_image if scale_factor < 1.0 else None,
        )
