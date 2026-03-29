"""Pre-trained emotion inference wrapper using ONNX Runtime.

All inference runs locally (CPU) — no network calls are made.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from facial_emotions import config
from facial_emotions.domain.enums import EmotionLabel
from facial_emotions.domain.models import EmotionPrediction, FaceDetection


class EmotionInferenceError(RuntimeError):
    """Raised when the emotion model cannot be loaded or produces invalid output."""


class EmotionInferenceService:
    """Wraps an ONNX emotion classification model.

    The model is expected to:
    - accept a single 1×1×48×48 or 1×3×224×224 float32 tensor (auto-detected)
    - output a 1×7 softmax probability vector matching EMOTION_LABELS order
    """

    def __init__(self, model_path: Optional[Path] = None) -> None:
        self._model_path = model_path or config.EMOTION_MODEL_PATH
        self._session = None  # lazy-loaded on first predict

    def _ensure_session(self) -> None:
        if self._session is not None:
            return
        try:
            import onnxruntime as ort  # local import for testability with mocking
        except ImportError as exc:
            raise EmotionInferenceError("onnxruntime is not installed") from exc

        mp = self._model_path
        if not Path(mp).exists():
            raise EmotionInferenceError(
                f"Emotion model not found at {mp}. "
                "Please download an ONNX emotion model and place it at that path."
            )
        self._session = ort.InferenceSession(str(mp), providers=["CPUExecutionProvider"])
        input_meta = self._session.get_inputs()[0]
        self._input_name: str = input_meta.name
        self._input_shape: List[int] = input_meta.shape  # e.g. [1, 1, 48, 48]

    def _preprocess(self, face_crop: np.ndarray) -> np.ndarray:
        """Resize and normalise a face crop to the model's expected input shape."""
        shape = self._input_shape  # [batch, channels, h, w]
        target_h, target_w = shape[-2], shape[-1]
        channels = shape[1] if len(shape) == 4 else 1

        resized = cv2.resize(face_crop, (target_w, target_h), interpolation=cv2.INTER_AREA)
        if channels == 1:
            if len(resized.shape) == 3:
                resized = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            tensor = resized.astype(np.float32) / 255.0
            tensor = tensor[np.newaxis, np.newaxis, :, :]  # 1×1×H×W
        else:
            if len(resized.shape) == 2:
                resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
            tensor = resized.astype(np.float32) / 255.0
            tensor = tensor.transpose(2, 0, 1)[np.newaxis, :, :, :]  # 1×C×H×W
        return tensor

    def predict_single_face(
        self,
        bgr_image: np.ndarray,
        face: FaceDetection,
    ) -> EmotionPrediction:
        """Run emotion inference on one cropped face region."""
        self._ensure_session()

        x, y, bw, bh = face.bbox_x, face.bbox_y, face.bbox_width, face.bbox_height
        crop = bgr_image[y : y + bh, x : x + bw]
        if crop.size == 0:
            return EmotionPrediction(
                face_id=face.face_id,
                emotion_label=EmotionLabel.NEUTRAL,
                confidence=0.0,
                top_k_scores={e.value: 0.0 for e in EmotionLabel},
            )

        tensor = self._preprocess(crop)
        outputs = self._session.run(None, {self._input_name: tensor})
        scores: np.ndarray = outputs[0].flatten()

        # Softmax if not already applied by the model
        exp_s = np.exp(scores - scores.max())
        probs = exp_s / exp_s.sum()

        top_idx = int(np.argmax(probs))
        labels = config.EMOTION_LABELS
        top_label = EmotionLabel(labels[top_idx])
        top_k = {label: float(probs[i]) for i, label in enumerate(labels)}

        return EmotionPrediction(
            face_id=face.face_id,
            emotion_label=top_label,
            confidence=round(float(probs[top_idx]), 4),
            top_k_scores=top_k,
        )

    def predict(
        self,
        bgr_image: np.ndarray,
        faces: List[FaceDetection],
    ) -> List[EmotionPrediction]:
        return [self.predict_single_face(bgr_image, f) for f in faces]
