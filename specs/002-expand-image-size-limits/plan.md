# Implementation Plan: Expand Image Size Limits

**Branch**: `002-expand-image-size-limits` | **Date**: 2026-03-29 | **Spec**: [spec.md](spec.md)  
**Input**: Feature specification from `specs/002-expand-image-size-limits/spec.md`

## Summary

Replace the hard 1920×1080 pixel cap with a two-tier size policy: images up to 8192×8192 px are accepted; images whose longest side exceeds 4096 px are automatically downscaled to a 4096 px processing frame before face detection and inference, then bounding boxes are remapped back to original image coordinates before results are returned. The annotated output image is rendered at full original resolution. An info notice is shown in the Gradio UI when auto-downscaling occurs.

## Technical Context

**Language/Version**: Python 3.13.5  
**Primary Dependencies**: OpenCV (`cv2`) for image I/O and resizing; ONNX Runtime for inference; Gradio for UI; MediaPipe for face detection; standard `logging` module  
**Storage**: N/A — stateless single-request processing; no image persistence  
**Testing**: pytest with numpy test images (no disk fixtures needed for size-limit tests)  
**Target Platform**: Local desktop application (Windows/macOS/Linux)  
**Project Type**: Desktop Gradio application  
**Performance Goals**: Full analysis of a 4032×3024 image ≤ 2 seconds (SC-001); processing ceiling downscale adds ≤ 50 ms overhead  
**Constraints**: Peak RAM per request ≤ ~250 MB (original + processing frame simultaneously); no external network calls; offline-only  
**Scale/Scope**: Single-user local tool; one image at a time

## Constitution Check

*Re-checked after Phase 1 design — all gates pass.*

| Principle | Status | Notes |
|-----------|--------|-------|
| I. Genuine Emotion Analysis | ✅ Pass | Feature only changes size handling; emotion + genuineness logic untouched |
| II. Dataset Utilization | ✅ Pass (N/A) | No dataset changes |
| III. Pre-Trained Models | ✅ Pass | FER+ ONNX model unchanged; it receives 64×64 crops regardless of input size |
| IV. Rule-Based / ML Classifiers | ✅ Pass (N/A) | Genuineness classifier unchanged |
| V. TDD | ✅ Pass | Tests written/updated before implementation per tasks.md |
| Ethical Considerations | ✅ Pass | No new data collection; no bias impact |
| Performance Standards | ✅ Pass | Auto-downscale to 4096 preserves ≤2 s budget |
| Quality Gates | ✅ Pass | All 168 existing tests must pass (SC-005); new tests added |

**No violations. No Complexity Tracking entries required.**

## Project Structure

### Documentation (this feature)

```text
specs/002-expand-image-size-limits/
├── plan.md              ← this file
├── research.md          ← Phase 0 complete
├── data-model.md        ← Phase 1 complete
├── contracts/
│   └── image-size-contract.md   ← Phase 1 complete
└── tasks.md             ← Phase 2 (created by /speckit.tasks)
```

### Source Code (affected files only)

```text
src/facial_emotions/
├── config.py                          MODIFY — replace IMAGE_MAX_WIDTH/HEIGHT with
│                                               IMAGE_PROCESSING_CEILING + IMAGE_REJECTION_CEILING
├── services/
│   ├── media_io.py                    MODIFY — update validate_image_dimensions(),
│                                               add downscale_to_processing_ceiling()
│   └── image_analysis_service.py     MODIFY — add downscale step + pass original_image to pipeline
├── pipelines/
│   └── image_pipeline.py             MODIFY — add optional original_image param,
│                                               add _remap_detections() helper
└── ui/
    └── gradio_app.py                  MODIFY — add info notice when downscaling occurs

.env.example                           MODIFY — document new env vars

tests/
├── unit/
│   ├── test_config.py                 MODIFY — add env var override tests for new constants
│   └── test_media_io_downscale.py     NEW    — unit tests for downscale_to_processing_ceiling
├── integration/
│   ├── test_image_input_boundaries.py MODIFY — update 2 tests that expected 1921×1080 rejection;
│   │                                           add tests for new rejection ceiling (8193 px)
│   └── test_coordinate_remapping.py   NEW    — integration tests for bbox remapping accuracy
└── (all other existing tests)         UNCHANGED
```

**Structure Decision**: Single-project layout (existing). No new packages, modules outside the above list, or new top-level directories are needed.

---

## Implementation Design

### A. `config.py` changes

Remove:
```python
IMAGE_MAX_WIDTH: int = 1920
IMAGE_MAX_HEIGHT: int = 1080
```

Add:
```python
IMAGE_PROCESSING_CEILING: int = int(os.getenv("IMAGE_PROCESSING_CEILING", "4096"))
IMAGE_REJECTION_CEILING: int = int(os.getenv("IMAGE_REJECTION_CEILING", "8192"))
```

### B. `media_io.py` changes

**`validate_image_dimensions` update** — replace dual width/height check:
```python
def validate_image_dimensions(img: np.ndarray) -> np.ndarray:
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
```

**New `downscale_to_processing_ceiling`**:
```python
import logging
_logger = logging.getLogger(__name__)

def downscale_to_processing_ceiling(img: np.ndarray) -> tuple[np.ndarray, float]:
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
```

### C. `image_analysis_service.py` changes

```python
from facial_emotions.services.media_io import (
    load_image,
    validate_image_dimensions,
    downscale_to_processing_ceiling,   # NEW
    get_image_dimensions,
)

def analyse_path(self, image_path: str) -> ImageAnalysisResult:
    img = load_image(image_path)
    validate_image_dimensions(img)
    frame, scale_factor = downscale_to_processing_ceiling(img)   # NEW
    original = img if scale_factor < 1.0 else None               # NEW
    return self._pipeline.analyse(frame, original_image=original)

def analyse_array(self, bgr_image, request_id=None) -> ImageAnalysisResult:
    validate_image_dimensions(bgr_image)
    frame, scale_factor = downscale_to_processing_ceiling(bgr_image)  # NEW
    original = bgr_image if scale_factor < 1.0 else None              # NEW
    return self._pipeline.analyse(frame, request_id=request_id, original_image=original)
```

### D. `image_pipeline.py` changes

```python
import dataclasses
from facial_emotions.domain.models import FaceDetection

def analyse(
    self,
    bgr_image: np.ndarray,
    request_id: Optional[str] = None,
    enable_genuineness: bool = True,
    original_image: Optional[np.ndarray] = None,   # NEW
) -> ImageAnalysisResult:
    ...
    # After face_detections obtained, before building results:
    annotation_img = bgr_image
    if original_image is not None:
        inv_scale = original_image.shape[1] / bgr_image.shape[1]
        face_detections = self._remap_detections(face_detections, inv_scale)
        annotation_img = original_image

    # render overlay on annotation_img (not bgr_image)
    annotated = draw_results(annotation_img, face_detections, emotions, genuineness_list)

    # ImageAnalysisResult uses annotation_img dimensions
    h, w = annotation_img.shape[:2]
    return ImageAnalysisResult(request_id=req_id, image_width=w, image_height=h, ...)

@staticmethod
def _remap_detections(
    faces: list[FaceDetection], inv_scale: float
) -> list[FaceDetection]:
    return [
        dataclasses.replace(
            f,
            bbox_x=round(f.bbox_x * inv_scale),
            bbox_y=round(f.bbox_y * inv_scale),
            bbox_width=round(f.bbox_width * inv_scale),
            bbox_height=round(f.bbox_height * inv_scale),
        )
        for f in faces
    ]
```

### E. `gradio_app.py` changes

```python
def analyse_image(pil_image):
    ...
    bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    try:
        validate_image_dimensions(bgr)
    except MediaIOError as e:
        return pil_image, f"Input validation error: {e}"

    from facial_emotions.services.media_io import downscale_to_processing_ceiling  # NEW
    frame, scale_factor = downscale_to_processing_ceiling(bgr)                     # NEW
    orig_h, orig_w = bgr.shape[:2]
    frame_h, frame_w = frame.shape[:2]

    original_for_pipeline = bgr if scale_factor < 1.0 else None
    result = pipeline.analyse(frame, original_image=original_for_pipeline)

    # Build summary + optional notice
    notice = ""
    if scale_factor < 1.0:
        notice = (
            f"> ℹ️ Large image auto-resized for processing "
            f"({orig_w}×{orig_h} → {frame_w}×{frame_h}).\n\n"
        )
    ...
    return annotated_rgb or pil_image, notice + "\n\n".join(summary_lines)
```

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| `dataclasses.replace` fails if `FaceDetection` is not a pure dataclass | Low | High | `FaceDetection` confirmed as `@dataclass` in `domain/models.py`; verified in research |
| `cv2.INTER_AREA` at extreme ratios degrades face detection accuracy | Low | Medium | Processing ceiling at 4096 px; faces in 8K images are still ≥ 200 px tall at 4096 reduction |
| Existing tests break due to removed `IMAGE_MAX_WIDTH` / `IMAGE_MAX_HEIGHT` | Medium | Low | 2 tests identified; update them before implementation starts (TDD) |
| Memory spike for simultaneous original + frame in analyse_path | Low | Low | At 8192×8192 peak: 251 MB total — within typical machine RAM |

---

## Constitution Check (Post-Design Re-evaluation)

All five principles still pass — confirmed. No new abstractions, services, or data stores introduced. The change is confined to configuration constants, one new utility function in `media_io.py`, one new optional parameter in `image_pipeline.py`, and UI notice text in `gradio_app.py`.
