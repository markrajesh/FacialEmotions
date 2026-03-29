# Data Model: Expand Image Size Limits

**Feature**: 002-expand-image-size-limits  
**Date**: 2026-03-29

---

## Changed Entities

### ImageSizeConstraints (config constants — replaces old IMAGE_MAX_WIDTH / IMAGE_MAX_HEIGHT)

These are module-level constants in `src/facial_emotions/config.py`, loaded from environment variables at startup.

| Constant | Type | Default | Env Var | Description |
|----------|------|---------|---------|-------------|
| `IMAGE_MIN_SIDE` | `int` | `224` | — (unchanged, not env-configurable) | Minimum pixel length of the shorter image side |
| `IMAGE_PROCESSING_CEILING` | `int` | `4096` | `IMAGE_PROCESSING_CEILING` | Longest-side threshold above which the image is proportionally downscaled before pipeline processing |
| `IMAGE_REJECTION_CEILING` | `int` | `8192` | `IMAGE_REJECTION_CEILING` | Longest-side threshold above which the image is rejected with an error |

**Removed**:
- `IMAGE_MAX_WIDTH: int = 1920`
- `IMAGE_MAX_HEIGHT: int = 1080`

**Validation rules**:
- `IMAGE_MIN_SIDE < IMAGE_PROCESSING_CEILING < IMAGE_REJECTION_CEILING` (all positive integers)
- Both new constants must be > 0 — no further validation is added at this stage (startup misconfiguration is caught by tests)

---

### ProcessingFrame (new concept — not a class, a named data flow stage)

A transient intermediate representation created when a submitted image exceeds the processing ceiling.

| Attribute | Type | Description |
|-----------|------|-------------|
| `frame` | `np.ndarray` (H×W×3 uint8) | The proportionally downscaled BGR image used as input to the face-detection and inference pipeline |
| `scale_factor` | `float` (0.0 < x ≤ 1.0) | Ratio of processing frame longest side to original longest side. `1.0` when no downscaling was performed. |
| `original_w` | `int` | Width of the original submitted image (pixels) |
| `original_h` | `int` | Height of the original submitted image (pixels) |

**Note**: This is not persisted or returned in the API contract. It lives only during a single `analyse_array()` / `analyse_path()` call.

**State transitions**:
```
original image (w×h)
    │
    ├─ max(w,h) > REJECTION_CEILING  → MediaIOError("too large")
    ├─ min(w,h) < IMAGE_MIN_SIDE     → MediaIOError("too small")
    └─ validated OK
            │
            ├─ max(w,h) > PROCESSING_CEILING → downscale → (frame, scale < 1.0)
            └─ max(w,h) ≤ PROCESSING_CEILING → pass-through → (original, scale = 1.0)
                    │
                    └─ pipeline.analyse(frame, original_image=original or None)
                            │
                            └─ ImageAnalysisResult (bbox in original coordinates)
```

---

### FaceDetection (unchanged schema, new remapping step)

`src/facial_emotions/domain/models.py` — `FaceDetection` dataclass is unchanged.

| Field | Type | Description |
|-------|------|-------------|
| `face_id` | `str` | Stable identifier within the request |
| `bbox_x` | `int` | X coordinate of top-left corner **in original image pixels** after remapping |
| `bbox_y` | `int` | Y coordinate of top-left corner **in original image pixels** after remapping |
| `bbox_width` | `int` | Bounding box width **in original image pixels** after remapping |
| `bbox_height` | `int` | Bounding box height **in original image pixels** after remapping |
| `detection_confidence` | `float` | 0.0–1.0 |

**New invariant** (post-remapping): all four bbox integer fields are expressed in original image coordinates regardless of whether a processing-ceiling downscale occurred.

**Remapping formula** (applied inside `ImageAnalysisPipeline.analyse()` when `original_image` is provided):
```
bbox_x_orig     = round(bbox_x_frame     / scale_factor)
bbox_y_orig     = round(bbox_y_frame     / scale_factor)
bbox_width_orig = round(bbox_width_frame / scale_factor)
bbox_height_orig= round(bbox_height_frame/ scale_factor)
```
Maximum rounding error per coordinate: ≤ 0.5 / scale_factor ≤ 2.0 px at scale_factor = 0.25 (8× downscale). This satisfies SC-003 (±2 px).

---

### ImageAnalysisResult (unchanged schema)

`src/facial_emotions/contracts/results.py` — no field additions. The `image_width` and `image_height` fields already carry the dimensions of the image that was analysed; after this feature they will correctly reflect **original** image dimensions when a downscale occurred.

| Field | Type | Note |
|-------|------|------|
| `request_id` | `str` | Unchanged |
| `image_width` | `int` | Now always reflects original image width |
| `image_height` | `int` | Now always reflects original image height |
| `faces` | `list[FaceResult]` | `bbox` coordinates now always in original space |
| `annotated_image` | `np.ndarray \| None` | Now rendered on original image (full resolution) |

---

## New Functions (not classes)

### `media_io.downscale_to_processing_ceiling`

```
Signature: (img: np.ndarray) -> tuple[np.ndarray, float]

Input:  A validated BGR image array that has passed validate_image_dimensions().
Output: (frame, scale_factor)
        - frame: proportionally downscaled image if longest side > IMAGE_PROCESSING_CEILING,
                 otherwise the original array (no copy).
        - scale_factor: float in (0, 1] — 1.0 means no downscale.

Side effects: Emits a DEBUG log entry when downscaling occurs.
Errors: None (image is guaranteed valid by the time this is called).
```

### `image_pipeline._remap_detections` (private helper)

```
Signature: (faces: list[FaceDetection], inv_scale: float) -> list[FaceDetection]

Input:  face detections in processing-frame coordinates + inverse scale factor (1/scale_factor).
Output: new list of FaceDetection with bbox fields scaled to original image coordinates.
        Uses dataclasses.replace() — original objects unchanged.
```

---

## Removed Entities / Constants

| Name | Location | Replacement |
|------|----------|-------------|
| `IMAGE_MAX_WIDTH` | `config.py` | `IMAGE_REJECTION_CEILING` (longest-side semantics) |
| `IMAGE_MAX_HEIGHT` | `config.py` | `IMAGE_REJECTION_CEILING` (longest-side semantics) |
| Width/height dual-check in `validate_image_dimensions` | `media_io.py` | Single `max(w,h) > IMAGE_REJECTION_CEILING` check |
