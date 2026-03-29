# Research: Expand Image Size Limits

**Feature**: 002-expand-image-size-limits  
**Date**: 2026-03-29  
**Status**: Complete — all NEEDS CLARIFICATION items resolved

---

## 1. How to validate against a "longest-side" ceiling rather than width × height separately

**Decision**: Replace the separate `IMAGE_MAX_WIDTH` / `IMAGE_MAX_HEIGHT` checks with a single `max(w, h) > IMAGE_REJECTION_CEILING` guard.

**Rationale**: The old approach had two fixed values (1920 wide, 1080 tall) that biased against portrait images. A single longest-side ceiling treats all orientations equally, which is the correct semantic for this domain.

**Implementation**:
```python
longest = max(w, h)
if longest > config.IMAGE_REJECTION_CEILING:
    raise MediaIOError(
        f"Image is too large ({w}×{h}). "
        f"Maximum accepted longest side is {config.IMAGE_REJECTION_CEILING} px."
    )
```

**Alternatives considered**:
- Keep width + height caps but raise them both to 8192 → inconsistent for portrait images (8192×8192 accepted but 4096×8193 rejected at height). Rejected.
- Megapixel-based cap (67 MP) → requires multiplication; error messages are less intuitive for users. Rejected.

---

## 2. How to proportionally downscale a large image to a processing ceiling

**Decision**: Add `downscale_to_processing_ceiling(img) -> tuple[np.ndarray, float]` to `media_io.py`. Uses `cv2.INTER_AREA` (best for downscaling), returns `(frame, scale_factor)` where `scale_factor = processing_ceiling / longest_side`. For images already within the ceiling, returns `(img, 1.0)` with no copy.

**Rationale**: `cv2.INTER_AREA` averages pixels when shrinking, which preserves face detail better than nearest-neighbour or bilinear at large reduction ratios. It is already used by `resize_for_model`.

**Scale factor arithmetic**:
```
scale_factor = IMAGE_PROCESSING_CEILING / max(original_w, original_h)
new_w = round(original_w * scale_factor)
new_h = round(original_h * scale_factor)
```

`round()` rather than `int()` avoids systematic floor bias on odd-sized frames.

**Alternatives considered**:
- Tile the image into 4096×4096 tiles → massively more complex; face detections would miss faces spanning tile boundaries. Rejected.
- Resize inside the pipeline — keeps `media_io.py` as a pure validation layer with no mutation. Decided to still put it in `media_io.py` because the service layer already imports from there and the transform is purely I/O-adjacent.

---

## 3. Where to perform bbox coordinate remapping (pipeline vs service layer)

**Decision**: Remap inside `ImageAnalysisPipeline.analyse()` by adding an optional `original_image: np.ndarray | None = None` parameter. When provided:
1. All face detection and inference run on `bgr_image` (the processing frame).
2. Before annotation rendering, bbox coordinates are scaled up by `1 / scale_factor`.
3. `draw_results` runs on `original_image`, not `bgr_image`.
4. `ImageAnalysisResult.image_width/height` reflect original dimensions.

**Rationale**: This keeps coordinate remapping encapsulated inside the pipeline, avoids exposing raw face-detection objects outside it, and requires only one new parameter. The service layer passes `original_image=original_bgr` when `scale_factor < 1.0`.

**Scale-up formula**:
```python
def _remap_detection(face: FaceDetection, inv_scale: float) -> FaceDetection:
    return dataclasses.replace(
        face,
        bbox_x=round(face.bbox_x * inv_scale),
        bbox_y=round(face.bbox_y * inv_scale),
        bbox_width=round(face.bbox_width * inv_scale),
        bbox_height=round(face.bbox_height * inv_scale),
    )
```

`dataclasses.replace` is safe because `FaceDetection` is a dataclass (confirmed in `domain/models.py`).

**±2 px accuracy** (SC-003): With `cv2.INTER_AREA` and `round()` rather than `int()`, the maximum rounding error per coordinate is 0.5 px × inv_scale. At the worst case (4× upscale from 1024 → 4096 input), error ≤ 2 px. Within SC-003 tolerance.

**Alternatives considered**:
- Remap in `image_analysis_service.py` by scaling `FaceResult.bbox` dict after the fact → requires re-rendering annotation with a separate `draw_results` call outside the pipeline, which would duplicate the rendering dependency. Rejected.
- Keep pipeline as-is and just render annotation at low resolution, then upscale the annotated image → visible quality loss at 4× ratios. Rejected.

---

## 4. Memory budget for 8192×8192 images

**Decision**: The spec assumption (≈200 MB for one raw RGB 8192×8192 image) is confirmed. Python/Pillow/OpenCV load images into contiguous uint8 arrays: 8192 × 8192 × 3 bytes = 201 MB. After `downscale_to_processing_ceiling` the processing frame is 4096 × 4096 × 3 = 50 MB. Both can coexist in RAM simultaneously (251 MB total) on any machine with ≥512 MB RAM, which is the realistic minimum for running the ONNX model and Gradio anyway.

**No memory guard needed** for this feature (per spec Assumption 3). A future feature can add a configurable hard RAM limit.

---

## 5. cv2.INTER_AREA suitability confirmed

OpenCV documentation confirms `INTER_AREA` is preferred for image decimation (downscaling). For the reduction ratios expected (1.0× to ~13× for 8K→4096), it outperforms `INTER_LINEAR` and `INTER_CUBIC` in terms of aliasing and moiré artefacts on skin texture, which is important for face detection accuracy.

---

## 6. DEBUG logging approach

**Decision**: Use Python's standard `logging` module at `DEBUG` level. The module-level logger in `media_io.py` (or the service layer) emits one line:
```python
logger.debug(
    "Image downscaled for processing: original=%dx%d → frame=%dx%d (scale=%.4f)",
    orig_w, orig_h, frame_w, frame_h, scale_factor,
)
```

No third-party logging library is needed. The app already uses Python's `logging` (verified in other services).

---

## 7. Existing tests that will break and require updating

| Test | Problem | Fix |
|------|---------|-----|
| `test_image_exceeding_max_width_rejected` (1921×1080) | Old test expects rejection; new ceiling is 8192 → image should now be accepted | Update to assert accepted |
| `test_image_exceeding_max_height_rejected` (1920×1081) | Same as above | Update to assert accepted |
| Any test that passes `IMAGE_MAX_WIDTH` or `IMAGE_MAX_HEIGHT` from config | Constants are being removed | Update to new constant names |

No other tests are expected to break: the minimum‑side floor (224 px) is unchanged, and the pipeline interface gains an optional parameter with a default of `None`.

---

## 8. UI info notice — Gradio implementation

Gradio's `gr.Image` + `gr.Markdown` outputs already exist. The notice string is returned alongside the annotated image as part of the text output. The simplest approach appends a line to the existing summary string rather than adding a new output component:
```python
if scale_factor < 1.0:
    summary_lines.insert(0, f"> ℹ️ Large image auto-resized for processing ({orig_w}×{orig_h} → {frame_w}×{frame_h}).\n")
```
This is rendered as a blockquote callout in Markdown, visually distinct from results.

---

## Summary of Decisions

| # | Question | Decision |
|---|----------|---------|
| 1 | Validation rule | Single `max(w, h) > REJECTION_CEILING` guard |
| 2 | Downscale algorithm | `cv2.INTER_AREA`, in `media_io.py` |
| 3 | Coordinate remapping | Inside pipeline via `original_image` param |
| 4 | Memory guard | Not needed for this feature |
| 5 | Downscale algorithm quality | INTER_AREA confirmed best for decimation |
| 6 | Logging | `logging.debug()` in media_io/service |
| 7 | Broken tests | 2 tests need updating, rest unaffected |
| 8 | UI notice | Markdown blockquote prepended to summary |
