# Feature Specification: Expand Image Size Limits

**Feature Branch**: `002-expand-image-size-limits`  
**Created**: 2026-03-29  
**Status**: Ready for Planning  
**Input**: User description: "Looks like the size of the image is too small. Most of the images are much bigger than the current supported size. can you please review and update the specs accordingly to support bigger size which are normal nowadays"

## Context

The existing application caps accepted images at **1920×1080 px** (Full HD) and rejects anything larger with an error. Modern image sources routinely produce much larger files:

| Source | Typical resolution |
|--------|--------------------|
| Smartphone (12–200 MP) | 4032×3024 – 16384×12288 |
| DSLR / mirrorless | 4000×6000 – 8736×5824 |
| 4K screenshot / video frame | 3840×2160 |
| 8K display screenshot | 7680×4320 |
| Scanned portrait / group photo | Up to 10000 px on longest side |

Rejecting these images prevents the application from being useful with real-world inputs. Rather than requiring users to manually downscale before uploading, the system should handle large images automatically.

## User Scenarios & Testing

### User Story 1 - Upload a Modern High-Resolution Image (Priority: P1)

A user uploads a photo taken on a modern smartphone or camera (e.g. 4K, 4032×3024, or larger) and expects the application to detect and analyse faces without any manual preprocessing.

**Why this priority**: This is the most common real-world use case and the main pain point. Users should not need external tools to downscale images before analysing them.

**Independent Test**: Upload a 4032×3024 JPEG from a smartphone and verify that face detection, emotion labelling, and genuineness assessment return results — exactly as they would for a smaller image.

**Acceptance Scenarios**:

1. **Given** an image with width > 1920 px or height > 1080 px, **When** the user submits it for analysis, **Then** the system accepts it, automatically scales it to fit within processing bounds, displays a subtle info notice ("Large image auto-resized for processing"), and returns per-face results.
2. **Given** a 4K image (3840×2160), **When** analysed, **Then** bounding boxes in the result are expressed relative to the **original** image dimensions so annotations overlay correctly on the source image.
3. **Given** an image whose longest side exceeds 8192 px (e.g. 9000×7000 px), **When** submitted, **Then** the system rejects it with a clear, friendly message stating the maximum accepted resolution.

---

### User Story 2 - Portrait-Orientation and Non-Standard Aspect Ratios (Priority: P2)

A user uploads a portrait-mode photo (e.g. 3024×4032, taller than wide) or an ultra-wide panorama, and expects the system to handle it without distorting faces or cropping content.

**Why this priority**: Portrait photos are extremely common (selfies, professional headshots) and the old 1920×1080 landscape-biased cap rejects them.

**Independent Test**: Upload a 3024×4032 portrait-mode image and verify that the annotated result retains correct proportions and face bounding boxes align with actual face positions.

**Acceptance Scenarios**:

1. **Given** a portrait-mode image (height > width), **When** submitted, **Then** the system accepts it and scales proportionally rather than cropping.
2. **Given** a panoramic image with a very wide aspect ratio (e.g. 5:1), **When** submitted, **Then** the system processes it without error and detects faces anywhere across the full width.

---

### User Story 3 - Analysis Completes Within Time Budget for Large Images (Priority: P2)

A user uploads a high-resolution image and expects the full analysis to still complete within the existing 2-second target.

**Why this priority**: Accepting larger images must not degrade perceived performance. Auto-downscaling before analysis ensures the inference step remains fast.

**Independent Test**: Submit a 4K image and measure wall-clock time from submission to result — must be under 2 seconds.

**Acceptance Scenarios**:

1. **Given** any accepted image resolution up to the new ceiling, **When** analysed, **Then** the result is returned in under 2 seconds.
2. **Given** a large image submitted multiple times in sequence, **When** measured, **Then** each analysis remains within the time budget.

---

### Edge Cases

- What happens when the image is exactly at the new maximum ceiling? → Accepted and processed normally.
- What happens when only one dimension exceeds the old cap but the other does not? → Still accepted; the system scales down uniformly preserving aspect ratio.
- What happens when a very large image contains no face? → Accepted, scaled, returns "no face detected" — not a size error.
- What happens when an image is below the minimum side length (224 px)? → Rejected with a clear message; the minimum floor is unchanged.
- How does the system handle memory for very large images? → Auto-downscale to the processing ceiling before the pipeline, limiting peak memory use.

---

## Requirements

### Functional Requirements

- **FR-001**: The system MUST accept images up to **8192 × 8192 px** (~67 MP), covering all common smartphone, DSLR, and 8K display sources.
- **FR-002**: The system MUST automatically downscale any image whose longest side exceeds a configurable **processing ceiling** before running face detection and emotion inference. The original dimensions are preserved for annotation purposes.
- **FR-003**: The default processing ceiling MUST be **4096 px** on the longest side, balancing detail retention with processing speed. This value MUST be configurable via environment variable without a code change.
- **FR-004**: Downscaling MUST preserve aspect ratio using proportional scaling; no cropping or distortion of content is permitted.
- **FR-005**: Bounding boxes returned in results MUST be expressed in **original image coordinates**, not in the downscaled processing frame, so that annotation overlays align correctly on the source image.
- **FR-006**: The system MUST reject images whose longest side exceeds **8192 px** with a clear message stating the maximum resolution and the actual image dimensions submitted.
- **FR-007**: The minimum accepted side length (224 px) MUST remain unchanged.
- **FR-008**: The UI MUST display a user-friendly error message when an oversized image is rejected (FR-006), clearly distinct from other error types.
- **FR-009**: All image size configuration values (minimum side, processing ceiling, rejection ceiling) MUST be changeable via environment variables, read at application startup.
- **FR-010**: When an image is automatically downscaled (FR-002), the UI MUST display a subtle, non-blocking info notice to the user (e.g., "Large image auto-resized for processing"). This notice is distinct from error messages and must not interrupt the results view.
- **FR-011**: When an image is automatically downscaled (FR-002), the application MUST emit a DEBUG-level log entry recording the original image dimensions, the processing frame dimensions, and the scale factor applied.

### Key Entities

- **Image Size Constraint**: Minimum floor, processing ceiling, and rejection ceiling expressed as pixel dimensions — drives validation, auto-scaling, and error messaging.
- **Processing Frame**: The downscaled internal representation of a large image used for face detection and inference. Never shown to the user.
- **Scale Factor**: The ratio between original image dimensions and processing frame dimensions, used to remap bounding box coordinates back to original space.

---

## Success Criteria

### Measurable Outcomes

- **SC-001**: Images up to 4032×3024 (standard 12 MP smartphone) are accepted and analysed in under 2 seconds in 95% of test cases.
- **SC-002**: Zero user-visible errors for images within the new 8192×8192 px ceiling that were previously blocked by the old 1920×1080 cap.
- **SC-003**: Bounding box coordinates on annotated results are accurate to within ±2 px measured in **original image pixel coordinates** after coordinate remapping from the processing frame.
- **SC-004**: The processing ceiling is successfully overridden by setting an environment variable, verified by a configuration unit test.
- **SC-005**: All 168 existing tests continue to pass after the change (no regressions).

---

## Assumptions

- The existing letterbox resize utility (`resize_for_model` in `media_io.py`) is extended or complemented to support full-image pre-downscaling before face detection, not only for model input preparation.
- Users upload images from disk via the Gradio UI; URL-based or streaming ingestion is out of scope for this feature.
- The video file and webcam pipelines are **out of scope** for this feature; size limit changes apply to static image uploads only.
- Accepted file formats (JPEG, PNG, BMP) are **unchanged** by this feature; WebP, HEIC/HEIF, and TIFF support are out of scope.
- Memory is sufficient to hold one full original image at up to 67 MP (~200 MB as raw RGB) during coordinate remapping; a processing ceiling of 4096 px keeps the working copy under ~50 MB.
- Python 3.13 with the current OpenCV and NumPy stack can decode JPEG/PNG images up to 8192×8192 px without special configuration.
- The 224 px minimum side floor is driven by the ONNX model's input requirement and is not affected by this feature.

## Clarifications

### Session 2026-03-29

- Q: When an image is automatically downscaled before processing, should the UI notify the user? → A: Show a subtle info notice (e.g., "Large image auto-resized for processing") — not silent, not overly detailed.
- Q: Do the new size limits and auto-downscale apply to video/webcam pipelines? → A: Images only — video and webcam pipelines are out of scope for this feature.
- Q: Should this feature extend accepted file formats (e.g., WebP, HEIC)? → A: Formats unchanged — JPEG, PNG, BMP only; format expansion is out of scope.
- Q: SC-003 bounding box ±2 px accuracy — measured in original image pixels or display-scaled pixels? → A: Original image pixel coordinates.
- Q: Should auto-downscale events be logged for diagnostics? → A: Yes, log at DEBUG level (original size, processing size, scale factor).
