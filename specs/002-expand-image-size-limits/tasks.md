# Tasks: Expand Image Size Limits

**Feature**: `002-expand-image-size-limits`  
**Branch**: `002-expand-image-size-limits`  
**Date**: 2026-03-29  
**Source**: plan.md + spec.md + research.md + data-model.md  
**Total tasks**: 22  
**Approach**: TDD — failing tests first, then implementation to make them pass

---

## Phase 1 — Setup (Foundational prerequisites)

> Goal: Establish the new configuration constants and update existing boundary tests to reflect the new rules. Everything downstream depends on these being correct first.

- [X] T001 Remove `IMAGE_MAX_WIDTH` and `IMAGE_MAX_HEIGHT` constants from `src/facial_emotions/config.py`; make `IMAGE_MIN_SIDE` env-configurable by changing it to `IMAGE_MIN_SIDE: int = int(os.getenv("IMAGE_MIN_SIDE", "224"))`; add `IMAGE_PROCESSING_CEILING: int = int(os.getenv("IMAGE_PROCESSING_CEILING", "4096"))` and `IMAGE_REJECTION_CEILING: int = int(os.getenv("IMAGE_REJECTION_CEILING", "8192"))` — this satisfies FR-009 which requires **all three** size constants to be env-configurable at startup
- [X] T002 Update `tests/unit/test_config.py`: add two tests — one verifying `IMAGE_PROCESSING_CEILING` defaults to `4096`, one verifying `IMAGE_REJECTION_CEILING` defaults to `8192`; add two more tests verifying each constant is overridable via its environment variable (use `monkeypatch.setenv` + `importlib.reload(config)`)
- [X] T003 Update `tests/integration/test_image_input_boundaries.py`: change `test_image_exceeding_max_width_rejected` (1921×1080) to assert the image is **accepted** (not rejected); change `test_image_exceeding_max_height_rejected` (1920×1081) to assert the image is **accepted**; add `test_image_exceeding_rejection_ceiling_rejected` using an image with longest side = 8193 px asserting `MediaIOError` with `match="too large"`; add `test_image_at_rejection_ceiling_accepted` using 8192×224 asserting acceptance

---

## Phase 2 — Foundational: `media_io.py` — validation + downscale utility

> Goal: Core size-validation logic and the new `downscale_to_processing_ceiling` function. All downstream pipeline and service tasks depend on this.
>
> **Independent Test**: `pytest tests/unit/tests_media_io_downscale.py tests/integration/test_image_input_boundaries.py` passes with 0 failures before touching the pipeline.

- [X] T004 [P] Write `tests/unit/test_media_io_downscale.py` with the following failing tests (the function does not exist yet):
  - `test_image_within_ceiling_returns_same_array_and_scale_1`: image 2000×1500 → scale_factor == 1.0, returned array is the same object
  - `test_image_at_ceiling_returns_same_array_and_scale_1`: image exactly 4096×3072 → scale_factor == 1.0
  - `test_landscape_image_downscaled`: 8192×6144 → longest side of returned frame == 4096, scale_factor ≈ 0.5
  - `test_portrait_image_downscaled`: 6048×8064 portrait image (longest=8064 > 4096) → verify frame longest side == 4096
  - `test_scale_factor_is_float_between_0_and_1`: any downscaled image → 0.0 < scale_factor < 1.0
  - `test_downscale_preserves_aspect_ratio`: after downscale, abs(frame_w/frame_h - orig_w/orig_h) < 0.01
  - `test_debug_log_emitted_on_downscale`: patch `facial_emotions.services.media_io._logger.debug` (the module-level logger, not the global `logging.Logger.debug`), call with 8192×6144 image, assert debug was called once

- [X] T005 Update `validate_image_dimensions` in `src/facial_emotions/services/media_io.py`: replace the `if w > config.IMAGE_MAX_WIDTH or h > config.IMAGE_MAX_HEIGHT` block with `if max(w, h) > config.IMAGE_REJECTION_CEILING` and update the error message to `f"Image is too large ({w}x{h}). Maximum accepted longest side is {config.IMAGE_REJECTION_CEILING}px."` Run `pytest tests/integration/test_image_input_boundaries.py` — all must pass

- [X] T006 Add `downscale_to_processing_ceiling(img: np.ndarray) -> tuple[np.ndarray, float]` to `src/facial_emotions/services/media_io.py` immediately after `validate_image_dimensions`. Add module-level logger `_logger = logging.getLogger(__name__)`. Implementation per plan.md §B. Run `pytest tests/unit/test_media_io_downscale.py` — all must pass

---

## Phase 3 — User Story 1: High-resolution image accepted and analysed (P1)

> **Story goal**: A user uploads a modern high-res image (> 1920×1080, ≤ 8192×8192). The system accepts it, auto-downscales to fit the processing ceiling, runs the full analysis pipeline, and returns face results with bounding boxes in original image coordinates.
>
> **Independent Test**: Upload a synthetic 4032×3024 numpy image through `ImageAnalysisService.analyse_array()` and verify: (a) no exception raised, (b) `result.image_width == 4032`, (c) `result.image_height == 3024`, (d) any detected face bbox values are within original image bounds.

- [X] T007 [P] [US1] Write `tests/integration/test_coordinate_remapping.py` with the following failing tests (pipeline does not support `original_image` param yet):
  - `test_analyse_returns_original_dimensions_when_downscaled`: create an 8000×6000 synthetic image (longest=8000 > 4096), downscale it to a processing frame, pass to `ImageAnalysisPipeline.analyse(frame, original_image=original)`; assert `result.image_width == 8000` and `result.image_height == 6000`
  - `test_no_downscale_path_result_dimensions_match_input`: 1920×1080 image, `original_image=None` → `result.image_width == 1920`
  - `test_bbox_remapped_to_original_coordinates`: use a mock `FaceDetector` that returns one face at (100, 100, 200, 200) on a 4096-px processing frame; call pipeline with `original_image` that is 2× larger (8192 px); assert returned `face.bbox["x"] == 200`, `face.bbox["y"] == 200`, `face.bbox["width"] == 400`
  - `test_bbox_remap_accuracy_within_2px`: compute expected remapped coordinate via `round(coord * inv_scale)`, assert `abs(actual - expected) <= 2`
  - `test_annotated_image_has_original_dimensions`: when `original_image` provided, `result.annotated_image.shape[:2] == (original_h, original_w)`

- [X] T008 [US1] Add `original_image: Optional[np.ndarray] = None` parameter to `ImageAnalysisPipeline.analyse()` in `src/facial_emotions/pipelines/image_pipeline.py`; add `import dataclasses` at top; add static method `_remap_detections(faces, inv_scale_x, inv_scale_y)` (separate axes — apply `inv_scale_x` to `bbox_x`/`bbox_width`, `inv_scale_y` to `bbox_y`/`bbox_height`; use `round()` on each result); update the `analyse` method body: after face detection, if `original_image is not None`, compute `inv_scale_x = original_image.shape[1] / bgr_image.shape[1]` and `inv_scale_y = original_image.shape[0] / bgr_image.shape[0]`, remap face detections, set `annotation_img = original_image`, render overlay and build result using `annotation_img` dimensions. Run `pytest tests/integration/test_coordinate_remapping.py` — all must pass

- [X] T009 [US1] Update `ImageAnalysisService.analyse_path()` in `src/facial_emotions/services/image_analysis_service.py`: import `downscale_to_processing_ceiling`; after `validate_image_dimensions`, call `frame, scale_factor = downscale_to_processing_ceiling(img)`; pass `original_image=img if scale_factor < 1.0 else None` to `self._pipeline.analyse(frame, ...)`. Update `analyse_array()` identically. Run `pytest tests/integration/test_image_input_boundaries.py tests/integration/test_coordinate_remapping.py` — all must pass

- [X] T010 [US1] Update `analyse_image` in `src/facial_emotions/ui/gradio_app.py`: add `downscale_to_processing_ceiling` to the **existing lazy import block** inside the function alongside `validate_image_dimensions` and `MediaIOError` (do not add a new top-level import); call `frame, scale_factor = downscale_to_processing_ceiling(bgr)` after validation; capture `orig_h, orig_w = bgr.shape[:2]` and `frame_h, frame_w = frame.shape[:2]`; pass `original_image=bgr if scale_factor < 1.0 else None` to `pipeline.analyse(frame, ...)`; prepend info-notice blockquote to summary when `scale_factor < 1.0` per plan.md §E. Add the following tests (in `tests/unit/` or a new `tests/unit/test_gradio_image_handler.py`): (a) `test_rejection_error_message_contains_too_large` — call `analyse_image` with a PIL image wrapping a 9000×9000 numpy array (no mock needed — validation fires before pipeline), assert returned text contains `"too large"` and returned image is not `None`; (b) `test_info_notice_present_when_downscaled` — call `analyse_image` with a synthetic 5000×4000 PIL image (mock the pipeline singleton to return an empty `ImageAnalysisResult`), assert returned text contains `"auto-resized"`. Run existing image integration tests — all must still pass

- [X] T011 [US1] Document `IMAGE_MIN_SIDE`, `IMAGE_PROCESSING_CEILING`, and `IMAGE_REJECTION_CEILING` in `.env.example` with descriptions, default values, and a note that `IMAGE_MIN_SIDE` must not be set below 224 (ONNX model input constraint)

---

## Phase 4 — User Story 2: Portrait and non-standard aspect ratios (P2)

> **Story goal**: Portrait-mode images (height > width) and panoramic images are accepted and processed correctly — proportional scaling only, no cropping, correct bbox positions.
>
> **Independent Test**: Pass a synthetic 3024×4032 portrait image through `ImageAnalysisService.analyse_array()` and verify no exception, `result.image_height == 4032`, result contains zero faces (synthetic blank image) but no size-related error.

- [X] T012 [P] [US2] Add portrait-orientation test cases to `tests/integration/test_image_input_boundaries.py`:
  - `test_portrait_4032x3024_accepted`: 4032-wide × 3024-tall longest=4032 — within ceiling → accepted with no error
  - `test_portrait_3024x4032_accepted`: 3024-wide × 4032-tall (portrait) longest=4032 — accepted
  - `test_tall_portrait_above_ceiling_downscaled_not_rejected`: 2000×9000 image (longest=9000 > 4096 but ≤ 8192) → verify `downscale_to_processing_ceiling` returns frame with longest side ≤ 4096
  - `test_panoramic_5000x1000_accepted`: longest=5000 > 4096 → `validate_image_dimensions` accepts it; `downscale_to_processing_ceiling` returns frame with longest side = 4096

- [X] T013 [US2] Add aspect-ratio preservation test to `tests/unit/test_media_io_downscale.py`:
  - `test_portrait_aspect_ratio_preserved`: input 2048×4096 (2:1 portrait) → frame should have w/h ratio ≈ 0.5 (within 1%)
  - `test_panoramic_aspect_ratio_preserved`: input 8192×2048 (4:1 landscape) → frame ratio ≈ 4.0 (within 1%)

---

## Phase 5 — User Story 3: Analysis completes within 2-second budget (P2)

> **Story goal**: The downscale step does not push analysis past the 2-second target for images up to the processing ceiling.
>
> **Independent Test**: Measure wall-clock time for `downscale_to_processing_ceiling` on an 8192×6144 synthetic image — must complete in under 200 ms.

- [X] T014 [P] [US3] Add performance test to `tests/integration/test_image_performance.py` (extend existing file):
  - `test_downscale_ceiling_completes_under_200ms`: create 8192×6144 numpy zeros image, time `downscale_to_processing_ceiling(img)`, assert elapsed < 0.2 s
  - `test_full_pipeline_4k_image_under_2s`: create 3840×2160 numpy zeros image, time `ImageAnalysisService().analyse_array(img)` end-to-end (mock detector to skip real detection), assert elapsed < 2.0 s

---

## Phase 6 — Polish & Cross-Cutting Concerns

> Goal: Regression safety, documentation, and final validation.

- [X] T015 [P] **Pre-commit gate**: Run `pytest -q` and confirm all 168 + newly added tests pass with zero failures — this gate must pass before T021 commit; do not proceed if any test fails

- [X] T016 [P] Review `tests/unit/test_config.py` to ensure no remaining references to `IMAGE_MAX_WIDTH` or `IMAGE_MAX_HEIGHT` — remove or update any that still exist

- [X] T017 [P] Search all test files for `match="Maximum supported size"` — update to match new error wording `"Maximum accepted longest side"` (see contracts/image-size-contract.md)

- [X] T018 [P] Verify `src/facial_emotions/services/media_io.py` no longer imports or references `IMAGE_MAX_WIDTH` / `IMAGE_MAX_HEIGHT` — remove any dead import if present

- [X] T019 [P] Verify `src/facial_emotions/ui/gradio_app.py` no longer calls `validate_image_dimensions` directly on the raw `bgr` and then separately calls the pipeline with the un-downscaled image — confirm the updated flow is coherent (validate → downscale → pipeline)

- [X] T020 [P] Add `quickstart.md` to `specs/002-expand-image-size-limits/` documenting: (1) how to test with a large image from the command line, (2) how to override processing ceiling and rejection ceiling via env vars, (3) expected Gradio UI notice text

- [X] T021 Commit all changes: `git add -A && git commit -m "feat(002): expand image size limits — 8192px ceiling, auto-downscale to 4096px processing frame, bbox coordinate remapping"`

- [X] T022 **SC-005 baseline**: Run `pytest -q` one final time after the commit; record the exact total test count (e.g. "X passed") as the new regression baseline, replacing the previous 168-test count for future features

---

## Dependencies

```
T001 (config constants)
  └─ T002 (config tests)
  └─ T003 (boundary test updates)
        └─ T004 (media_io downscale tests — written first as failing)
              └─ T005 (validate_image_dimensions update)
              └─ T006 (downscale_to_processing_ceiling implementation)
                    └─ T007 (coordinate remapping tests — written first as failing)
                          └─ T008 (pipeline original_image param)
                                └─ T009 (service layer downscale integration)
                                      └─ T010 (Gradio UI notice)
                                      └─ T011 (env.example docs)
                                └─ T012 (portrait boundary tests)
                                └─ T013 (aspect ratio unit tests)
                                └─ T014 (performance tests)
T015–T022 (polish — after all implementation tasks complete)
```

---

## Parallel Execution Opportunities

### Phase 1 (after T001 unblocks all):
- T002 and T003 can be written in parallel (different test files)

### Phase 2 (after T003):
- T004 (new test file) can be written in parallel with T005 + T006 (implementation)
  - Note: T004 tests will fail until T006 is complete — write them first per TDD

### Phase 3 (after T006):
- T007 (coordinate remapping tests) can be written before T008 (implementation)
- T011 (env.example docs) can be done any time after T001

### Phase 4 + 5 (after T009):
- T012, T013, T014 can all be worked in parallel — they touch different test files

### Phase 6 (after all implementation):
- T015, T016, T017, T018, T019 can all be run in parallel

---

## Implementation Strategy

**MVP scope (minimum to unblock US1 and confirm no regression):**
T001 → T003 → T005 → T006 → T008 → T009 → T015

This minimal path updates the config, fixes the validation, adds the downscale utility, wires it into the pipeline, and runs all existing tests. US2 and US3 tasks (T012–T014) add test coverage for orientation and performance but do not change behaviour beyond what US1 already delivers.

---

## Format Validation

All 22 tasks follow the required checklist format:
- ✅ Checkbox `- [ ]`
- ✅ Task ID (T001–T022) in execution order
- ✅ `[P]` marker on parallelizable tasks touching independent files
- ✅ `[US1]`/`[US2]`/`[US3]` labels on user-story-phase tasks
- ✅ Concrete file paths in every description
