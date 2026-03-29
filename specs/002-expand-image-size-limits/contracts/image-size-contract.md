# Contracts: Expand Image Size Limits

**Feature**: 002-expand-image-size-limits  
**Date**: 2026-03-29

---

## Analysis Result Contract — No Breaking Changes

The `ImageAnalysisResult` schema (defined in `src/facial_emotions/contracts/results.py` and tested in `tests/contract/`) is **unchanged** by this feature. All field names and types are identical.

The key invariant that **tightens** (semantically, not structurally):

> `FaceResult.bbox` coordinates (`x`, `y`, `width`, `height`) are **always expressed in original image pixel coordinates**, regardless of whether internal downscaling occurred during processing.

This was already the expected contract; the feature makes it technically true for large images.

---

## New Configuration Contract (env vars)

These environment variables are now part of the application's public configuration surface. They must be documented in `.env.example`.

| Variable | Type | Default | Valid range | Description |
|----------|------|---------|-------------|-------------|
| `IMAGE_PROCESSING_CEILING` | integer | `4096` | 224–IMAGE_REJECTION_CEILING | Longest-side pixel threshold above which images are auto-downscaled before analysis |
| `IMAGE_REJECTION_CEILING` | integer | `8192` | IMAGE_PROCESSING_CEILING+1 – any | Longest-side pixel threshold above which images are rejected with an error |

**Removed env vars**: None (the old `IMAGE_MAX_WIDTH` / `IMAGE_MAX_HEIGHT` were not env-configurable — they were hard-coded constants). No downstream env files need updating other than adding the new variables to `.env.example`.

---

## Error Message Contract

Error messages returned to callers of `validate_image_dimensions()` have a defined format:

### Too small (unchanged)
```
Image is too small ({w}x{h}). Minimum side must be ≥ {IMAGE_MIN_SIDE}px.
```

### Too large (updated)
Old:
```
Image is too large ({w}x{h}). Maximum supported size is {IMAGE_MAX_WIDTH}x{IMAGE_MAX_HEIGHT}.
```
New:
```
Image is too large ({w}x{h}). Maximum accepted longest side is {IMAGE_REJECTION_CEILING}px.
```

This change affects:
- The error message string returned to end users via the Gradio UI
- Tests that `match="too large"` (still match — the word remains)
- Tests that `match="Maximum supported size"` → must be updated to match new wording

---

## UI Notice Contract (new)

When auto-downscaling occurs, the Gradio `analyse_image` handler prepends an info notice to the markdown summary string:

```
> ℹ️ Large image auto-resized for processing ({orig_w}×{orig_h} → {frame_w}×{frame_h}).
```

This is a soft contract: it only appears in the Gradio UI text output and is not part of any machine-readable API. No downstream systems consume this text.
