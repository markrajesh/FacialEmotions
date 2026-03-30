# Quickstart: Expanded Image Size Limits (feature 002)

This guide shows how to use and configure the expanded image size support added in feature 002.

---

## 1. Testing with a large image from the command line

Run the analysis service directly against a high-resolution image file:

```bash
python - <<'EOF'
from facial_emotions.services.image_analysis_service import ImageAnalysisService

svc = ImageAnalysisService()
result = svc.analyse_path("path/to/your/large_image.jpg")

print(f"Result dimensions : {result.image_width}×{result.image_height}")
print(f"Faces detected    : {len(result.faces)}")
for face in result.faces:
    print(f"  Face {face.face_id}: {face.emotion} ({face.emotion_confidence:.0%}), bbox={face.bbox}")
EOF
```

Images up to **8192 px on the longest side** are accepted.  
Images whose longest side exceeds **4096 px** are automatically downscaled to 4096 px for
analysis; bounding-box coordinates in the result are remapped back to the **original image
resolution**.

---

## 2. Overriding size limits via environment variables

All three size constants are configurable at startup through environment variables:

| Variable | Default | Description |
|---|---|---|
| `IMAGE_MIN_SIDE` | `224` | Minimum accepted side length (px). Must be ≥ 224 (ONNX model constraint). |
| `IMAGE_PROCESSING_CEILING` | `4096` | Longest-side ceiling for processing. Images above this are downscaled transparently. |
| `IMAGE_REJECTION_CEILING` | `8192` | Hard limit — images whose longest side exceeds this are rejected outright. |

### Example: tighten the rejection ceiling to 4096 px

```bash
IMAGE_REJECTION_CEILING=4096 python -m facial_emotions.cli analyse path/to/image.jpg
```

### Example: raise the processing ceiling to 6144 px

```bash
IMAGE_PROCESSING_CEILING=6144 python -m facial_emotions.cli analyse path/to/image.jpg
```

> **Note**: `IMAGE_PROCESSING_CEILING` must always be ≤ `IMAGE_REJECTION_CEILING`.

You can also set these in your `.env` file (see `.env.example` for the full list).

---

## 3. Expected Gradio UI notice text

When a large image is submitted via the Gradio web interface and auto-downscaled for
processing, the analysis summary will include a blockquote notice at the top:

```
> ℹ️ Large image auto-resized for processing (8000×6000 → 4096×3072).
```

The dimensions shown are `{original_width}×{original_height} → {processing_width}×{processing_height}`.

The bounding-box coordinates in the result (and in the annotated overlay image) are **always
expressed in original image coordinates**, regardless of whether downscaling occurred.
