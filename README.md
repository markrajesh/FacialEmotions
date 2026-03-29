# FacialEmotions

Local facial emotion detection with genuine-versus-posed analysis. Built with Python 3.11+ (tested on 3.13), Gradio, OpenCV, MediaPipe, and ONNX Runtime. Runs fully on-device — no cloud dependencies.

## Features

- **Single-image analysis** — upload a JPEG/PNG/BMP image and get per-face emotion labels, confidence scores, and genuine-vs-posed assessments
- **Multi-face support** — detect and report up to 5 faces per image with aligned results
- **Video analysis** — upload an MP4/AVI/MOV clip and receive timestamped frame-by-frame emotion summaries
- **Webcam live mode** — real-time emotion detection at ≥10 FPS with per-frame latency under 250 ms
- **Interpretable results** — Duchenne-style landmark heuristics explain why an expression is labelled genuine or posed

## Quickstart

```sh
# 1. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download or place the emotion ONNX model
#    Place at: models/emotion/emotion_model.onnx
#    (Any FER-2013-compatible 7-class ONNX model works; see quickstart.md)
#
# 4. Optional: download MediaPipe face landmarker for full landmark support
#    Place at: models/face_landmarks/face_landmarker.task
#    Without it, a geometry-estimate fallback is used automatically.

# 5. Run the application
python -m facial_emotions.app
```

The Gradio UI will open automatically at http://localhost:7860

## Running Tests

```sh
pytest --cov=facial_emotions --cov-report=term-missing
```

## Project Structure

```
src/facial_emotions/    # Application source
tests/                  # contract/, integration/, unit/
models/                 # emotion model files (not committed)
assets/fixtures/        # test images and videos
specs/                  # design documents and task list
```

## Privacy

All processing happens locally on your device. No images, video frames, or analysis results are sent to any server.

## Dataset & Ethics

See [specs/001-you-please-create/quickstart.md](specs/001-you-please-create/quickstart.md) for dataset provenance, preprocessing steps, privacy constraints, and ethical-use documentation.
