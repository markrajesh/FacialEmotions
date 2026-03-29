# FacialEmotions Development Guidelines

Auto-generated from all feature plans. Last updated: 2026-03-29

## Active Technologies
- Python 3.13.5 + OpenCV (`cv2`) for image I/O and resizing; ONNX Runtime for inference; Gradio for UI; MediaPipe for face detection; standard `logging` module (002-expand-image-size-limits)
- N/A — stateless single-request processing; no image persistence (002-expand-image-size-limits)

- Python 3.11 + Gradio, OpenCV, MediaPipe, NumPy, ONNX Runtime, scikit-learn, pytest, pytest-cov (001-you-please-create)

## Project Structure

```text
src/
tests/
```

## Commands

cd src; pytest; ruff check .

## Code Style

Python 3.11: Follow standard conventions

## Recent Changes
- 002-expand-image-size-limits: Added Python 3.13.5 + OpenCV (`cv2`) for image I/O and resizing; ONNX Runtime for inference; Gradio for UI; MediaPipe for face detection; standard `logging` module

- 001-you-please-create: Added Python 3.11 + Gradio, OpenCV, MediaPipe, NumPy, ONNX Runtime, scikit-learn, pytest, pytest-cov

<!-- MANUAL ADDITIONS START -->
<!-- MANUAL ADDITIONS END -->
