# Quickstart

## Prerequisites
- Python 3.11
- A webcam for live testing
- Local copies of the chosen public facial-emotion dataset and the pre-trained emotion model file

## Environment Setup
1. Create a virtual environment:
   ```powershell
   py -3.11 -m venv .venv
   ```
2. Activate it:
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```
3. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```

## Recommended Dependency Set
Add these packages to the initial requirements file:
- gradio
- opencv-python
- mediapipe
- numpy
- scikit-learn
- onnxruntime
- pytest
- pytest-cov

## Project Bootstrap
1. Place the pre-trained emotion model under a local models directory.
2. Configure dataset and model paths in the application settings file.
3. Start the local UI:
   ```powershell
   python -m facial_emotions.ui.gradio_app
   ```

## Manual Validation Flow
1. Upload a JPEG or PNG image and confirm that detected faces, emotions, and genuineness labels are displayed.
2. Upload a short MP4 video and confirm that sampled timestamps and per-face results are shown.
3. Start webcam mode and confirm that live predictions refresh with bounded latency and confidence scores.

## Test Commands
- Run unit and integration tests:
  ```powershell
  pytest
  ```
- Run coverage:
  ```powershell
  pytest --cov=src --cov-report=term-missing
  ```
