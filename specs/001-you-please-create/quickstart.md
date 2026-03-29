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
1. Download the pre-trained ONNX emotion model (e.g. a FER-2013-trained model such as
   [this one](https://github.com/nicholasc8797/emotion-recognition-onnx) or any model with
   7-class output matching `config.EMOTION_LABELS`) and place it at:
   ```
   models/emotion/emotion_model.onnx
   ```
2. *(Optional)* Download the MediaPipe Face Landmarker task file for full landmark support:
   ```
   models/face_landmarks/face_landmarker.task
   ```
   Without it, a geometry-estimate fallback is used automatically.
3. Configure dataset and model paths via environment variables if needed (see `.env.example`).
4. Start the local UI:
   ```powershell
   python -m facial_emotions.app
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

---

## Dataset Provenance, Ethics, and Privacy

### Dataset: FER-2013

| Property | Detail |
|----------|--------|
| Name | Facial Expression Recognition 2013 (FER-2013) |
| Source | Kaggle / originally published at ICML 2013 by Goodfellow et al. |
| URL | https://www.kaggle.com/datasets/msambare/fer2013 |
| Licence | Open for academic and non-commercial research use |
| Format | 48×48 greyscale images, 7 emotion classes |
| Size | ~35,000 training / ~3,500 validation / ~3,500 test images |

### Preprocessing Steps

1. Download the FER-2013 `fer2013.csv` from Kaggle.
2. Run `src/facial_emotions/services/evaluation.py --prepare` to split the CSV into per-class directories under `assets/dataset/`.
3. Resize images to 48×48 (already in dataset) or convert to match the ONNX model's expected input size.
4. No face re-identification or de-anonymisation is performed.

### Privacy Constraints

- Images are used only for offline model evaluation and benchmark reporting.
- No dataset images are stored in session logs or transmitted.
- The held-out test split is never used during training or threshold tuning.
- All processing remains on the local device (see `LOCAL_ONLY = True` in `config.py`).

### Consent & Ethical Use

- FER-2013 images were collected from web searches and labelled via crowd-sourcing. Subjects did not individually give informed consent.
- **Acceptable uses**: academic research, internal benchmarking, student projects with non-commercial intent.
- **Unacceptable uses**: production surveillance, biometric identification, commercial sale of emotion-tagged profiles.
- The application must not be deployed in contexts where individuals can be identified without their knowledge or consent.
- Any custom dataset used in place of FER-2013 must carry equivalent documented provenance and consent records.

### Local-Only Inference Guarantee

All image, video, and webcam analysis is performed on the user's local device.
The flag `LOCAL_ONLY = True` in `src/facial_emotions/config.py` is an explicit
reminder that no network calls are made during analysis. The unit test
`tests/unit/test_config.py::test_local_only_is_true` verifies this at the
configuration level.
