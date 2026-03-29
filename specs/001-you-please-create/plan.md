# Implementation Plan: Facial Emotion Detection

**Branch**: `[001-you-please-create]` | **Date**: March 28, 2026 | **Spec**: `/specs/001-you-please-create/spec.md`
**Input**: Feature specification from `/specs/001-you-please-create/spec.md`

## Summary

Build a local Python 3.11 application with a simple Gradio UI that analyzes images, uploaded videos, and webcam input to detect facial emotions, report confidence scores, support multi-face scenes, and assess whether expressions are likely genuine or posed. The implementation will combine OpenCV media handling, MediaPipe face detection and facial landmarks, a CPU-friendly pre-trained emotion model executed through ONNX Runtime, and an interpretable landmark-based genuine-expression assessment that can later be refined with a lightweight scikit-learn classifier.

## Technical Context

**Language/Version**: Python 3.11  
**Primary Dependencies**: Gradio, OpenCV, MediaPipe, NumPy, ONNX Runtime, scikit-learn, pytest, pytest-cov  
**Storage**: Local filesystem only for configuration, model weights, temporary analysis outputs, and test fixtures  
**Testing**: pytest for unit, contract, and integration tests; fixed image and video fixtures for reproducible media tests  
**Target Platform**: Local Windows-first desktop/laptop execution with webcam support; keep cross-platform compatibility for macOS and Linux where dependencies allow  
**Project Type**: Single Python application with lightweight browser-based UI  
**Performance Goals**: Static image analysis under 2 seconds end-to-end; webcam refresh at 10 to 15 analyzed FPS with per-frame inference under 250 ms on mid-range CPU; offline video analysis with timestamped summaries for sampled frames  
**Constraints**: CPU-first execution, no cloud dependency, support JPEG/PNG/BMP and MP4/AVI/MOV inputs, handle up to 5 faces per frame, provide interpretable genuine-versus-posed reasoning flags, preserve local privacy by keeping processing on-device  
**Scale/Scope**: Single-user research/demo application, one active analysis session at a time, public dataset-backed calibration, short uploaded videos up to roughly 10 minutes for the initial milestone

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

### Pre-Research Gate Review

- Genuine Emotion Analysis: PASS. The plan explicitly includes landmark-derived features such as mouth curvature, smile symmetry, eye constriction, and cheek raise to support genuine-versus-posed analysis.
- Dataset Utilization: PASS. The plan uses a public facial-emotion dataset for baseline calibration and requires documented preprocessing and local data handling.
- Pre-Trained Models: PASS. Emotion detection is based on a pre-trained model rather than custom end-to-end training.
- Rule-Based and ML Classifiers: PASS. The first implementation uses interpretable heuristics and leaves room for a lightweight scikit-learn classifier without breaking the contract.
- Test-Driven Development: PASS. The project structure and quickstart define unit, integration, and contract testing as first-class work.

### Post-Design Gate Review

- Genuine Emotion Analysis: PASS. The data model records landmark-derived features and reasoning flags, and the contract preserves genuineness outputs separately from raw emotion labels.
- Dataset Utilization: PASS. The design keeps dataset use in offline preparation and evaluation rather than user-session storage.
- Pre-Trained Models: PASS. The design isolates inference behind a service boundary so model files can be swapped without UI changes.
- Rule-Based and ML Classifiers: PASS. The design supports both rule-based scoring and optional classifier-backed assessment through the same assessment schema.
- Test-Driven Development: PASS. Contract, unit, and integration test layers are reflected in the proposed source structure.

## Project Structure

### Documentation (this feature)

```text
specs/001-you-please-create/
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
│   └── analysis-contract.yaml
└── tasks.md
```

### Source Code (repository root)

```text
src/
└── facial_emotions/
    ├── app.py
    ├── config.py
    ├── contracts/
    │   └── results.py
    ├── domain/
    │   ├── enums.py
    │   └── models.py
    ├── pipelines/
    │   ├── image_pipeline.py
    │   ├── video_pipeline.py
    │   └── webcam_pipeline.py
    ├── services/
    │   ├── emotion_inference.py
    │   ├── face_detection.py
    │   ├── genuineness_assessment.py
    │   ├── landmark_features.py
    │   └── overlay_renderer.py
    └── ui/
        └── gradio_app.py

tests/
├── contract/
├── integration/
└── unit/

models/
├── emotion/
└── metadata/

assets/
└── fixtures/
    ├── images/
    └── videos/
```

**Structure Decision**: Use a single Python project with explicit separation between UI, media pipelines, domain models, and inference services. This keeps the implementation simple enough for a student project while still isolating the pre-trained model wrapper and the genuine-expression logic behind clear contracts.

## Complexity Tracking

No constitution violations are currently required by the design.
