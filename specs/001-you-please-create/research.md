# Phase 0 Research

## Decision: Use Python 3.11 for the implementation baseline
- Rationale: Python 3.11 gives a good balance of runtime performance and package compatibility across OpenCV, MediaPipe, ONNX Runtime, Gradio, and scikit-learn on Windows.
- Alternatives considered: Python 3.10 was acceptable but slower; Python 3.12 was rejected because some computer vision packages still have weaker wheel support.

## Decision: Use Gradio for the simple UI layer
- Rationale: Gradio provides a low-friction Python UI for image upload, video upload, and webcam input without requiring a separate frontend codebase.
- Alternatives considered: Streamlit was viable but less natural for continuous media workflows; PySimpleGUI and PyQt were rejected for higher desktop UI maintenance cost.

## Decision: Use OpenCV plus MediaPipe for media handling, face detection, and landmark extraction
- Rationale: OpenCV covers image, video, and webcam ingestion, while MediaPipe gives efficient multi-face detection and dense landmarks needed for genuine-vs-posed feature extraction.
- Alternatives considered: Dlib was rejected for heavier setup and slower landmark extraction; MTCNN was rejected as too slow for CPU-first webcam use.

## Decision: Use a pre-trained emotion model through ONNX Runtime
- Rationale: ONNX Runtime keeps inference lightweight and portable on CPU while allowing a pre-trained facial-emotion model to be packaged behind a stable inference wrapper.
- Alternatives considered: DeepFace was rejected because it pulls in a heavier dependency stack; training a custom CNN from scratch was rejected because the project explicitly prefers pre-trained models.

## Decision: Start genuine-versus-fake detection with interpretable landmark heuristics, then allow a lightweight classifier on top
- Rationale: The constitution requires interpretable genuine-emotion analysis. Features such as smile symmetry, lip corner lift, eye constriction, blink frequency, and cheek raise can support a rule-based Duchenne-style assessment first, with an optional scikit-learn classifier later.
- Alternatives considered: An end-to-end black-box classifier was rejected because it is harder to justify and debug; a landmark-only thresholding system with no extensibility was rejected because it limits future tuning.

## Decision: Use FER-2013 as the primary public dataset for baseline validation and calibration
- Rationale: FER-2013 is widely available, easy to obtain, and sufficient for validating core emotion categories before real-world image, video, and webcam evaluation.
- Alternatives considered: AffectNet was rejected for higher acquisition complexity; relying only on ad hoc sample media was rejected because it would weaken reproducibility.

## Decision: Target CPU-first performance with selective frame processing
- Rationale: The system is intended to run locally on common student hardware. Processing every frame is unnecessary for video and webcam use, so the pipeline should sample frames for inference and interpolate display updates between analyzed frames.
- Alternatives considered: GPU-only assumptions were rejected because they narrow portability; full-frame dense inference at 30 FPS was rejected as unrealistic for the initial milestone.
