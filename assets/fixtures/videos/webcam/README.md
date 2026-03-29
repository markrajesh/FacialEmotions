# Webcam Test Fixtures

This directory contains pre-recorded webcam fixture frames for deterministic
webcam pipeline tests that do not require real camera hardware.

## Purpose

Files here are used by integration and performance tests to verify:
- `WebcamAnalysisPipeline.analyse_frame()` handles real-resolution BGR frames correctly.
- Per-frame latency stays within the `WEBCAM_FRAME_BUDGET_MS` threshold (default 250 ms).
- The frame throttle (`should_process_frame`) correctly gates frames to `WEBCAM_TARGET_FPS`.

## Expected Directory Layout

```
webcam/
├── single_face_neutral.npy     # Serialised numpy BGR array, 480×640×3
├── single_face_happy.npy       # Subject smiling — genuineness = genuine
├── no_face.npy                 # Frame with no face — expects empty faces list
└── README.md                   # This file
```

## Adding a Fixture

1. Capture a frame from a local camera using OpenCV:
   ```python
   import cv2, numpy as np
   cap = cv2.VideoCapture(0)
   ret, frame = cap.read()
   cap.release()
   np.save("single_face_neutral.npy", frame)
   ```
2. Verify the subject has consented to use in ML research.
3. Strip any identifying metadata from the file.
4. Document the expected outcomes in the table below.

## Known Fixtures

| Filename | Faces | Expected Emotion | Notes |
|----------|-------|-----------------|-------|
| *(none committed)* | — | — | Mocked in tests via numpy synthetic frames |

## Privacy & Ethics

All webcam fixtures must comply with the project ethics policy in
[`specs/001-you-please-create/quickstart.md`](../../../../../specs/001-you-please-create/quickstart.md).
No images of unconsenting individuals may be committed to this repository.
