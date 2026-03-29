# Multi-Face Image Fixtures

This directory holds test images containing **two or more distinct faces** for validating
multi-face detection and per-face result alignment.

## Purpose

Images here are used by integration and contract tests to verify:
- All visible faces are detected and assigned unique `face_id` values.
- Each face receives an independent emotion prediction and genuineness assessment.
- Bounding boxes do not incorrectly overlap and are mapped to the correct identity.
- The `MAX_FACES` cap is respected (default: 5).

## Expected Directory Layout

```
multiface/
├── two_faces_happy.jpg         # Two subjects, both smiling
├── two_faces_mixed.jpg         # One happy, one neutral
├── five_faces_group.jpg        # Up to MAX_FACES subjects
└── README.md                   # This file
```

## Sourcing Guidelines

- Only use images where **all subjects have provided explicit written consent** for use
  in emotion-analysis research.
- Preferred sources: self-captured images with written consent forms, or a dataset
  licensed for academic use (e.g. AffectNet, RAF-DB) that permits redistribution.
- Do **not** commit images scraped from the internet or social media — this violates
  privacy and likely breaches copyright.
- Strip all EXIF metadata before committing any image file.

## Adding a New Fixture

1. Place the image in this directory using a descriptive snake_case filename.
2. Record the expected number of faces and dominant emotions in the table below.
3. Add a corresponding entry in `tests/conftest.py` if a new shared fixture is needed.

## Known Fixtures

| Filename | Faces | Expected Emotions | Notes |
|----------|-------|-------------------|-------|
| *(none committed — add real images here)* | — | — | Synthetic fixtures used in tests via mocks |

## Ethics & Privacy

All fixture images must comply with the project's data ethics policy documented in
[`specs/001-you-please-create/quickstart.md`](../../../../specs/001-you-please-create/quickstart.md)
and in the repository root [README.md](../../../../README.md).
