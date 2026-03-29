# Test Video Fixtures

## Required Files

Place the following test videos in this directory before running video and webcam tests:

| Filename | Description | Expected Outcome |
|----------|-------------|-----------------|
| `short_clip.mp4` | 5-second clip, one face with changing expressions | timestamped frames |
| `multi_face_clip.mp4` | 5-second clip with 2 faces | per-face results |

### Video Requirements

- Format: MP4 (preferred), AVI, or MOV
- Duration: up to 600 seconds for integration tests; clips under 30 s for fast tests
- Consented subjects: same rules as images above

## Webcam Fixtures (`webcam/`)

Pre-recorded webcam capture frames for deterministic webcam pipeline tests.
