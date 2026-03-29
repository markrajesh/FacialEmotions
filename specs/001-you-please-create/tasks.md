# Tasks: Facial Emotion Detection

**Input**: Design documents from `/specs/001-you-please-create/`
**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md, contracts/

**Tests**: Tests are required for this feature because the project constitution mandates test-driven development.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and basic structure

- [ ] T001 Create the Python application folder structure in src/facial_emotions/, tests/, models/, and assets/fixtures/
- [ ] T002 Initialize dependency and tooling files in requirements.txt, pyproject.toml, and .gitignore
- [ ] T003 [P] Create the package entry points in src/facial_emotions/__init__.py and src/facial_emotions/app.py
- [ ] T004 [P] Create baseline configuration scaffolding in src/facial_emotions/config.py and .env.example
- [ ] T005 [P] Add initial README setup and run instructions in README.md

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [ ] T006 Create shared enums and dataclasses in src/facial_emotions/domain/enums.py and src/facial_emotions/domain/models.py
- [ ] T007 [P] Implement contract result schemas in src/facial_emotions/contracts/results.py
- [ ] T008 [P] Implement media input validation and path handling utilities in src/facial_emotions/services/media_io.py
- [ ] T009 [P] Implement face detection service wrapper in src/facial_emotions/services/face_detection.py
- [ ] T010 [P] Implement landmark extraction and feature engineering in src/facial_emotions/services/landmark_features.py
- [ ] T011 [P] Implement pre-trained emotion inference wrapper in src/facial_emotions/services/emotion_inference.py
- [ ] T012 [P] Implement genuine-versus-posed assessment service in src/facial_emotions/services/genuineness_assessment.py
- [ ] T013 Implement analysis result rendering helpers in src/facial_emotions/services/overlay_renderer.py
- [ ] T014 Implement shared image and frame assembly logic in src/facial_emotions/pipelines/base_pipeline.py
- [ ] T015 Configure shared pytest fixtures and media samples in tests/conftest.py, assets/fixtures/images/, and assets/fixtures/videos/
- [ ] T016 Document dataset provenance, licensing, preprocessing, privacy, consent, and ethical-use constraints in README.md and specs/001-you-please-create/quickstart.md
- [ ] T017 Implement held-out evaluation dataset preparation and benchmark support in src/facial_emotions/services/evaluation.py and tests/conftest.py

**Checkpoint**: Foundation ready - user story implementation can now begin in parallel

---

## Phase 3: User Story 1 - Detect Facial Emotions (Priority: P1) 🎯 MVP

**Goal**: Analyze a single image, detect supported emotions with confidence scores, and produce a genuine-versus-posed assessment for each detected face.

**Independent Test**: Upload a JPEG or PNG image with known expressions and verify that the UI returns per-face emotion labels, confidence scores, and genuineness reasoning without depending on video or webcam flows.

### Tests for User Story 1 ⚠️

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [ ] T018 [P] [US1] Create contract tests for image analysis output in tests/contract/test_image_analysis_contract.py
- [ ] T019 [P] [US1] Create integration tests for single-image emotion detection in tests/integration/test_image_pipeline.py
- [ ] T020 [P] [US1] Create unit tests for emotion inference service behavior in tests/unit/test_emotion_inference.py
- [ ] T021 [P] [US1] Create unit tests for genuine assessment heuristics in tests/unit/test_genuineness_assessment.py
- [ ] T062 [P] [US1] Create static-image latency validation for the under-2-second target in tests/integration/test_image_performance.py
- [ ] T063 [P] [US1] Create image input boundary tests for supported and out-of-range sizes in tests/integration/test_image_input_boundaries.py
- [ ] T022 [US1] Approval gate: review and approve failing US1 tests before implementation begins in Phase 3

### Implementation for User Story 1

- [ ] T023 [US1] Implement the image analysis pipeline in src/facial_emotions/pipelines/image_pipeline.py
- [ ] T024 [US1] Implement image-specific request assembly and response mapping in src/facial_emotions/services/image_analysis_service.py
- [ ] T025 [US1] Implement the image upload and prediction UI flow in src/facial_emotions/ui/gradio_app.py
- [ ] T026 [US1] Connect the application entry point to the image workflow in src/facial_emotions/app.py
- [ ] T027 [US1] Add image analysis fixture metadata for known outcomes in assets/fixtures/images/README.md
- [ ] T028 [US1] Implement per-emotion accuracy benchmark tests in tests/integration/test_accuracy_benchmark.py and connect them to src/facial_emotions/services/evaluation.py

**Checkpoint**: At this point, User Story 1 should be fully functional and testable independently

---

## Phase 4: User Story 2 - Handle Multiple Faces (Priority: P2)

**Goal**: Detect and report emotions for multiple faces within a single image while preserving per-face bounding boxes and result alignment.

**Independent Test**: Analyze an image containing at least two faces and verify that the system returns separate face detections, emotion labels, confidence scores, and genuineness assessments for each face.

### Tests for User Story 2 ⚠️

- [ ] T029 [P] [US2] Create contract tests for multi-face image responses in tests/contract/test_multiface_contract.py
- [ ] T030 [P] [US2] Create integration tests for multi-face image analysis in tests/integration/test_multiface_image_pipeline.py
- [ ] T031 [P] [US2] Create unit tests for face-to-result alignment in tests/unit/test_face_detection_alignment.py
- [ ] T032 [US2] Approval gate: review and approve failing US2 tests before implementation begins in Phase 4

### Implementation for User Story 2

- [ ] T033 [US2] Extend face detection and matching logic for multi-face ordering in src/facial_emotions/services/face_detection.py
- [ ] T034 [US2] Extend landmark and emotion aggregation for multiple faces in src/facial_emotions/pipelines/image_pipeline.py
- [ ] T035 [US2] Update overlay rendering for per-face annotations in src/facial_emotions/services/overlay_renderer.py
- [ ] T036 [US2] Update the image UI to present multi-face summaries in src/facial_emotions/ui/gradio_app.py
- [ ] T037 [US2] Add multi-face image fixtures and expected outputs in assets/fixtures/images/multiface/

**Checkpoint**: At this point, User Stories 1 and 2 should both work independently

---

## Phase 5: User Story 4 - Video Emotion Detection (Priority: P2)

**Goal**: Analyze uploaded video files, sample frames, and return timestamped emotion and genuineness results for one or more faces across the clip.

**Independent Test**: Upload a short MP4 video and verify that the system returns timestamped frame analyses, including transitions between emotions and distinct results for multiple faces when they appear in the same frame.

### Tests for User Story 4 ⚠️

- [ ] T038 [P] [US4] Create contract tests for video analysis sessions in tests/contract/test_video_analysis_contract.py
- [ ] T039 [P] [US4] Create integration tests for timestamped video analysis in tests/integration/test_video_pipeline.py
- [ ] T040 [P] [US4] Create unit tests for frame sampling and timestamp generation in tests/unit/test_video_sampling.py
- [ ] T041 [US4] Approval gate: review and approve failing US4 tests before implementation begins in Phase 5

### Implementation for User Story 4

- [ ] T042 [US4] Implement the video analysis pipeline in src/facial_emotions/pipelines/video_pipeline.py
- [ ] T043 [US4] Implement video decoding, frame sampling, and timestamp extraction in src/facial_emotions/services/video_analysis_service.py
- [ ] T044 [US4] Extend contract mapping for aggregated video session results in src/facial_emotions/contracts/results.py
- [ ] T045 [US4] Add video upload, playback preview, and timestamped results UI in src/facial_emotions/ui/gradio_app.py
- [ ] T046 [US4] Add representative video fixtures and expected timestamp summaries in assets/fixtures/videos/README.md
- [ ] T047 [US4] Add video latency and timestamp-alignment validation in tests/integration/test_video_performance.py

**Checkpoint**: At this point, User Stories 1, 2, and 4 should be independently functional

---

## Phase 6: User Story 3 - Real-Time Emotion Detection (Priority: P3)

**Goal**: Process webcam input in near real time and display live emotion predictions with confidence scores and genuineness labels.

**Independent Test**: Start webcam mode and verify that the interface refreshes live predictions with bounded latency for visible faces without requiring the uploaded-video path.

### Tests for User Story 3 ⚠️

- [ ] T048 [P] [US3] Create contract tests for webcam frame analysis in tests/contract/test_webcam_contract.py
- [ ] T049 [P] [US3] Create integration tests for webcam session processing with recorded fixtures in tests/integration/test_webcam_pipeline.py
- [ ] T050 [P] [US3] Create unit tests for webcam frame throttling and refresh behavior in tests/unit/test_webcam_pipeline.py
- [ ] T051 [US3] Approval gate: review and approve failing US3 tests before implementation begins in Phase 6

### Implementation for User Story 3

- [ ] T052 [US3] Implement the webcam analysis pipeline in src/facial_emotions/pipelines/webcam_pipeline.py
- [ ] T053 [US3] Implement webcam capture orchestration and frame throttling in src/facial_emotions/services/webcam_session.py
- [ ] T054 [US3] Extend the Gradio UI for live webcam streaming and confidence updates in src/facial_emotions/ui/gradio_app.py
- [ ] T055 [US3] Wire webcam startup and shutdown handling in src/facial_emotions/app.py
- [ ] T056 [US3] Add recorded webcam test fixtures and expected outputs in assets/fixtures/videos/webcam/
- [ ] T057 [US3] Add webcam latency validation against the target FPS and per-frame inference thresholds in tests/integration/test_webcam_performance.py

**Checkpoint**: All user stories should now be independently functional

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories

- [ ] T058 [P] Finalize end-user setup, model download, and run documentation in README.md and specs/001-you-please-create/quickstart.md
- [ ] T059 Improve performance logging and configurable thresholds in src/facial_emotions/config.py and src/facial_emotions/app.py
- [ ] T060 [P] Add unit tests for shared domain models and configuration loading in tests/unit/test_domain_models.py and tests/unit/test_config.py
- [ ] T061 Validate the quickstart workflow against the implemented application in specs/001-you-please-create/quickstart.md
- [ ] T064 [P] Document and validate local-only inference assumptions in README.md, specs/001-you-please-create/quickstart.md, and tests/unit/test_config.py

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - blocks all user stories
- **User Stories (Phase 3+)**: All depend on Foundational phase completion
- **Polish (Phase 7)**: Depends on completion of the targeted user stories

### User Story Dependencies

- **User Story 1 (P1)**: Starts after Foundational - no dependency on other user stories
- **User Story 2 (P2)**: Starts after Foundational and builds on shared image-analysis foundations from User Story 1 while remaining independently testable
- **User Story 4 (P2)**: Starts after Foundational and reuses shared detection services plus evaluation support but does not depend on webcam support
- **User Story 3 (P3)**: Starts after Foundational and benefits from shared frame-analysis logic established by User Story 4

### Within Each User Story

- Tests must be written, reviewed, approved, and fail before implementation
- Core pipelines depend on shared services from Phase 2
- UI integration follows pipeline implementation
- Fixture and metadata updates complete the story for repeatable validation

### Parallel Opportunities

- T003, T004, and T005 can run in parallel during Setup
- T007 through T012 can run in parallel during Foundational once the domain model shape is agreed
- Test tasks within each user story can run in parallel
- User Story 2 and User Story 4 can proceed in parallel after User Story 1 establishes the baseline image workflow if team capacity allows

---

## Parallel Example: User Story 1

```text
Task: T018 Create contract tests for image analysis output in tests/contract/test_image_analysis_contract.py
Task: T019 Create integration tests for single-image emotion detection in tests/integration/test_image_pipeline.py
Task: T020 Create unit tests for emotion inference service behavior in tests/unit/test_emotion_inference.py
Task: T021 Create unit tests for genuine assessment heuristics in tests/unit/test_genuineness_assessment.py
```

## Parallel Example: User Story 4

```text
Task: T038 Create contract tests for video analysis sessions in tests/contract/test_video_analysis_contract.py
Task: T039 Create integration tests for timestamped video analysis in tests/integration/test_video_pipeline.py
Task: T040 Create unit tests for frame sampling and timestamp generation in tests/unit/test_video_sampling.py
```

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational
3. Complete Phase 3: User Story 1
4. Validate image upload analysis independently before expanding scope

### Incremental Delivery

1. Complete Setup and Foundational work
2. Deliver User Story 1 as the MVP image-analysis slice
3. Add User Story 2 for multi-face images
4. Add User Story 4 for uploaded-video analysis
5. Add User Story 3 for live webcam analysis
6. Finish with cross-cutting documentation, performance, and validation work

### Parallel Team Strategy

1. One developer owns shared services during Setup and Foundational phases
2. After Foundational, one developer can advance image and multi-face work while another builds video support
3. Webcam support can begin once the frame-analysis contracts from video processing are stable

## Notes

- All tasks follow the required checklist format with Task ID, optional parallel marker, story label where required, and exact file paths
- Story phases are independently testable and ordered by priority, with User Story 1 as the suggested MVP
- The generated tasks align with the feature plan, data model, quickstart, and analysis contract
