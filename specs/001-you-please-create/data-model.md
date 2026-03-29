# Data Model

## AnalysisRequest
- Purpose: Captures a user-initiated analysis job from the UI.
- Fields:
  - request_id: string, unique identifier
  - input_mode: enum(image, video, webcam)
  - source_path: optional string for uploaded file inputs
  - stream_id: optional string for active webcam sessions
  - enable_genuineness_check: boolean
  - sampling_rate_fps: optional float for video or webcam frame sampling
  - max_faces: integer, default 5
- Validation:
  - input_mode must be one of image, video, webcam
  - source_path is required for image and video
  - stream_id is required for webcam
  - max_faces must be between 1 and 10

## InputAsset
- Purpose: Describes the media under analysis.
- Fields:
  - asset_id: string
  - media_type: enum(image, video, webcam_frame)
  - format: enum(jpeg, png, bmp, mp4, avi, mov, live)
  - width: integer
  - height: integer
  - duration_seconds: optional float
  - frame_rate: optional float
- Validation:
  - images must use jpeg, png, or bmp
  - videos must use mp4, avi, or mov
  - dimensions must be positive integers

## FaceDetection
- Purpose: Represents one detected face in an image or frame.
- Fields:
  - face_id: string
  - bbox_x: integer
  - bbox_y: integer
  - bbox_width: integer
  - bbox_height: integer
  - detection_confidence: float
- Validation:
  - detection_confidence must be between 0.0 and 1.0
  - bounding box values must remain within frame bounds

## LandmarkSet
- Purpose: Stores extracted facial landmarks for a detected face.
- Fields:
  - face_id: string
  - landmark_points: list of 2D or 3D normalized points
  - eye_aspect_ratio: float
  - mouth_curvature: float
  - smile_symmetry_score: float
  - cheek_raise_score: float
- Validation:
  - landmark_points cannot be empty when a face is accepted for analysis
  - derived feature values must be finite numbers

## EmotionPrediction
- Purpose: Stores the predicted emotion label for a detected face.
- Fields:
  - face_id: string
  - emotion_label: enum(happiness, sadness, anger, surprise, fear, disgust, neutral)
  - confidence: float
  - top_k_scores: map<string, float>
- Validation:
  - confidence must be between 0.0 and 1.0
  - top_k_scores must sum to approximately 1.0 when probability output is available

## GenuinenessAssessment
- Purpose: Determines whether an expression is likely genuine or posed.
- Fields:
  - face_id: string
  - predicted_state: enum(genuine, posed, uncertain)
  - confidence: float
  - reasoning_flags: list<string>
  - classifier_version: string
- Validation:
  - predicted_state must be genuine, posed, or uncertain
  - reasoning_flags should record triggered heuristics or model features

## FrameAnalysis
- Purpose: Aggregates all outputs for one image or one sampled video/webcam frame.
- Fields:
  - frame_id: string
  - timestamp_ms: integer
  - faces: list<FaceDetection>
  - landmarks: list<LandmarkSet>
  - emotions: list<EmotionPrediction>
  - genuineness: list<GenuinenessAssessment>
- Relationships:
  - each face_id should appear consistently across faces, landmarks, emotions, and genuineness entries

## AnalysisSession
- Purpose: Captures the full result set for one request.
- Fields:
  - request: AnalysisRequest
  - asset: InputAsset
  - analyzed_frames: list<FrameAnalysis>
  - summary_counts: map<string, integer>
  - peak_emotion_segments: list<string>
  - generated_at: datetime string
- Validation:
  - image sessions should contain exactly one analyzed frame
  - video and webcam sessions may contain multiple sampled frames

## State Transitions
1. Request created
2. Media validated
3. Frames collected or sampled
4. Faces detected
5. Landmarks extracted
6. Emotion inference completed
7. Genuine-versus-posed assessment completed
8. Results rendered in UI and optionally exported
