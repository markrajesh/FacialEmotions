"""Contract tests for image analysis output schema.

These tests verify that the ImageAnalysisResult returned from the analysis
pipeline satisfies the structural contract defined in contracts/results.py.
They should FAIL before the image pipeline is implemented.
"""

from __future__ import annotations

import pytest

from facial_emotions.contracts.results import FaceResult, ImageAnalysisResult


class TestImageAnalysisResultSchema:
    def test_image_analysis_result_has_required_fields(self):
        result = ImageAnalysisResult(
            request_id="req-001",
            image_width=640,
            image_height=480,
            faces=[],
        )
        assert result.request_id == "req-001"
        assert result.image_width == 640
        assert result.image_height == 480
        assert result.faces == []

    def test_face_result_has_required_fields(self):
        face = FaceResult(
            face_id="face-001",
            bbox={"x": 0, "y": 0, "width": 100, "height": 100},
            detection_confidence=0.9,
            emotion="happiness",
            emotion_confidence=0.8,
            top_k_emotions={"happiness": 0.8, "neutral": 0.2},
            genuineness_state="genuine",
            genuineness_confidence=0.7,
            genuineness_flags=["symmetric_smile"],
        )
        assert face.face_id == "face-001"
        assert face.emotion == "happiness"
        assert face.genuineness_state == "genuine"
        assert "x" in face.bbox

    def test_face_result_serialisable_to_dict(self):
        face = FaceResult(
            face_id="face-001",
            bbox={"x": 10, "y": 20, "width": 80, "height": 90},
            detection_confidence=0.85,
            emotion="sadness",
            emotion_confidence=0.6,
            top_k_emotions={"sadness": 0.6},
            genuineness_state="uncertain",
            genuineness_confidence=0.4,
            genuineness_flags=[],
        )
        d = face.to_dict()
        assert isinstance(d, dict)
        assert d["emotion"] == "sadness"
        assert "bbox" in d

    def test_image_analysis_result_serialisable_to_dict(self):
        result = ImageAnalysisResult(
            request_id="req-002",
            image_width=320,
            image_height=240,
            faces=[],
        )
        d = result.to_dict()
        assert "request_id" in d
        assert "faces" in d
        assert "annotated_image" not in d  # non-serialisable field excluded

    def test_genuineness_state_is_valid_value(self):
        for state in ("genuine", "posed", "uncertain"):
            face = FaceResult(
                face_id="f",
                bbox={"x": 0, "y": 0, "width": 1, "height": 1},
                detection_confidence=0.5,
                emotion="neutral",
                emotion_confidence=0.5,
                top_k_emotions={},
                genuineness_state=state,
                genuineness_confidence=0.5,
                genuineness_flags=[],
            )
            assert face.genuineness_state == state

    def test_image_pipeline_returns_image_analysis_result(
        self, mock_emotion_service, mock_face_detector, sample_bgr_image
    ):
        """Integration-style: image pipeline must return ImageAnalysisResult."""
        from facial_emotions.pipelines.image_pipeline import ImageAnalysisPipeline

        pipeline = ImageAnalysisPipeline(
            face_detector=mock_face_detector,
            emotion_service=mock_emotion_service,
        )
        result = pipeline.analyse(sample_bgr_image, request_id="req-contract")
        assert isinstance(result, ImageAnalysisResult)
        assert result.request_id == "req-contract"

    def test_image_pipeline_face_list_is_list(
        self, mock_emotion_service, mock_face_detector, sample_bgr_image
    ):
        from facial_emotions.pipelines.image_pipeline import ImageAnalysisPipeline

        pipeline = ImageAnalysisPipeline(
            face_detector=mock_face_detector,
            emotion_service=mock_emotion_service,
        )
        result = pipeline.analyse(sample_bgr_image, request_id="req-list")
        assert isinstance(result.faces, list)

    def test_image_pipeline_face_has_correct_schema(
        self, mock_emotion_service, mock_face_detector, sample_bgr_image
    ):
        from facial_emotions.pipelines.image_pipeline import ImageAnalysisPipeline

        pipeline = ImageAnalysisPipeline(
            face_detector=mock_face_detector,
            emotion_service=mock_emotion_service,
        )
        result = pipeline.analyse(sample_bgr_image, request_id="req-schema")
        assert len(result.faces) >= 1
        face = result.faces[0]
        assert isinstance(face, FaceResult)
        assert face.emotion in [
            "anger", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"
        ]
        assert 0.0 <= face.emotion_confidence <= 1.0
        assert face.genuineness_state in ("genuine", "posed", "uncertain")
