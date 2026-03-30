"""Gradio application — image, video, and webcam analysis UI."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Optional, Tuple

import cv2
import gradio as gr
import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Lazy-loaded pipeline singletons
# ---------------------------------------------------------------------------
_image_pipeline = None
_video_pipeline = None
_webcam_pipeline = None


def _get_image_pipeline():
    global _image_pipeline
    if _image_pipeline is None:
        from facial_emotions.pipelines.image_pipeline import ImageAnalysisPipeline
        _image_pipeline = ImageAnalysisPipeline()
    return _image_pipeline


def _get_video_pipeline():
    global _video_pipeline
    if _video_pipeline is None:
        from facial_emotions.pipelines.video_pipeline import VideoAnalysisPipeline
        _video_pipeline = VideoAnalysisPipeline()
    return _video_pipeline


def _get_webcam_pipeline():
    global _webcam_pipeline
    if _webcam_pipeline is None:
        from facial_emotions.pipelines.webcam_pipeline import WebcamAnalysisPipeline
        _webcam_pipeline = WebcamAnalysisPipeline()
    return _webcam_pipeline


# ---------------------------------------------------------------------------
# Image analysis handler
# ---------------------------------------------------------------------------

def analyse_image(pil_image: Optional[Image.Image]) -> Tuple[Optional[Image.Image], str]:
    if pil_image is None:
        return None, "No image provided."
    from facial_emotions.services.media_io import validate_image_dimensions
    from facial_emotions.services.media_io import MediaIOError
    from facial_emotions.services.media_io import downscale_to_processing_ceiling

    bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    try:
        validate_image_dimensions(bgr)
    except MediaIOError as e:
        return pil_image, f"Input validation error: {e}"

    orig_h, orig_w = bgr.shape[:2]
    frame, scale_factor = downscale_to_processing_ceiling(bgr)
    frame_h, frame_w = frame.shape[:2]

    notice = (
        f"> \u2139\ufe0f Large image auto-resized for processing "
        f"({orig_w}\u00d7{orig_h} \u2192 {frame_w}\u00d7{frame_h}).\n\n"
        if scale_factor < 1.0
        else ""
    )

    pipeline = _get_image_pipeline()
    result = pipeline.analyse(
        frame,
        original_image=bgr if scale_factor < 1.0 else None,
    )

    if not result.faces:
        return pil_image, notice + "No faces detected in the image."

    annotated_rgb = None
    if result.annotated_image is not None:
        annotated_rgb = Image.fromarray(cv2.cvtColor(result.annotated_image, cv2.COLOR_BGR2RGB))

    summary_lines = [f"Detected **{len(result.faces)}** face(s):\n"]
    for i, face in enumerate(result.faces, 1):
        conf = f"{face.emotion_confidence:.0%}"
        flags = ", ".join(face.genuineness_flags) or "—"
        summary_lines.append(
            f"**Face {i}** — {face.emotion} ({conf})  \n"
            f"  Genuineness: **{face.genuineness_state}** "
            f"({face.genuineness_confidence:.0%} confidence)  \n"
            f"  Heuristics: {flags}"
        )

    return annotated_rgb or pil_image, notice + "\n\n".join(summary_lines)


# ---------------------------------------------------------------------------
# Video analysis handler
# ---------------------------------------------------------------------------

def analyse_video(video_path: Optional[str]) -> str:
    if video_path is None:
        return "No video provided."
    try:
        pipeline = _get_video_pipeline()
        result = pipeline.analyse(video_path)
    except Exception as exc:
        return f"Error analysing video: {exc}"

    if not result.sampled_frames:
        return "No frames with detected faces found in the video."

    lines = [
        f"Video analysed: **{result.duration_seconds:.1f}s** | "
        f"**{len(result.sampled_frames)}** sampled frames\n"
    ]
    for frame in result.sampled_frames[:20]:  # limit display
        ts = frame.timestamp_ms / 1000
        face_summaries = []
        for face in frame.faces:
            face_summaries.append(
                f"{face.emotion} ({face.emotion_confidence:.0%}, {face.genuineness_state})"
            )
        lines.append(
            f"**{ts:.2f}s** — {', '.join(face_summaries) or 'no faces'}"
        )
    if len(result.sampled_frames) > 20:
        lines.append(f"... and {len(result.sampled_frames) - 20} more frames")
    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# Webcam handler
# ---------------------------------------------------------------------------

def analyse_webcam_frame(frame: Optional[np.ndarray]) -> Tuple[Optional[np.ndarray], str]:
    if frame is None:
        return None, "Webcam not active."
    pipeline = _get_webcam_pipeline()
    result = pipeline.analyse_frame(frame)
    annotated = result.annotated_frame if result.annotated_frame is not None else frame
    if not result.faces:
        return annotated, "No faces detected."
    summaries = [
        f"{f.emotion} ({f.emotion_confidence:.0%}, {f.genuineness_state})"
        for f in result.faces
    ]
    return annotated, "  |  ".join(summaries)


# ---------------------------------------------------------------------------
# App builder
# ---------------------------------------------------------------------------

def build_app() -> gr.Blocks:
    with gr.Blocks(title="Facial Emotion Detection") as app:
        gr.Markdown("# Facial Emotion Detection\n*Local, privacy-preserving analysis.*")

        with gr.Tab("Image"):
            with gr.Row():
                img_input = gr.Image(type="pil", label="Upload Image")
                img_output = gr.Image(type="pil", label="Annotated Result")
            img_summary = gr.Markdown()
            img_btn = gr.Button("Analyse Image")
            img_btn.click(
                analyse_image,
                inputs=[img_input],
                outputs=[img_output, img_summary],
            )

        with gr.Tab("Video"):
            vid_input = gr.Video(label="Upload Video")
            vid_summary = gr.Markdown()
            vid_btn = gr.Button("Analyse Video")
            vid_btn.click(
                analyse_video,
                inputs=[vid_input],
                outputs=[vid_summary],
            )

        with gr.Tab("Webcam"):
            with gr.Row():
                cam_input = gr.Image(sources=["webcam"], streaming=True, label="Webcam")
                cam_output = gr.Image(label="Annotated Frame")
            cam_label = gr.Markdown()
            cam_input.stream(
                analyse_webcam_frame,
                inputs=[cam_input],
                outputs=[cam_output, cam_label],
            )

    return app
