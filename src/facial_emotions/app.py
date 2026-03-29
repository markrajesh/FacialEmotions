"""Application entry point."""

from __future__ import annotations

import logging

from facial_emotions import config
from facial_emotions.ui.gradio_app import build_app

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("facial_emotions")


def main() -> None:
    logger.info("Starting Facial Emotions (LOCAL_ONLY=%s)", config.LOCAL_ONLY)
    logger.info(
        "Performance thresholds — image: %.1fs | webcam: %.0fms @ %.0f FPS | video: %.1f FPS",
        config.IMAGE_ANALYSIS_TIMEOUT_S,
        config.WEBCAM_FRAME_BUDGET_MS,
        config.WEBCAM_TARGET_FPS,
        config.VIDEO_SAMPLE_FPS,
    )
    app = build_app()
    app.launch(
        server_name="127.0.0.1",
        share=False,       # local-only — no tunnelling to external servers
        inbrowser=True,
    )


if __name__ == "__main__":
    main()
