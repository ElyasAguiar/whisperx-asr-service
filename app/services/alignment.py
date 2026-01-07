"""
Alignment Service - Handles word-level timestamp alignment
"""

import logging
from typing import Any, Dict, List

import whisperx

from ..config import config
from ..context_managers import GPUModelContext

logger = logging.getLogger(__name__)


class AlignmentService:
    """Service for word-level timestamp alignment using WhisperX."""

    @staticmethod
    def align(
        segments: List[Dict[str, Any]],
        audio,
        language: str,
    ) -> Dict[str, Any]:
        """
        Align transcription segments with word-level timestamps.

        Args:
            segments: Transcription segments from WhisperX
            audio: Audio data (numpy array from whisperx.load_audio)
            language: Language code for alignment model

        Returns:
            Dictionary containing aligned segments with word-level timestamps

        Raises:
            Exception: If alignment fails (caller should handle gracefully)
        """
        logger.info(f"Starting timestamp alignment for language: {language}")

        try:
            # Load alignment model
            model_a, metadata = whisperx.load_align_model(
                language_code=language,
                device=config.DEVICE,
                model_dir=config.CACHE_DIR,
            )

            # Align segments
            with GPUModelContext() as ctx:
                ctx.register(model_a)
                result = whisperx.align(
                    segments,
                    model_a,
                    metadata,
                    audio,
                    config.DEVICE,
                    return_char_alignments=False,
                )

            logger.info("Timestamp alignment complete")
            return result

        except Exception as e:
            logger.warning(
                f"Timestamp alignment failed: {str(e)}, continuing without word-level timestamps"
            )
            raise
