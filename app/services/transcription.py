"""
Transcription Service - Handles WhisperX transcription
"""

import logging
from typing import Any, Dict, Optional

from ..config import config
from ..context_managers import GPUModelContext
from ..models import load_whisper_model

logger = logging.getLogger(__name__)


class TranscriptionService:
    """Service for audio transcription using WhisperX."""

    def __init__(self, model_name: str = config.DEFAULT_MODEL):
        """
        Initialize transcription service.

        Args:
            model_name: WhisperX model name (tiny, base, small, medium, large-v2, large-v3)
        """
        self.model_name = model_name

    def transcribe(
        self,
        audio,
        language: Optional[str] = None,
        task: str = "transcribe",
        initial_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Transcribe audio using WhisperX.

        Args:
            audio: Audio data (numpy array from whisperx.load_audio)
            language: Language code (e.g., 'en', 'es', 'fr'). If None, auto-detect.
            task: 'transcribe' or 'translate'
            initial_prompt: Optional prompt to guide the model

        Returns:
            Dictionary containing:
                - segments: List of transcription segments
                - language: Detected or specified language
        """
        logger.info(f"Starting transcription with model: {self.model_name}")

        # Load model (uses cache from models.py)
        whisper_model = load_whisper_model(self.model_name)

        # Prepare transcription options
        transcribe_options = {
            "batch_size": config.BATCH_SIZE,
            "language": language,
            "task": task,
        }

        if initial_prompt:
            transcribe_options["initial_prompt"] = initial_prompt

        # Transcribe
        with GPUModelContext() as ctx:
            ctx.register(whisper_model)
            result = whisper_model.transcribe(audio, **transcribe_options)

        detected_language = result.get("language", language or "en")
        logger.info(f"Transcription complete. Detected language: {detected_language}")

        return result
