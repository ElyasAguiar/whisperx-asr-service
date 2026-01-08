"""
ASR Pipeline Orchestrator

Coordinates the full ASR pipeline: transcription, alignment, and diarization.
"""

import logging
from typing import Any, Dict, Optional, Tuple

from ..services import (
    AlignmentService,
    AudioService,
    DiarizationService,
    TranscriptionService,
)

logger = logging.getLogger(__name__)


class ASRPipeline:
    """
    Orchestrates the complete ASR pipeline.

    Coordinates transcription, alignment, and diarization services
    with proper error handling and resource management.
    """

    def __init__(
        self,
        model: str,
        enable_alignment: bool = True,
        enable_diarization: bool = True,
        hf_token: Optional[str] = None,
    ):
        """
        Initialize ASR pipeline.

        Args:
            model: WhisperX model name (tiny, base, small, medium, large-v2, large-v3)
            enable_alignment: Enable word-level timestamp alignment
            enable_diarization: Enable speaker diarization
            hf_token: HuggingFace token for diarization models
        """
        self.model = model
        self.enable_alignment = enable_alignment
        self.enable_diarization = enable_diarization

        # Initialize services
        self.transcription_service = TranscriptionService(model_name=model)
        self.alignment_service = AlignmentService()
        self.diarization_service = DiarizationService(hf_token=hf_token)

    async def process(
        self,
        audio_path: str,
        language: Optional[str] = None,
        task: str = "transcribe",
        initial_prompt: Optional[str] = None,
        num_speakers: Optional[int] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        return_speaker_embeddings: bool = False,
    ) -> Tuple[Dict[str, Any], str, Optional[Dict[str, Any]]]:
        """
        Process audio through the complete ASR pipeline.

        Args:
            audio_path: Path to audio file
            language: Language code (e.g., 'en', 'es', 'fr'). If None, auto-detect.
            task: 'transcribe' or 'translate'
            initial_prompt: Optional prompt to guide the model
            num_speakers: Exact number of speakers (overrides min/max)
            min_speakers: Minimum number of speakers for diarization
            max_speakers: Maximum number of speakers for diarization
            return_speaker_embeddings: Return speaker embeddings

        Returns:
            Tuple of (result, detected_language, speaker_embeddings)
            - result: Dictionary with segments and word_segments
            - detected_language: Detected or specified language
            - speaker_embeddings: Speaker embeddings (None if not requested)
        """
        logger.info(f"Starting ASR pipeline: model={self.model}, language={language}")

        # Load audio
        audio = AudioService.load_audio(audio_path)

        # Step 1: Transcription
        result = self.transcription_service.transcribe(
            audio=audio,
            language=language,
            task=task,
            initial_prompt=initial_prompt,
        )

        detected_language = result.get("language", language or "en")

        # Step 2: Alignment (word-level timestamps)
        if self.enable_alignment:
            try:
                result = self.alignment_service.align(
                    segments=result["segments"],
                    audio=audio,
                    language=detected_language,
                )
            except Exception as e:
                logger.warning(f"Alignment failed, continuing without it: {e}")
                # Continue with original result

        # Step 3: Speaker Diarization
        speaker_embeddings = None
        if self.enable_diarization:
            try:
                result, speaker_embeddings = self.diarization_service.diarize(
                    audio=audio,
                    result=result,
                    num_speakers=num_speakers,
                    min_speakers=min_speakers,
                    max_speakers=max_speakers,
                    return_embeddings=return_speaker_embeddings,
                )
            except ValueError as e:
                # HF_TOKEN not configured
                logger.warning(f"Diarization not available: {e}")
            except Exception as e:
                logger.warning(f"Diarization failed, continuing without it: {e}")

        logger.info("ASR pipeline complete")
        return result, detected_language, speaker_embeddings
