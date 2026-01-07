"""
Diarization Service - Handles speaker diarization
"""

import logging
from typing import Any, Dict, Optional, Tuple

import torch
import whisperx
from pyannote.audio import Pipeline

from ..config import config
from ..context_managers import GPUModelContext

logger = logging.getLogger(__name__)


class DiarizationService:
    """Service for speaker diarization using pyannote.audio."""

    def __init__(self, hf_token: Optional[str] = None):
        """
        Initialize diarization service.

        Args:
            hf_token: HuggingFace token for accessing pyannote models
        """
        self.hf_token = hf_token or config.HF_TOKEN

    def diarize(
        self,
        audio,
        result: Dict[str, Any],
        num_speakers: Optional[int] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        return_embeddings: bool = False,
    ) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        """
        Perform speaker diarization on transcription result.

        Args:
            audio: Audio data (numpy array from whisperx.load_audio)
            result: Transcription result with segments
            num_speakers: Exact number of speakers (overrides min/max)
            min_speakers: Minimum number of speakers
            max_speakers: Maximum number of speakers
            return_embeddings: If True, return speaker embeddings

        Returns:
            Tuple of (updated_result, speaker_embeddings)
            - updated_result: Result dict with speaker information added to segments
            - speaker_embeddings: Dict of speaker embeddings (None if not requested)

        Raises:
            ValueError: If HF_TOKEN is not configured
            Exception: If diarization fails (caller should handle gracefully)
        """
        if not self.hf_token:
            raise ValueError("HF_TOKEN not configured for speaker diarization")

        logger.info(
            "Starting speaker diarization with pyannote speaker-diarization-3.1..."
        )

        try:
            # Load pyannote diarization pipeline
            diarize_model = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                token=self.hf_token,
            )
            diarize_model.to(torch.device(config.DEVICE))

            # Prepare diarization parameters
            diarize_params = {}
            if num_speakers is not None:
                # If exact number is provided, use it (overrides min/max)
                diarize_params["num_speakers"] = num_speakers
                logger.info(f"Diarization with exact speaker count: {num_speakers}")
            else:
                # Otherwise use min/max range
                if min_speakers is not None:
                    diarize_params["min_speakers"] = min_speakers
                if max_speakers is not None:
                    diarize_params["max_speakers"] = max_speakers
                logger.info(
                    f"Diarization with speaker range: {min_speakers}-{max_speakers}"
                )

            # Add return_embeddings parameter if requested
            if return_embeddings:
                diarize_params["return_embeddings"] = True
                logger.info("Speaker embeddings will be returned")

            # CRITICAL FIX: Convert audio to proper format for pyannote
            # pyannote expects {"waveform": tensor, "sample_rate": int}
            # whisperx.load_audio() returns numpy array
            audio_input = {
                "waveform": torch.from_numpy(audio).unsqueeze(0),
                "sample_rate": 16000,
            }

            # Run diarization with context manager for GPU cleanup
            with GPUModelContext() as ctx:
                ctx.register(diarize_model)
                diarize_output = diarize_model(audio_input, **diarize_params)

            # Check if embeddings were returned
            speaker_embeddings = None
            if return_embeddings and isinstance(diarize_output, tuple):
                diarize_segments, speaker_embeddings = diarize_output
                logger.info(
                    f"Received speaker embeddings for {len(speaker_embeddings)} speakers"
                )
            else:
                diarize_segments = diarize_output

            # Try to access exclusive_speaker_diarization (new in community-1)
            # This simplifies reconciliation with transcription timestamps
            if hasattr(diarize_segments, "exclusive_speaker_diarization"):
                diarize_segments = diarize_segments.exclusive_speaker_diarization
                logger.info(
                    "Using exclusive speaker diarization for better timestamp reconciliation"
                )

            # Assign speakers to words
            result = whisperx.assign_word_speakers(diarize_segments, result)

            # Log speakers detected in segments for debugging
            unique_speakers = set()
            for segment in result.get("segments", []):
                if "speaker" in segment:
                    unique_speakers.add(segment["speaker"])

            if unique_speakers:
                logger.info(
                    f"Speaker diarization complete: {len(unique_speakers)} unique speakers detected: {sorted(unique_speakers)}"
                )
            else:
                logger.warning(
                    "Speaker diarization completed but no speakers were assigned to segments"
                )

            return result, speaker_embeddings

        except Exception as e:
            logger.warning(
                f"Speaker diarization failed: {str(e)}, continuing without diarization"
            )
            raise
