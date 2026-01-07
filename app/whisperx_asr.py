"""
WhisperX ASR Implementation
Unified implementation used by both REST and gRPC APIs.
"""

import logging
import os
import tempfile
import warnings
from typing import Any, Dict, Optional, Tuple

import whisperx

from .asr_interface import ASRConfig, ASRInterface, ASRResult, Segment, WordInfo
from .config import config as app_config
from .services.diarization import DiarizationConfig, DiarizationService
from .utils.gpu import clear_gpu_memory

# Suppress pyannote warnings
warnings.filterwarnings("ignore", message=".*degrees of freedom is <= 0.*")

logger = logging.getLogger(__name__)


class WhisperXASR(ASRInterface):
    """
    WhisperX implementation of ASR interface.

    This is the single source of truth for transcription logic,
    used by both REST and gRPC APIs.
    """

    def __init__(
        self,
        device: Optional[str] = None,
        compute_type: Optional[str] = None,
        batch_size: Optional[int] = None,
        hf_token: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize WhisperX ASR.

        Args:
            device: Device to run on. Defaults to app config.
            compute_type: Compute type (float16/int8). Defaults to app config.
            batch_size: Batch size for transcription. Defaults to app config.
            hf_token: HuggingFace token. Defaults to app config.
            cache_dir: Model cache directory. Defaults to app config.
        """
        self.device = device or app_config.device
        self.compute_type = compute_type or app_config.compute_type
        self.batch_size = batch_size or app_config.batch_size
        self.hf_token = hf_token or app_config.hf_token
        self.cache_dir = cache_dir or app_config.cache_dir

        # Model cache
        self._loaded_models: Dict[str, Any] = {}

        # Lazy-initialized diarization service
        self._diarization_service: Optional[DiarizationService] = None

        logger.info(
            f"WhisperX ASR initialized - device: {self.device}, "
            f"compute_type: {self.compute_type}, batch_size: {self.batch_size}"
        )

    @property
    def diarization_service(self) -> DiarizationService:
        """Lazy-load diarization service."""
        if self._diarization_service is None:
            self._diarization_service = DiarizationService(
                device=self.device,
                hf_token=self.hf_token,
            )
        return self._diarization_service

    def _load_model(self, model_name: str):
        """Load WhisperX model with caching."""
        if model_name not in self._loaded_models:
            logger.info(f"Loading WhisperX model: {model_name}")
            model = whisperx.load_model(
                model_name,
                device=self.device,
                compute_type=self.compute_type,
                download_root=self.cache_dir,
            )
            self._loaded_models[model_name] = model
            logger.info(f"Model {model_name} loaded successfully")

        return self._loaded_models[model_name]

    def transcribe(
        self,
        audio_data: bytes,
        config: ASRConfig,
    ) -> Tuple[ASRResult, Optional[Dict[str, Any]]]:
        """
        Transcribe audio using WhisperX.

        Args:
            audio_data: Raw audio bytes
            config: ASR configuration

        Returns:
            Tuple of (ASRResult, speaker_embeddings or None)
        """
        temp_audio_path = None
        speaker_embeddings = None

        try:
            # Save audio to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                temp_audio_path = temp_file.name
                temp_file.write(audio_data)

            # Load model
            model = self._load_model(config.model)

            # Load audio
            logger.info("Loading audio for transcription...")
            audio = whisperx.load_audio(temp_audio_path)

            # Step 1: Transcribe
            logger.info("Starting transcription...")
            transcribe_options = {
                "batch_size": self.batch_size,
                "language": config.language,
                "task": config.task,
            }

            if config.initial_prompt:
                transcribe_options["initial_prompt"] = config.initial_prompt

            result = model.transcribe(audio, **transcribe_options)
            detected_language = result.get("language", config.language or "en")
            logger.info(f"Transcription complete. Language: {detected_language}")

            clear_gpu_memory(self.device)

            # Step 2: Align timestamps if requested
            if config.enable_word_timestamps:
                result = self._align_timestamps(audio, result, detected_language)

            # Step 3: Speaker diarization if requested
            if config.enable_diarization:
                result, speaker_embeddings = self._perform_diarization(
                    audio, result, config
                )

            # Convert to ASRResult format
            asr_result = self._build_asr_result(
                result, detected_language, audio, config
            )

            return asr_result, speaker_embeddings

        finally:
            # Clean up temporary file
            if temp_audio_path and os.path.exists(temp_audio_path):
                try:
                    os.unlink(temp_audio_path)
                except Exception as e:
                    logger.warning(f"Failed to delete temp file: {e}")

    def _align_timestamps(
        self,
        audio,
        result: Dict[str, Any],
        language: str,
    ) -> Dict[str, Any]:
        """Align whisper output with word-level timestamps."""
        logger.info("Aligning timestamps...")
        try:
            model_a, metadata = whisperx.load_align_model(
                language_code=language,
                device=self.device,
                model_dir=self.cache_dir,
            )
            result = whisperx.align(
                result["segments"],
                model_a,
                metadata,
                audio,
                self.device,
                return_char_alignments=False,
            )
            logger.info("Alignment complete")

            del model_a
            clear_gpu_memory(self.device)

        except Exception as e:
            logger.warning(f"Alignment failed: {e}")

        return result

    def _perform_diarization(
        self,
        audio,
        result: Dict[str, Any],
        config: ASRConfig,
    ) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        """Perform speaker diarization using unified DiarizationService."""
        speaker_embeddings = None

        if not self.hf_token:
            logger.warning(
                "Diarization requested but no HF token available. "
                "Set HF_TOKEN environment variable."
            )
            return result, None

        try:
            diarize_config = DiarizationConfig(
                enabled=True,
                model=config.diarization_model,
                num_speakers=config.num_speakers,
                min_speakers=config.min_speakers,
                max_speakers=config.max_speakers,
                return_embeddings=getattr(config, "return_speaker_embeddings", False),
            )

            diarize_result = self.diarization_service.diarize(audio, diarize_config)

            if diarize_result.segments is not None:
                result = self.diarization_service.assign_speakers(
                    diarize_result, result
                )
                speaker_embeddings = diarize_result.embeddings

            clear_gpu_memory(self.device)

        except Exception as e:
            logger.warning(f"Diarization failed: {e}")

        return result, speaker_embeddings

    def _build_asr_result(
        self,
        result: Dict[str, Any],
        language: str,
        audio,
        config: ASRConfig,
    ) -> ASRResult:
        """Convert WhisperX result dict to ASRResult dataclass."""
        segments = []
        for seg in result.get("segments", []):
            words = []
            for word in seg.get("words", []):
                words.append(
                    WordInfo(
                        word=word.get("word", ""),
                        start=word.get("start", 0.0),
                        end=word.get("end", 0.0),
                        confidence=word.get("score", 1.0),
                        speaker=word.get("speaker"),
                    )
                )

            segments.append(
                Segment(
                    text=seg.get("text", ""),
                    start=seg.get("start", 0.0),
                    end=seg.get("end", 0.0),
                    words=words,
                    speaker=seg.get("speaker"),
                    confidence=1.0,
                )
            )

        # Calculate audio duration
        duration = audio.shape[0] / config.sample_rate if len(audio.shape) > 0 else 0.0

        return ASRResult(
            segments=segments,
            language=language,
            duration=duration,
        )

    def transcribe_file(
        self,
        file_path: str,
        config: ASRConfig,
    ) -> Tuple[ASRResult, Optional[Dict[str, Any]]]:
        """
        Transcribe audio from file path.

        Args:
            file_path: Path to audio file.
            config: ASR configuration.

        Returns:
            Tuple of (ASRResult, speaker_embeddings or None)
        """
        with open(file_path, "rb") as f:
            audio_data = f.read()
        return self.transcribe(audio_data, config)

    def is_available(self) -> bool:
        """Check if WhisperX is available."""
        return True

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models."""
        return {
            "type": "WhisperX",
            "device": self.device,
            "compute_type": self.compute_type,
            "loaded_models": list(self._loaded_models.keys()),
            "batch_size": self.batch_size,
            "diarization_model": app_config.diarization_model,
        }

    def cleanup(self) -> None:
        """Release all resources."""
        self._loaded_models.clear()
        if self._diarization_service:
            self._diarization_service.cleanup()
        clear_gpu_memory(self.device)
        logger.info("WhisperXASR resources cleaned up")
