"""
Unified Diarization Service
Single implementation used by both REST and gRPC APIs.
"""

import logging
import warnings
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Dict, Optional

import torch

# Suppress deprecation warnings from torchaudio/pyannote
warnings.filterwarnings("ignore", message=".*torchaudio._backend.*")
warnings.filterwarnings("ignore", message=".*In 2.9, this function.*")

# PyTorch 2.6+ Compatibility Layer
# ------------------------------------------------------------------------------
# PyTorch 2.6+ changed the default of `weights_only` from False to True for
# security reasons. However, pyannote.audio models contain TorchVersion objects
# that are not in the default allowlist, causing loading to fail.
#
# This patch ensures compatibility with both old and new PyTorch versions while
# maintaining security for trusted model sources (HuggingFace official models).
# ------------------------------------------------------------------------------

_TORCH_LOAD_PATCHED = False


def _patch_torch_load_for_pyannote() -> None:
    """
    Patch torch.load and lightning_fabric to allow loading pyannote models.

    This is required for PyTorch 2.6+ compatibility where weights_only=True
    became the default. Pyannote models are trusted (from HuggingFace) and
    contain objects that need to be unpickled.

    Thread-safe: Only patches once using a module-level flag.
    """
    global _TORCH_LOAD_PATCHED

    if _TORCH_LOAD_PATCHED:
        return

    logger = logging.getLogger(__name__)

    try:
        # Patch 1: torch.load
        _original_torch_load = torch.load

        @wraps(_original_torch_load)
        def _safe_torch_load(*args, **kwargs):
            # Force weights_only=False for trusted HuggingFace models
            kwargs["weights_only"] = False
            return _original_torch_load(*args, **kwargs)

        torch.load = _safe_torch_load
        logger.debug("Patched torch.load for PyTorch 2.6+ compatibility")

        # Patch 2: lightning_fabric (used by pyannote internally)
        try:
            from lightning_fabric.utilities import cloud_io

            _original_pl_load = cloud_io._load

            @wraps(_original_pl_load)
            def _safe_pl_load(path_or_url, map_location=None, weights_only=None):
                # Ignore weights_only and force False
                return _original_pl_load(
                    path_or_url, map_location=map_location, weights_only=False
                )

            cloud_io._load = _safe_pl_load
            logger.debug(
                "Patched lightning_fabric._load for PyTorch 2.6+ compatibility"
            )

        except ImportError:
            # lightning_fabric not installed or different version
            pass

        _TORCH_LOAD_PATCHED = True

    except Exception as e:
        logger.warning(f"Failed to patch torch.load for PyTorch 2.6+: {e}")


# Apply patches on module import
_patch_torch_load_for_pyannote()

# Now safe to import pyannote
import whisperx
from pyannote.audio import Pipeline

from ..config import config as app_config
from ..utils.gpu import clear_gpu_memory

logger = logging.getLogger(__name__)


@dataclass
class DiarizationConfig:
    """Configuration for speaker diarization."""

    enabled: bool = True
    model: str = field(default_factory=lambda: app_config.diarization_model)
    num_speakers: Optional[int] = None
    min_speakers: Optional[int] = None
    max_speakers: Optional[int] = None
    return_embeddings: bool = False


@dataclass
class DiarizationResult:
    """Result from diarization processing."""

    segments: Any  # pyannote Annotation or similar
    embeddings: Optional[Dict[str, Any]] = None


class DiarizationService:
    """
    Unified speaker diarization service using pyannote.audio Pipeline.

    This service is the single source of truth for diarization across
    both REST and gRPC APIs, eliminating code duplication.
    """

    def __init__(
        self,
        device: Optional[str] = None,
        hf_token: Optional[str] = None,
        default_model: Optional[str] = None,
    ):
        """
        Initialize diarization service.

        Args:
            device: Device to run on (cuda/cpu). Defaults to app config.
            hf_token: HuggingFace token for gated models. Defaults to app config.
            default_model: Default diarization model. Defaults to app config.
        """
        self.device = device or app_config.device
        self.hf_token = hf_token or app_config.hf_token
        self.default_model = default_model or app_config.diarization_model
        self._cached_pipeline: Optional[Pipeline] = None
        self._cached_model_name: Optional[str] = None

        logger.info(
            f"DiarizationService initialized - device: {self.device}, "
            f"model: {self.default_model}, token_set: {bool(self.hf_token)}"
        )

    def _load_pipeline(self, model_name: str) -> Pipeline:
        """
        Load diarization pipeline with caching.

        Args:
            model_name: Name of the pyannote model to load.

        Returns:
            Loaded Pipeline instance.

        Raises:
            ValueError: If no HF token is available for gated models.
            RuntimeError: If pipeline loading fails.
        """
        # Return cached pipeline if same model
        if self._cached_pipeline and self._cached_model_name == model_name:
            return self._cached_pipeline

        if not self.hf_token:
            raise ValueError(
                "HuggingFace token required for diarization. "
                "Set HF_TOKEN environment variable and accept model terms at "
                f"https://hf.co/{model_name}"
            )

        logger.info(f"Loading diarization pipeline: {model_name}")

        try:
            pipeline = Pipeline.from_pretrained(
                model_name,
                token=self.hf_token,
            )
            pipeline.to(torch.device(self.device))

            # Cache for reuse
            self._cached_pipeline = pipeline
            self._cached_model_name = model_name

            logger.info(f"Diarization pipeline {model_name} loaded successfully")
            return pipeline

        except Exception as e:
            logger.error(f"Failed to load diarization pipeline: {e}")
            raise RuntimeError(f"Failed to load diarization pipeline: {e}") from e

    def diarize(
        self,
        audio: Any,
        config: Optional[DiarizationConfig] = None,
    ) -> DiarizationResult:
        """
        Perform speaker diarization on audio.

        Args:
            audio: Audio data (numpy array from whisperx.load_audio).
            config: Diarization configuration. Uses defaults if not provided.

        Returns:
            DiarizationResult with segments and optional embeddings.
        """
        config = config or DiarizationConfig()

        if not config.enabled:
            logger.debug("Diarization disabled, skipping")
            return DiarizationResult(segments=None)

        model_name = config.model or self.default_model
        logger.info(f"Starting diarization with model: {model_name}")

        pipeline = self._load_pipeline(model_name)

        # Build diarization parameters
        diarize_params = {}
        if config.num_speakers is not None:
            diarize_params["num_speakers"] = config.num_speakers
            logger.info(f"Diarization with exact speaker count: {config.num_speakers}")
        else:
            if config.min_speakers is not None:
                diarize_params["min_speakers"] = config.min_speakers
            if config.max_speakers is not None:
                diarize_params["max_speakers"] = config.max_speakers
            logger.info(
                f"Diarization with speaker range: "
                f"{config.min_speakers}-{config.max_speakers}"
            )

        if config.return_embeddings:
            diarize_params["return_embeddings"] = True
            logger.info("Speaker embeddings will be returned")

        # Run diarization
        try:
            output = pipeline(audio, **diarize_params)

            # Handle embeddings if returned
            embeddings = None
            if config.return_embeddings and isinstance(output, tuple):
                segments, embeddings = output
                logger.info(
                    f"Received speaker embeddings for {len(embeddings)} speakers"
                )
            else:
                segments = output

            # Use exclusive diarization if available (better timestamp reconciliation)
            if hasattr(segments, "exclusive_speaker_diarization"):
                segments = segments.exclusive_speaker_diarization
                logger.debug("Using exclusive speaker diarization")

            logger.info("Diarization complete")
            return DiarizationResult(segments=segments, embeddings=embeddings)

        except Exception as e:
            logger.error(f"Diarization failed: {e}")
            raise

    def assign_speakers(
        self,
        diarization_result: DiarizationResult,
        transcription_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Assign speaker labels to transcription segments.

        Args:
            diarization_result: Result from diarize().
            transcription_result: WhisperX transcription result dict.

        Returns:
            Updated transcription result with speaker labels.
        """
        if diarization_result.segments is None:
            return transcription_result

        return whisperx.assign_word_speakers(
            diarization_result.segments,
            transcription_result,
        )

    def cleanup(self) -> None:
        """Release resources and clear GPU memory."""
        if self._cached_pipeline:
            del self._cached_pipeline
            self._cached_pipeline = None
            self._cached_model_name = None
        clear_gpu_memory(self.device)
        logger.debug("DiarizationService resources cleaned up")
