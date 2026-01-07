"""WhisperX ASR API Service"""

__version__ = "0.2.0"

from .asr_interface import ASRConfig, ASRResult, Segment, WordInfo
from .config import config
from .services.diarization import DiarizationConfig, DiarizationService
from .whisperx_asr import WhisperXASR

__all__ = [
    "config",
    "ASRConfig",
    "ASRResult",
    "Segment",
    "WordInfo",
    "WhisperXASR",
    "DiarizationConfig",
    "DiarizationService",
]
