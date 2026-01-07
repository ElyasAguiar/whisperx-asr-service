"""
ASR Interface - Abstraction layer for different ASR models
This makes the service agnostic to the underlying LLM/ASR implementation
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .config import config as app_config


@dataclass
class ASRConfig:
    """Configuration for ASR processing"""

    # Transcription settings
    language: Optional[str] = None
    task: str = "transcribe"
    model: str = field(default_factory=lambda: app_config.default_model)
    initial_prompt: Optional[str] = None
    enable_word_timestamps: bool = True
    sample_rate: int = 16000
    audio_encoding: str = "LINEAR_PCM"

    # Diarization settings
    enable_diarization: bool = True
    diarization_model: str = field(default_factory=lambda: app_config.diarization_model)
    num_speakers: Optional[int] = None
    min_speakers: Optional[int] = None
    max_speakers: Optional[int] = None
    return_speaker_embeddings: bool = False


@dataclass
class WordInfo:
    """Word-level information"""

    word: str
    start: float
    end: float
    confidence: float = 1.0
    speaker: Optional[str] = None


@dataclass
class Segment:
    """Segment of transcribed text"""

    text: str
    start: float
    end: float
    words: List[WordInfo]
    speaker: Optional[str] = None
    confidence: float = 1.0


@dataclass
class ASRResult:
    """Result of ASR processing"""

    segments: List[Segment]
    language: str
    duration: float


class ASRInterface(ABC):
    """Abstract interface for ASR implementations"""

    @abstractmethod
    def transcribe(self, audio_data: bytes, config: ASRConfig) -> ASRResult:
        """
        Transcribe audio data

        Args:
            audio_data: Raw audio bytes
            config: ASR configuration

        Returns:
            ASRResult with transcription
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the ASR model is available and loaded"""
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        pass
