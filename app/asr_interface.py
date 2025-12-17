"""
ASR Interface - Abstraction layer for different ASR models
This makes the service agnostic to the underlying LLM/ASR implementation
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from dataclasses import dataclass


@dataclass
class ASRConfig:
    """Configuration for ASR processing"""
    language: Optional[str] = None
    task: str = "transcribe"
    model: str = "large-v3"
    enable_diarization: bool = True
    enable_word_timestamps: bool = True
    num_speakers: Optional[int] = None
    min_speakers: Optional[int] = None
    max_speakers: Optional[int] = None
    initial_prompt: Optional[str] = None
    sample_rate: int = 16000
    audio_encoding: str = "LINEAR_PCM"
    diarization_model: str = "pyannote/speaker-diarization-community-1"


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
