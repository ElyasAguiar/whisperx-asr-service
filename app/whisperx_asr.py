"""
WhisperX ASR Implementation
"""

import os
import tempfile
import logging
import warnings
import gc
from typing import Optional, Dict, Any
from pathlib import Path

import whisperx
from whisperx.diarize import DiarizationPipeline
import torch
import numpy as np

from .asr_interface import ASRInterface, ASRConfig, ASRResult, Segment, WordInfo

# Suppress pyannote warnings
warnings.filterwarnings("ignore", message=".*degrees of freedom is <= 0.*")

logger = logging.getLogger(__name__)


class WhisperXASR(ASRInterface):
    """WhisperX implementation of ASR interface"""
    
    def __init__(
        self,
        device: str = "cuda",
        compute_type: str = "float16",
        batch_size: int = 16,
        hf_token: Optional[str] = None,
        cache_dir: str = "/.cache"
    ):
        self.device = device
        self.compute_type = compute_type
        self.batch_size = batch_size
        self.hf_token = hf_token
        self.cache_dir = cache_dir
        self.loaded_models = {}
        
        logger.info(f"WhisperX ASR initialized - device: {device}, compute_type: {compute_type}")
    
    def _load_model(self, model_name: str):
        """Load WhisperX model with caching"""
        if model_name not in self.loaded_models:
            logger.info(f"Loading WhisperX model: {model_name}")
            model = whisperx.load_model(
                model_name,
                device=self.device,
                compute_type=self.compute_type,
                download_root=self.cache_dir
            )
            self.loaded_models[model_name] = model
            logger.info(f"Model {model_name} loaded successfully")
        
        return self.loaded_models[model_name]
    
    def _clear_gpu_memory(self):
        """Clear GPU memory cache"""
        if self.device == "cuda":
            gc.collect()
            torch.cuda.empty_cache()
    
    def transcribe(self, audio_data: bytes, config: ASRConfig) -> ASRResult:
        """
        Transcribe audio using WhisperX
        
        Args:
            audio_data: Raw audio bytes
            config: ASR configuration
            
        Returns:
            ASRResult with transcription
        """
        temp_audio_path = None
        
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
            
            # Transcribe
            logger.info("Starting transcription...")
            transcribe_options = {
                "batch_size": self.batch_size,
                "language": config.language,
                "task": config.task
            }
            
            if config.initial_prompt:
                transcribe_options["initial_prompt"] = config.initial_prompt
            
            result = model.transcribe(audio, **transcribe_options)
            detected_language = result.get("language", config.language or "en")
            logger.info(f"Transcription complete. Language: {detected_language}")
            
            # Clear GPU memory
            self._clear_gpu_memory()
            
            # Align timestamps if requested
            if config.enable_word_timestamps:
                logger.info("Aligning timestamps...")
                try:
                    model_a, metadata = whisperx.load_align_model(
                        language_code=detected_language,
                        device=self.device,
                        model_dir=self.cache_dir
                    )
                    result = whisperx.align(
                        result["segments"],
                        model_a,
                        metadata,
                        audio,
                        self.device,
                        return_char_alignments=False
                    )
                    logger.info("Alignment complete")
                    
                    del model_a
                    self._clear_gpu_memory()
                except Exception as e:
                    logger.warning(f"Alignment failed: {str(e)}")
            
            # Speaker diarization if requested
            if config.enable_diarization and self.hf_token:
                logger.info(f"Starting speaker diarization with model: {config.diarization_model}")
                try:
                    diarize_model = DiarizationPipeline(
                        model_name=config.diarization_model,
                        use_auth_token=self.hf_token,
                        device=torch.device(self.device)
                    )
                    
                    diarize_params = {}
                    if config.num_speakers is not None:
                        diarize_params["num_speakers"] = config.num_speakers
                    else:
                        if config.min_speakers is not None:
                            diarize_params["min_speakers"] = config.min_speakers
                        if config.max_speakers is not None:
                            diarize_params["max_speakers"] = config.max_speakers
                    
                    diarize_segments = diarize_model(audio, **diarize_params)
                    
                    # Use exclusive diarization if available
                    if hasattr(diarize_segments, 'exclusive_speaker_diarization'):
                        diarize_segments = diarize_segments.exclusive_speaker_diarization
                    
                    result = whisperx.assign_word_speakers(diarize_segments, result)
                    logger.info("Diarization complete")
                    
                    del diarize_model
                    self._clear_gpu_memory()
                except Exception as e:
                    logger.warning(f"Diarization failed: {str(e)}")
            
            # Convert to ASRResult format
            segments = []
            for seg in result.get("segments", []):
                words = []
                for word in seg.get("words", []):
                    words.append(WordInfo(
                        word=word.get("word", ""),
                        start=word.get("start", 0.0),
                        end=word.get("end", 0.0),
                        confidence=word.get("score", 1.0),
                        speaker=word.get("speaker", None)
                    ))
                
                segments.append(Segment(
                    text=seg.get("text", ""),
                    start=seg.get("start", 0.0),
                    end=seg.get("end", 0.0),
                    words=words,
                    speaker=seg.get("speaker", None),
                    confidence=1.0
                ))
            
            # Calculate audio duration
            duration = audio.shape[0] / config.sample_rate if len(audio.shape) > 0 else 0.0
            
            return ASRResult(
                segments=segments,
                language=detected_language,
                duration=duration
            )
            
        finally:
            # Clean up temporary file
            if temp_audio_path and os.path.exists(temp_audio_path):
                try:
                    os.unlink(temp_audio_path)
                except Exception as e:
                    logger.warning(f"Failed to delete temp file: {str(e)}")
    
    def is_available(self) -> bool:
        """Check if WhisperX is available"""
        return True
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models"""
        return {
            "type": "WhisperX",
            "device": self.device,
            "compute_type": self.compute_type,
            "loaded_models": list(self.loaded_models.keys()),
            "batch_size": self.batch_size
        }
