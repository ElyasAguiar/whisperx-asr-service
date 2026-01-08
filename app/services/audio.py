"""
Audio Service - Handles audio file management and loading
"""

import logging
from pathlib import Path
from typing import Tuple

import whisperx
from fastapi import HTTPException, UploadFile

from ..config import config

logger = logging.getLogger(__name__)


class AudioService:
    """Service for audio file operations."""

    @staticmethod
    async def validate_file_size(content: bytes, filename: str) -> float:
        """
        Validate audio file size against configured limits.

        Args:
            content: File content bytes
            filename: Original filename

        Returns:
            File size in MB

        Raises:
            HTTPException: If file exceeds maximum size
        """
        file_size_mb = len(content) / (1024 * 1024)

        if file_size_mb > config.MAX_FILE_SIZE_MB:
            raise HTTPException(
                status_code=413,
                detail=f"File too large ({file_size_mb:.1f}MB). Maximum allowed: {config.MAX_FILE_SIZE_MB}MB. "
                f"Large files may cause out-of-memory errors.",
            )

        if file_size_mb > 100:
            logger.warning(
                f"Processing large file ({file_size_mb:.1f}MB) - may consume significant VRAM"
            )

        logger.info(f"Audio file validated: {filename} ({file_size_mb:.1f}MB)")
        return file_size_mb

    @staticmethod
    def load_audio(audio_path: str):
        """
        Load audio file using WhisperX.

        Args:
            audio_path: Path to audio file

        Returns:
            Numpy array containing audio data (16kHz mono)
        """
        logger.debug(f"Loading audio from: {audio_path}")
        audio = whisperx.load_audio(audio_path)
        logger.debug(f"Audio loaded successfully, shape: {audio.shape}")
        return audio

    @staticmethod
    def get_file_extension(filename: str) -> str:
        """
        Extract file extension from filename.

        Args:
            filename: Original filename

        Returns:
            File extension (e.g., '.mp3')
        """
        return Path(filename).suffix
