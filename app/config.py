"""
Configuration module for WhisperX ASR Service
"""

import logging
import os

import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class Config:
    """Application configuration from environment variables"""

    def __init__(self):
        # Device configuration
        self.DEVICE = os.getenv(
            "DEVICE", "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.COMPUTE_TYPE = os.getenv(
            "COMPUTE_TYPE", "float16" if self.DEVICE == "cuda" else "int8"
        )
        self.BATCH_SIZE = int(os.getenv("BATCH_SIZE", "16"))

        # Model configuration
        self.DEFAULT_MODEL = os.getenv("PRELOAD_MODEL", "large-v3")
        self.PRELOAD_MODEL = os.getenv("PRELOAD_MODEL", None)

        # Hugging Face token for diarization
        self.HF_TOKEN = os.getenv("HF_TOKEN", None)

        # Cache directory
        self.CACHE_DIR = os.getenv("CACHE_DIR", "/.cache")

        # File size limits
        self.MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "1000"))

        self.DIARIZATION_MODEL = os.getenv(
            "DIARIZATION_MODEL", "pyannote/speaker-diarization-3.1"
        )

        # Log configuration
        logger.info(f"WhisperX ASR Service initialized on device: {self.DEVICE}")
        logger.info(f"Compute type: {self.COMPUTE_TYPE}, Batch size: {self.BATCH_SIZE}")
        logger.info(f"Default model: {self.DEFAULT_MODEL}")


# Global configuration instance
config = Config()
