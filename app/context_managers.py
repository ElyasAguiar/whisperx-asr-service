"""
Context Managers for Resource Management

Provides context managers for automatic cleanup of GPU models, memory, and temporary files.
"""

import logging
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional

import torch

from .config import config

logger = logging.getLogger(__name__)


class TemporaryAudioFile:
    """
    Context manager for temporary audio files with automatic cleanup.

    Usage:
        async with TemporaryAudioFile(uploaded_file) as audio_path:
            # Use audio_path
            pass
        # File automatically deleted on exit
    """

    def __init__(self, uploaded_file, suffix: Optional[str] = None):
        """
        Args:
            uploaded_file: FastAPI UploadFile object
            suffix: File extension (e.g., '.mp3'). If None, extracted from filename.
        """
        self.uploaded_file = uploaded_file
        self.suffix = suffix or Path(uploaded_file.filename).suffix
        self.temp_path: Optional[str] = None

    async def __aenter__(self):
        """Create temporary file and save uploaded content."""
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=self.suffix)
        self.temp_path = temp_file.name

        try:
            content = await self.uploaded_file.read()
            temp_file.write(content)
            temp_file.close()
            logger.debug(f"Created temporary audio file: {self.temp_path}")
            return self.temp_path
        except Exception as e:
            temp_file.close()
            if self.temp_path and os.path.exists(self.temp_path):
                os.unlink(self.temp_path)
            raise e

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Clean up temporary file."""
        if self.temp_path and os.path.exists(self.temp_path):
            try:
                os.unlink(self.temp_path)
                logger.debug(f"Deleted temporary audio file: {self.temp_path}")
            except Exception as e:
                logger.warning(f"Failed to delete temporary file {self.temp_path}: {e}")
        return False


class GPUModelContext:
    """
    Context manager for GPU models with automatic memory cleanup.

    Ensures GPU memory is properly released even if an exception occurs.

    Usage:
        with GPUModelContext() as ctx:
            model = load_model()
            ctx.register(model)
            # Use model
            pass
        # Model deleted and GPU memory cleared automatically
    """

    def __init__(self, auto_clear: bool = True):
        """
        Args:
            auto_clear: If True, automatically clear GPU memory on exit
        """
        self.auto_clear = auto_clear
        self.models: list[Any] = []

    def __enter__(self):
        """Enter context."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up models and GPU memory."""
        # Delete all registered models
        for model in self.models:
            try:
                del model
            except Exception as e:
                logger.warning(f"Failed to delete model: {e}")

        self.models.clear()

        # Clear GPU memory if enabled
        if self.auto_clear and config.DEVICE != "cpu":
            try:
                clear_gpu_memory()
                logger.debug("GPU memory cleared")
            except Exception as e:
                logger.warning(f"Failed to clear GPU memory: {e}")

        return False

    def register(self, model: Any):
        """
        Register a model for automatic cleanup.

        Args:
            model: Model to register (e.g., WhisperX model, alignment model, diarization pipeline)
        """
        self.models.append(model)
        return model


@contextmanager
def gpu_memory_guard():
    """
    Context manager that guarantees GPU memory cleanup.

    Use for operations that don't need to keep models but must ensure cleanup.

    Usage:
        with gpu_memory_guard():
            # Perform GPU operations
            result = model.process(data)
        # GPU memory guaranteed to be cleared here
    """
    try:
        yield
    finally:
        if config.DEVICE != "cpu":
            try:
                clear_gpu_memory()
            except Exception as e:
                logger.warning(f"Failed to clear GPU memory: {e}")


def clear_gpu_memory():
    """
    Clear GPU memory cache.
    Should be called after each major processing step to prevent OOM errors.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
