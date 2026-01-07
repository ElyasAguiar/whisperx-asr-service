"""
GPU Memory Management Utilities
Shared across all modules to prevent VRAM buildup.
"""

import gc
import logging
from contextlib import contextmanager
from typing import Optional

import torch

logger = logging.getLogger(__name__)


def clear_gpu_memory(device: Optional[str] = None) -> None:
    """
    Clear GPU memory cache to prevent VRAM buildup.

    Args:
        device: Device string. If None or "cuda", will clear CUDA cache.
    """
    if device is None or device == "cuda" or device.startswith("cuda:"):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug("GPU memory cache cleared")


class GPUMemoryManager:
    """
    Context manager for GPU memory lifecycle.
    Automatically clears memory when exiting context.
    """

    def __init__(self, device: str = "cuda"):
        self.device = device

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        clear_gpu_memory(self.device)
        return False

    def clear(self) -> None:
        """Manually clear GPU memory."""
        clear_gpu_memory(self.device)


@contextmanager
def gpu_memory_scope(device: str = "cuda"):
    """
    Context manager that clears GPU memory on exit.

    Usage:
        with gpu_memory_scope():
            model = load_model()
            result = model.process(data)
        # GPU memory cleared here
    """
    try:
        yield
    finally:
        clear_gpu_memory(device)
