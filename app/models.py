"""
Model management module for WhisperX ASR Service
"""

import gc
import logging

import torch
import whisperx
from fastapi import HTTPException

from .config import config

logger = logging.getLogger(__name__)

# Model cache
loaded_models = {}


def load_whisper_model(model_name: str):
    """
    Load WhisperX model with caching

    Args:
        model_name: Name of the WhisperX model to load

    Returns:
        Loaded WhisperX model

    Raises:
        HTTPException: If model loading fails
    """
    if model_name not in loaded_models:
        logger.info(f"Loading WhisperX model: {model_name}")
        try:
            model = whisperx.load_model(
                model_name,
                device=config.DEVICE,
                compute_type=config.COMPUTE_TYPE,
                download_root=config.CACHE_DIR,
            )
            loaded_models[model_name] = model
            logger.info(f"Model {model_name} loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {str(e)}")
            raise HTTPException(
                status_code=500, detail=f"Failed to load model: {str(e)}"
            )

    return loaded_models[model_name]


def clear_gpu_memory():
    """Clear GPU memory cache to prevent VRAM buildup"""
    if config.DEVICE == "cuda":
        gc.collect()
        torch.cuda.empty_cache()
        logger.debug("GPU memory cache cleared")


def get_loaded_models():
    """
    Get list of currently loaded model names

    Returns:
        List of model names
    """
    return list(loaded_models.keys())


def preload_model(model_name: str):
    """
    Preload a model at startup

    Args:
        model_name: Name of the model to preload
    """
    if model_name:
        logger.info(f"Preloading model on startup: {model_name}")
        try:
            load_whisper_model(model_name)
            logger.info(f"Successfully preloaded model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to preload model {model_name}: {str(e)}")
