"""
Utility functions for WhisperX ASR Service
"""

import math

import numpy as np


def format_timestamp(seconds: float) -> str:
    """
    Convert seconds to SRT timestamp format

    Args:
        seconds: Time in seconds

    Returns:
        Formatted timestamp string (HH:MM:SS,mmm)
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def sanitize_float_values(obj):
    """
    Recursively sanitize float values in nested structures to ensure JSON compliance.
    Replaces NaN and Inf values with None, and converts numpy arrays to lists.

    Args:
        obj: Object to sanitize (dict, list, array, or scalar)

    Returns:
        Sanitized object
    """
    if isinstance(obj, dict):
        return {key: sanitize_float_values(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [sanitize_float_values(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return sanitize_float_values(obj.tolist())
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    elif isinstance(obj, (np.floating, np.integer)):
        value = float(obj)
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    else:
        return obj
