"""
Output Formatters
Convert ASR results to various output formats.
"""

import math
from dataclasses import asdict
from typing import Any, Dict, List, Optional

import numpy as np

from ..asr_interface import ASRResult


def format_timestamp(seconds: float) -> str:
    """Convert seconds to SRT timestamp format (HH:MM:SS,mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def sanitize_float_values(obj: Any) -> Any:
    """
    Recursively sanitize float values in nested structures for JSON compliance.
    Replaces NaN and Inf values with None, converts numpy arrays to lists.
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
    return obj


def format_as_json(
    result: ASRResult,
    raw_segments: List[Dict[str, Any]],
    speaker_embeddings: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Format result as JSON response."""
    response = {
        "text": raw_segments,
        "language": result.language,
        "segments": raw_segments,
        "word_segments": [],  # Kept for compatibility
    }

    if speaker_embeddings:
        response["speaker_embeddings"] = sanitize_float_values(speaker_embeddings)

    return response


def format_as_text(result: ASRResult) -> Dict[str, str]:
    """Format result as plain text."""
    text = " ".join([seg.text for seg in result.segments])
    return {"text": text}


def format_as_srt(result: ASRResult) -> Dict[str, str]:
    """Format result as SRT subtitles."""
    srt_content = []
    for i, segment in enumerate(result.segments, 1):
        start_time = format_timestamp(segment.start)
        end_time = format_timestamp(segment.end)
        text = segment.text.strip()

        if segment.speaker:
            text = f"[{segment.speaker}] {text}"

        srt_content.append(f"{i}\n{start_time} --> {end_time}\n{text}\n")

    return {"srt": "\n".join(srt_content)}


def format_as_vtt(result: ASRResult) -> Dict[str, str]:
    """Format result as WebVTT subtitles."""
    vtt_content = ["WEBVTT\n"]
    for segment in result.segments:
        start_time = format_timestamp(segment.start).replace(",", ".")
        end_time = format_timestamp(segment.end).replace(",", ".")
        text = segment.text.strip()

        if segment.speaker:
            text = f"[{segment.speaker}] {text}"

        vtt_content.append(f"{start_time} --> {end_time}\n{text}\n")

    return {"vtt": "\n".join(vtt_content)}


def format_as_tsv(result: ASRResult) -> Dict[str, str]:
    """Format result as TSV (tab-separated values)."""
    tsv_content = ["start\tend\ttext\tspeaker"]
    for segment in result.segments:
        speaker = segment.speaker or ""
        tsv_content.append(
            f"{segment.start}\t{segment.end}\t{segment.text.strip()}\t{speaker}"
        )

    return {"tsv": "\n".join(tsv_content)}


class OutputFormatter:
    """Factory for output formatters."""

    FORMATTERS = {
        "json": format_as_json,
        "text": format_as_text,
        "srt": format_as_srt,
        "vtt": format_as_vtt,
        "tsv": format_as_tsv,
    }

    @classmethod
    def get_supported_formats(cls) -> List[str]:
        """Get list of supported output formats."""
        return list(cls.FORMATTERS.keys())

    @classmethod
    def is_supported(cls, format_name: str) -> bool:
        """Check if format is supported."""
        return format_name.lower() in cls.FORMATTERS

    @classmethod
    def format(
        cls,
        format_name: str,
        result: ASRResult,
        raw_segments: Optional[List[Dict[str, Any]]] = None,
        speaker_embeddings: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Format ASR result to specified output format.

        Args:
            format_name: Output format name (json, text, srt, vtt, tsv).
            result: ASRResult to format.
            raw_segments: Raw segments dict (for JSON format).
            speaker_embeddings: Speaker embeddings (for JSON format).

        Returns:
            Formatted output dictionary.

        Raises:
            ValueError: If format is not supported.
        """
        format_name = format_name.lower()
        if format_name not in cls.FORMATTERS:
            raise ValueError(
                f"Unsupported format: {format_name}. "
                f"Supported: {cls.get_supported_formats()}"
            )

        formatter = cls.FORMATTERS[format_name]

        if format_name == "json":
            return formatter(result, raw_segments or [], speaker_embeddings)
        return formatter(result)
