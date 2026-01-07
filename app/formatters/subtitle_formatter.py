"""
Subtitle Formatter - Formats transcription results as SRT or VTT
"""

import logging
from typing import Any, Dict, List

from ..utils import format_timestamp

logger = logging.getLogger(__name__)


class SubtitleFormatter:
    """Formatter for subtitle formats (SRT and VTT)."""

    @staticmethod
    def format_srt(result: Dict[str, Any]) -> Dict[str, str]:
        """
        Format transcription result as SRT subtitles.

        Args:
            result: Transcription result with segments

        Returns:
            Dictionary with 'srt' key containing subtitle content
        """
        srt_content = []

        for i, segment in enumerate(result.get("segments", []), 1):
            start_time = format_timestamp(segment.get("start", 0))
            end_time = format_timestamp(segment.get("end", 0))
            text = segment.get("text", "").strip()
            speaker = segment.get("speaker", "")

            # Prepend speaker label if available
            if speaker:
                text = f"[{speaker}] {text}"

            srt_content.append(f"{i}\n{start_time} --> {end_time}\n{text}\n")

        return {"srt": "\n".join(srt_content)}

    @staticmethod
    def format_vtt(result: Dict[str, Any]) -> Dict[str, str]:
        """
        Format transcription result as VTT (WebVTT) subtitles.

        Args:
            result: Transcription result with segments

        Returns:
            Dictionary with 'vtt' key containing subtitle content
        """
        vtt_content = ["WEBVTT\n"]

        for segment in result.get("segments", []):
            # VTT uses dots instead of commas for milliseconds
            start_time = format_timestamp(segment.get("start", 0)).replace(",", ".")
            end_time = format_timestamp(segment.get("end", 0)).replace(",", ".")
            text = segment.get("text", "").strip()
            speaker = segment.get("speaker", "")

            # Prepend speaker label if available
            if speaker:
                text = f"[{speaker}] {text}"

            vtt_content.append(f"{start_time} --> {end_time}\n{text}\n")

        return {"vtt": "\n".join(vtt_content)}
