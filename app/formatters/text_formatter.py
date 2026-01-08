"""
Text Formatter - Formats transcription results as plain text or TSV
"""

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class TextFormatter:
    """Formatter for text and TSV output formats."""

    @staticmethod
    def format_text(result: Dict[str, Any]) -> Dict[str, str]:
        """
        Format transcription result as plain text.

        Args:
            result: Transcription result with segments

        Returns:
            Dictionary with 'text' key containing concatenated text
        """
        text = " ".join([seg.get("text", "") for seg in result.get("segments", [])])
        return {"text": text}

    @staticmethod
    def format_tsv(result: Dict[str, Any]) -> Dict[str, str]:
        """
        Format transcription result as TSV (tab-separated values).

        Args:
            result: Transcription result with segments

        Returns:
            Dictionary with 'tsv' key containing TSV content
        """
        tsv_content = ["start\tend\ttext\tspeaker"]

        for segment in result.get("segments", []):
            start = segment.get("start", 0)
            end = segment.get("end", 0)
            text = segment.get("text", "").strip()
            speaker = segment.get("speaker", "")
            tsv_content.append(f"{start}\t{end}\t{text}\t{speaker}")

        return {"tsv": "\n".join(tsv_content)}
