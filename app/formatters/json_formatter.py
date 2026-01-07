"""
JSON Formatter - Formats transcription results as JSON
"""

import logging
from typing import Any, Dict, Optional

from fastapi.responses import JSONResponse

from ..utils import sanitize_float_values

logger = logging.getLogger(__name__)


class JSONFormatter:
    """Formatter for JSON output format."""

    @staticmethod
    def format(
        result: Dict[str, Any],
        language: str,
        speaker_embeddings: Optional[Dict[str, Any]] = None,
    ) -> JSONResponse:
        """
        Format transcription result as JSON response.

        Args:
            result: Transcription result with segments and word_segments
            language: Detected or specified language
            speaker_embeddings: Optional speaker embeddings (256-dimensional vectors)

        Returns:
            FastAPI JSONResponse with sanitized data
        """
        # Sanitize all data structures to ensure JSON compliance (remove NaN/Inf values)
        response_data = {
            "text": sanitize_float_values(result.get("segments", [])),
            "language": language,
            "segments": sanitize_float_values(result.get("segments", [])),
            "word_segments": sanitize_float_values(result.get("word_segments", [])),
        }

        # Add speaker embeddings if they were requested and available
        if speaker_embeddings:
            response_data["speaker_embeddings"] = sanitize_float_values(
                speaker_embeddings
            )
            logger.info(
                f"Including speaker embeddings in response: {list(speaker_embeddings.keys())}"
            )

        return JSONResponse(content=response_data)
