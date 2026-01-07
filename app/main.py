"""
WhisperX ASR API Service
Compatible with openai-whisper-asr-webservice API endpoints

This is a thin API layer that delegates to WhisperXASR for all processing.
"""

import logging
import os
import tempfile
import warnings
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse

from .asr_interface import ASRConfig
from .config import config
from .formatters import OutputFormatter, sanitize_float_values
from .whisperx_asr import WhisperXASR

# Suppress pyannote pooling warnings
warnings.filterwarnings("ignore", message=".*degrees of freedom is <= 0.*")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Singleton ASR service instance
_asr_service: Optional[WhisperXASR] = None


def get_asr_service() -> WhisperXASR:
    """Get or create the ASR service singleton."""
    global _asr_service
    if _asr_service is None:
        _asr_service = WhisperXASR()
    return _asr_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    Handles startup (model preloading) and shutdown (cleanup).
    """
    global _asr_service

    # Startup
    logger.info(f"WhisperX ASR Service starting on device: {config.device}")
    logger.info(f"Compute type: {config.compute_type}, Batch size: {config.batch_size}")
    logger.info(f"Default model: {config.default_model}")

    if config.default_model:
        logger.info(f"Preloading model on startup: {config.default_model}")
        try:
            asr = get_asr_service()
            asr._load_model(config.default_model)
            logger.info(f"Successfully preloaded model: {config.default_model}")
        except Exception as e:
            logger.error(f"Failed to preload model {config.default_model}: {e}")

    yield  # Application runs here

    # Shutdown
    if _asr_service:
        _asr_service.cleanup()
        _asr_service = None
    logger.info("ASR service shut down")


# Initialize FastAPI app with lifespan
app = FastAPI(
    title="WhisperX ASR API",
    description="Automatic Speech Recognition API with Speaker Diarization using WhisperX",
    version="0.2.0",
    lifespan=lifespan,
)


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "running",
        "service": "WhisperX ASR API",
        "device": config.device,
        "compute_type": config.compute_type,
    }


@app.post("/asr")
async def transcribe_audio(
    audio_file: UploadFile = File(...),
    task: str = Form("transcribe"),
    language: Optional[str] = Form(None),
    initial_prompt: Optional[str] = Form(None),
    word_timestamps: bool = Form(True),
    output_format: str = Form("json"),
    output: Optional[str] = Query(None),
    model: str = Form(None),
    num_speakers: Optional[int] = Form(None),
    min_speakers: Optional[int] = Query(None),
    max_speakers: Optional[int] = Query(None),
    diarize: Optional[bool] = Query(True),
    enable_diarization: Optional[bool] = Query(None),
    return_speaker_embeddings: Optional[bool] = Query(None),
):
    """
    Main ASR endpoint compatible with openai-whisper-asr-webservice.

    Args:
        audio_file: Audio file to transcribe
        task: transcribe or translate
        language: Language code (e.g., 'en', 'es', 'fr')
        initial_prompt: Optional prompt to guide the model
        word_timestamps: Return word-level timestamps
        output_format: json, text, srt, vtt, or tsv
        model: WhisperX model name (tiny, base, small, medium, large-v2, large-v3)
        num_speakers: Exact number of speakers (if known, overrides min/max)
        min_speakers: Minimum number of speakers for diarization
        max_speakers: Maximum number of speakers for diarization
        diarize: Enable speaker diarization
        enable_diarization: Alias for diarize (deprecated)
        return_speaker_embeddings: Return speaker embeddings
    """
    temp_audio_path = None

    try:
        # Handle legacy parameter names
        if output is not None:
            output_format = output

        # Validate output format
        if not OutputFormatter.is_supported(output_format):
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported output format: {output_format}. "
                f"Supported: {OutputFormatter.get_supported_formats()}",
            )

        # Resolve diarization flag
        if diarize is not None or enable_diarization is not None:
            should_diarize = (diarize is True) or (enable_diarization is True)

        # Save uploaded file
        file_suffix = Path(audio_file.filename or "audio.wav").suffix
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=file_suffix,
        ) as temp_file:
            temp_audio_path = temp_file.name
            content = await audio_file.read()
            temp_file.write(content)

        # Check file size
        file_size_mb = len(content) / (1024 * 1024)
        if file_size_mb > config.max_file_size_mb:
            raise HTTPException(
                status_code=413,
                detail=f"File too large ({file_size_mb:.1f}MB). "
                f"Maximum allowed: {config.max_file_size_mb}MB.",
            )

        if file_size_mb > 100:
            logger.warning(f"Processing large file ({file_size_mb:.1f}MB)")

        logger.info(
            f"Processing: {audio_file.filename} ({file_size_mb:.1f}MB), "
            f"model: {model or config.default_model}, language: {language}"
        )

        # Build ASR config
        asr_config = ASRConfig(
            language=language,
            task=task,
            model=model or config.default_model,
            initial_prompt=initial_prompt,
            enable_word_timestamps=word_timestamps,
            enable_diarization=should_diarize,
            diarization_model=config.diarization_model,
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            return_speaker_embeddings=return_speaker_embeddings or False,
        )

        # Process with ASR service
        asr = get_asr_service()
        result, speaker_embeddings = asr.transcribe(content, asr_config)

        # Build raw segments for JSON format (backward compatibility)
        raw_segments = [
            {
                "text": seg.text,
                "start": seg.start,
                "end": seg.end,
                "speaker": seg.speaker,
                "words": [
                    {
                        "word": w.word,
                        "start": w.start,
                        "end": w.end,
                        "score": w.confidence,
                        "speaker": w.speaker,
                    }
                    for w in seg.words
                ],
            }
            for seg in result.segments
        ]

        # Format output
        if output_format == "json":
            response_data = {
                "text": raw_segments,
                "language": result.language,
                "segments": raw_segments,
                "word_segments": [],
            }
            if speaker_embeddings:
                response_data["speaker_embeddings"] = sanitize_float_values(
                    speaker_embeddings
                )
                logger.info(
                    f"Including speaker embeddings: {list(speaker_embeddings.keys())}"
                )
            return JSONResponse(content=response_data)

        return OutputFormatter.format(output_format, result, raw_segments)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Transcription error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if temp_audio_path and os.path.exists(temp_audio_path):
            try:
                os.unlink(temp_audio_path)
            except Exception as e:
                logger.warning(f"Failed to delete temp file: {e}")


@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    asr = get_asr_service()
    return {
        "status": "healthy",
        "device": config.device,
        "loaded_models": asr.get_model_info().get("loaded_models", []),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=config.rest_port)
