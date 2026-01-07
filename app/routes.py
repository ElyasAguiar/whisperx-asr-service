"""
Route handlers for WhisperX ASR Service
"""

import logging
from typing import Optional

from fastapi import File, Form, HTTPException, UploadFile

from .config import config
from .context_managers import TemporaryAudioFile
from .formatters import JSONFormatter, SubtitleFormatter, TextFormatter
from .models import get_loaded_models
from .pipelines import ASRPipeline
from .services import AudioService

logger = logging.getLogger(__name__)


async def root():
    """Health check endpoint"""
    return {
        "status": "running",
        "service": "WhisperX ASR API",
        "device": config.DEVICE,
        "compute_type": config.COMPUTE_TYPE,
    }


async def health_check():
    """Health check endpoint for monitoring"""
    return {
        "status": "healthy",
        "device": config.DEVICE,
        "loaded_models": get_loaded_models(),
    }


async def transcribe_audio(
    audio_file: UploadFile = File(...),
    task: str = Form("transcribe"),
    language: Optional[str] = Form(None),
    initial_prompt: Optional[str] = Form(None),
    word_timestamps: bool = Form(True),
    output_format: str = Form("json"),
    output: Optional[str] = Form(None),  # Legacy parameter name compatibility
    model: str = Form(config.DEFAULT_MODEL),
    num_speakers: Optional[int] = Form(None),
    min_speakers: Optional[int] = Form(None),
    max_speakers: Optional[int] = Form(None),
    diarize: Optional[bool] = Form(
        None
    ),  # Enable speaker diarization (compatible with whisper-asr-webservice)
    enable_diarization: Optional[bool] = Form(None),  # Alias for diarize (deprecated)
    return_speaker_embeddings: Optional[bool] = Form(None),
):
    """
    Main ASR endpoint compatible with openai-whisper-asr-webservice

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
        diarize: Enable speaker diarization (compatible with whisper-asr-webservice)
        enable_diarization: Alias for diarize (deprecated, use diarize instead)
        return_speaker_embeddings: Return speaker embeddings (256-dimensional vectors)
    """
    try:
        # Handle legacy parameter names
        if output is not None:
            output_format = output  # Support legacy 'output' parameter

        # Handle diarize/enable_diarization: use either param, default to True if neither specified
        if diarize is not None or enable_diarization is not None:
            should_diarize = (diarize is True) or (enable_diarization is True)
        else:
            should_diarize = True  # Default to enabled

        if return_speaker_embeddings is None:
            return_speaker_embeddings = False

        # Use context manager for temporary audio file (automatic cleanup)
        async with TemporaryAudioFile(audio_file) as audio_path:
            # Validate file size
            content = await audio_file.read()
            await audio_file.seek(0)  # Reset for potential re-reads
            file_size_mb = await AudioService.validate_file_size(
                content, audio_file.filename
            )

            logger.info(
                f"Processing audio: {audio_file.filename} ({file_size_mb:.1f}MB), "
                f"model: {model}, language: {language}"
            )

            # Initialize and run ASR pipeline
            pipeline = ASRPipeline(
                model=model,
                enable_alignment=word_timestamps,
                enable_diarization=should_diarize,
                hf_token=config.HF_TOKEN,
            )

            result, detected_language, speaker_embeddings = await pipeline.process(
                audio_path=audio_path,
                language=language,
                task=task,
                initial_prompt=initial_prompt,
                num_speakers=num_speakers,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
                return_speaker_embeddings=return_speaker_embeddings,
            )

            # Format output based on requested format
            if output_format == "json":
                return JSONFormatter.format(
                    result, detected_language, speaker_embeddings
                )
            elif output_format == "text":
                return TextFormatter.format_text(result)
            elif output_format == "srt":
                return SubtitleFormatter.format_srt(result)
            elif output_format == "vtt":
                return SubtitleFormatter.format_vtt(result)
            elif output_format == "tsv":
                return TextFormatter.format_tsv(result)
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported output format: {output_format}",
                )

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error(f"Transcription error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
