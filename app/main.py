"""
WhisperX ASR API Service
Compatible with openai-whisper-asr-webservice API endpoints
"""

import logging

from fastapi import FastAPI

from . import __version__
from .config import config
from .models import preload_model
from .routes import health_check, root, transcribe_audio

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="WhisperX ASR API",
    description="Automatic Speech Recognition API with Speaker Diarization using WhisperX",
    version=__version__,
)


@app.on_event("startup")
async def startup_event():
    """Preload models on startup"""
    preload_model(config.PRELOAD_MODEL)


# Register routes
app.get("/")(root)
app.post("/asr")(transcribe_audio)
app.get("/health")(health_check)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=9000)
