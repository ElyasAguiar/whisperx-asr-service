"""
Centralized Configuration - Single Source of Truth
All environment variables and defaults in one place.
"""

import os
from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass(frozen=True)
class AppConfig:
    """Immutable application configuration loaded from environment."""

    # Device configuration
    device: str = field(
        default_factory=lambda: os.getenv(
            "DEVICE", "cuda" if torch.cuda.is_available() else "cpu"
        )
    )
    compute_type: str = field(
        default_factory=lambda: os.getenv(
            "COMPUTE_TYPE",
            "float16" if torch.cuda.is_available() else "int8",
        )
    )

    # Model configuration
    default_model: str = field(
        default_factory=lambda: os.getenv("PRELOAD_MODEL", "large-v3")
    )
    batch_size: int = field(default_factory=lambda: int(os.getenv("BATCH_SIZE", "16")))
    cache_dir: str = field(default_factory=lambda: os.getenv("CACHE_DIR", "/.cache"))

    # Authentication
    hf_token: Optional[str] = field(default_factory=lambda: os.getenv("HF_TOKEN"))

    # Diarization defaults
    diarization_model: str = field(
        default_factory=lambda: os.getenv(
            "DIARIZATION_MODEL", "pyannote/speaker-diarization-3.1"
        )
    )

    # API limits
    max_file_size_mb: int = field(
        default_factory=lambda: int(os.getenv("MAX_FILE_SIZE_MB", "1000"))
    )

    # Server configuration
    rest_port: int = field(default_factory=lambda: int(os.getenv("REST_PORT", "9000")))
    grpc_port: int = field(default_factory=lambda: int(os.getenv("GRPC_PORT", "50051")))

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.device == "cuda" and not torch.cuda.is_available():
            # Use object.__setattr__ since dataclass is frozen
            object.__setattr__(self, "device", "cpu")
            object.__setattr__(self, "compute_type", "int8")


# Singleton instance - import this in other modules
config = AppConfig()
