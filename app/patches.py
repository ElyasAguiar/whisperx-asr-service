"""
PyTorch and Lightning Fabric Compatibility Patches

These patches resolve compatibility issues with PyTorch 2.6+ and pyannote.audio:
- PyTorch 2.6+ changed default behavior for torch.load() to require weights_only=True
- pyannote.audio models were saved with pickle and need weights_only=False
- lightning_fabric explicitly passes weights_only=True which breaks loading

IMPORTANT: This module must be imported BEFORE any imports of pyannote.audio or whisperx.
"""

import warnings
from functools import wraps

import torch

# Suppress deprecation warnings
warnings.filterwarnings("ignore", message=".*torchaudio._backend.*")
warnings.filterwarnings("ignore", message=".*In 2.9, this function.*")
warnings.filterwarnings("ignore", message=".*degrees of freedom is <= 0.*")


def apply_patches():
    """
    Apply all compatibility patches for PyTorch 2.6+ and pyannote.audio.
    Must be called before importing pyannote.audio or whisperx.
    """
    _patch_torch_load()
    _patch_lightning_fabric()


def _patch_torch_load():
    """
    Patch torch.load to force weights_only=False even if explicitly set to True.
    This resolves PyTorch 2.6+ incompatibility with pyannote.audio models.
    """
    _original_torch_load = torch.load

    @wraps(_original_torch_load)
    def _patched_torch_load(*args, **kwargs):
        # Force weights_only=False even if explicitly set to True
        kwargs["weights_only"] = False
        return _original_torch_load(*args, **kwargs)

    torch.load = _patched_torch_load


def _patch_lightning_fabric():
    """
    Patch lightning_fabric.utilities.cloud_io._load to force weights_only=False.
    lightning_fabric explicitly passes weights_only=True which breaks pyannote model loading.
    """
    try:
        from lightning_fabric.utilities import cloud_io

        _original_pl_load = cloud_io._load

        @wraps(_original_pl_load)
        def _patched_pl_load(path_or_url, map_location=None, weights_only=None):
            # Ignore weights_only argument and force False
            return _original_pl_load(
                path_or_url, map_location=map_location, weights_only=False
            )

        cloud_io._load = _patched_pl_load
    except ImportError:
        # lightning_fabric not installed, skip this patch
        pass
