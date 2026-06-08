"""validators.py — Input validation helpers for the HomeHands pipeline.

Each function raises ValueError with a descriptive message on bad input.
"""
from pathlib import Path
from typing import Union

# ── Video file ─────────────────────────────────────────────────────
SUPPORTED_EXTS = {".mp4", ".mov", ".avi", ".mkv"}

def validate_video_path(path: Union[str, Path]) -> Path:
    """Ensure path points to an existing, supported video file."""
    p = Path(path)
    if not p.exists():
        raise ValueError(f"Video not found: {p}")
    if p.suffix.lower() not in SUPPORTED_EXTS:
        raise ValueError(f"Unsupported extension '{p.suffix}'")
    return p

def validate_video_dir(path: Union[str, Path]) -> Path:
    """Ensure path is a directory containing at least one .mp4 file."""
    p = Path(path)
    if not p.is_dir():
        raise ValueError(f"Not a directory: {p}")
    if not list(p.glob("*.mp4")):
        raise ValueError(f"No .mp4 files found in {p}")
    return p
