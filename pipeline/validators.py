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

# ── Numeric bounds ─────────────────────────────────────────────────
def validate_score(value: float, lo: float = 0.0, hi: float = 100.0,
                   name: str = "score") -> float:
    if not (lo <= value <= hi):
        raise ValueError(f"{name} must be in [{lo}, {hi}], got {value}")
    return float(value)

def validate_confidence(value: float, name: str = "confidence") -> float:
    return validate_score(value, 0.0, 1.0, name)

def validate_fps(fps: float) -> float:
    if fps <= 0:
        raise ValueError(f"FPS must be positive, got {fps}")
    return float(fps)

def validate_resolution(width: int, height: int):
    if width <= 0 or height <= 0:
        raise ValueError(f"Resolution must be positive, got {width}x{height}")
    return width, height
