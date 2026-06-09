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

def validate_duration(seconds: float, min_sec: float = 1.0,
                      max_sec: float = 300.0) -> float:
    if not (min_sec <= seconds <= max_sec):
        raise ValueError(
            f"Duration {seconds:.1f}s outside accepted range [{min_sec}, {max_sec}]s"
        )
    return float(seconds)

# ── String / code validators ───────────────────────────────────────
LANGUAGE_CODES = {
    "en","hi","bn","te","mr","ta","gu","kn","pa","ml",
    "ur","or","as","zh","ja","ko","fr","de","es","ar"
}

def validate_language(code: str) -> str:
    code = code.strip().lower()
    if code not in LANGUAGE_CODES:
        raise ValueError(f"Unknown language code '{code}'")
    return code

def validate_clip_name(name: str) -> str:
    import re
    if not re.match(r'^[A-Za-z0-9_\-]+$', name):
        raise ValueError(
            f"Clip name '{name}' has invalid characters. Use letters, digits, _ or -."
        )
    return name

# ── Model / path ──────────────────────────────────────────────────
def validate_model_path(path: Union[str, Path]) -> Path:
    p = Path(path)
    if not p.exists():
        raise ValueError(f"Model checkpoint not found: {p}")
    return p

def validate_output_dir(path: Union[str, Path]) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

# ── Config dict ───────────────────────────────────────────────────
def validate_quality_weights(weights: dict) -> dict:
    total = sum(weights.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Quality weights must sum to 1.0, got {total:.4f}")
    return weights

def validate_batch_size(n: int) -> int:
    if n < 1:
        raise ValueError(f"Batch size must be >= 1, got {n}")
    return int(n)

__all__ = [
    "validate_video_path", "validate_video_dir",
    "validate_score", "validate_confidence",
    "validate_fps", "validate_resolution", "validate_duration",
    "validate_language", "validate_clip_name",
    "validate_model_path", "validate_output_dir",
    "validate_quality_weights", "validate_batch_size",
]
