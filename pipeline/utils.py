"""
utils.py
────────
Shared utility functions used across the HomeHands pipeline.

Covers: JSON I/O, video metadata, directory setup, and progress printing.
"""

import json
import sys
import time
from pathlib import Path

import cv2


# ── JSON helpers ──────────────────────────────────────────────────────────────

def load_json(path: Path) -> dict | list | None:
    """Load a JSON file and return its contents, or None if the file doesn't exist."""
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: dict | list, path: Path, indent: int = 2):
    """Write data to a JSON file, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent)


# ── Video metadata ────────────────────────────────────────────────────────────

def video_info(video_path: Path) -> dict:
    """
    Return basic metadata for a video file without decoding any frames.

    Returns:
        {
          "width": int, "height": int,
          "fps": float, "total_frames": int,
          "duration_sec": float
        }
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    info = {
        "width":        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height":       int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps":          cap.get(cv2.CAP_PROP_FPS),
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    cap.release()
    fps = info["fps"] or 1
    info["duration_sec"] = round(info["total_frames"] / fps, 3)
    return info


# ── Directory helpers ─────────────────────────────────────────────────────────

def ensure_dirs(*paths: Path):
    """Create one or more directories (and any missing parents) if they don't exist."""
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


# ── Progress printing ─────────────────────────────────────────────────────────

class Timer:
    """Simple wall-clock timer."""

    def __init__(self):
        self._start = time.time()

    def elapsed(self) -> float:
        return time.time() - self._start

    def elapsed_str(self) -> str:
        s = self.elapsed()
        if s < 60:
            return f"{s:.1f}s"
        return f"{int(s // 60)}m {int(s % 60)}s"


def print_progress(frame_id: int, total: int, timer: Timer, every: int = 100):
    """Print a progress line every `every` frames."""
    if frame_id % every == 0 and total > 0:
        pct = frame_id / total * 100
        print(f"  Frame {frame_id:>6} / {total}  ({pct:5.1f}%)  |  {timer.elapsed_str()}")


def print_header(title: str, width: int = 60):
    print(f"\n{'═' * width}")
    print(f"  {title}")
    print(f"{'═' * width}\n")


def print_done(label: str, timer: Timer):
    print(f"\n  DONE  —  {label}  ({timer.elapsed_str()})\n")
