"""batch_runner.py — Parallel batch processing for the HomeHands pipeline.

Wraps individual stage scripts in a concurrent.futures executor.

Usage:
  python pipeline/batch_runner.py assets/videos/ --workers 2
  python pipeline/batch_runner.py assets/videos/ --resume
"""
import sys
import concurrent.futures
from pathlib import Path

from utils import Timer, print_header, print_done, ensure_dirs

try:
    from config import (
        VIDEOS_DIR, ANNOTATIONS_DIR, REJECTED_DIR,
        BATCH_SIZE, MAX_WORKERS,
    )
except ImportError:
    VIDEOS_DIR      = Path("assets/videos")
    ANNOTATIONS_DIR = Path("assets/processed/annotations")
    REJECTED_DIR    = Path("assets/videos/rejected")
    BATCH_SIZE      = 4
    MAX_WORKERS     = 2
