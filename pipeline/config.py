"""config.py — Centralised pipeline configuration.
"""
from pathlib import Path

# ── Directories ───────────────────────────────────────────────────
ASSETS_DIR    = Path("assets")
VIDEOS_DIR    = ASSETS_DIR / "videos"
PROCESSED_DIR = ASSETS_DIR / "processed"
REJECTED_DIR  = VIDEOS_DIR / "rejected"

# ── Processed sub-directories ─────────────────────────────────────
ANNOTATIONS_DIR = PROCESSED_DIR / "annotations"
HAND_DIR        = PROCESSED_DIR / "hand_pose"
ARM_DIR         = PROCESSED_DIR / "arm_pose"
SEG_DIR         = PROCESSED_DIR / "segmentation"
DEPTH_DIR       = PROCESSED_DIR / "depth"
QUALITY_DIR     = PROCESSED_DIR / "quality"
TRANSCRIPT_DIR  = PROCESSED_DIR / "transcripts"
