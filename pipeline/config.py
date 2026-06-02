"""config.py — Centralised pipeline configuration.
"""
from pathlib import Path

# ── Directories ───────────────────────────────────────────────────
ASSETS_DIR    = Path("assets")
VIDEOS_DIR    = ASSETS_DIR / "videos"
PROCESSED_DIR = ASSETS_DIR / "processed"
REJECTED_DIR  = VIDEOS_DIR / "rejected"
