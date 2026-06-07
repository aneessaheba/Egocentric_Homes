"""logger.py — Shared logging utilities for the HomeHands pipeline.
"""
import logging
import sys
import time

import uuid
from pathlib import Path

# ── ANSI colour codes ─────────────────────────────────────────────
_RESET  = "\033[0m"
_BOLD   = "\033[1m"
_RED    = "\033[31m"
_YELLOW = "\033[33m"
_GREEN  = "\033[32m"
_CYAN   = "\033[36m"
_GREY   = "\033[90m"

_COLOURS = {
    "DEBUG":    _GREY,
    "INFO":     _CYAN,
    "WARNING":  _YELLOW,
    "ERROR":    _RED,
    "CRITICAL": _RED + _BOLD,
}
