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


class _ColourFormatter(logging.Formatter):
    _use_colour = sys.stdout.isatty()

    def format(self, record):
        msg = super().format(record)
        if self._use_colour:
            colour = _COLOURS.get(record.levelname, "")
            msg = f"{colour}{msg}{_RESET}"
        return msg


class Logger:
    """Per-session logger with optional file sink and stage timing."""

    def __init__(self, name: str = "pipeline", level: str = "INFO",
                 log_dir=None, verbose: bool = False):
        self.session_id = uuid.uuid4().hex[:8]
        self._stage_start: dict = {}

        self._log = logging.getLogger(f"{name}.{self.session_id}")
        self._log.setLevel(getattr(logging, level.upper(), logging.INFO))
        self._log.propagate = False

        fmt     = "%(asctime)s  %(levelname)-8s  %(message)s"
        datefmt = "%H:%M:%S"

        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(_ColourFormatter(fmt, datefmt=datefmt))
        self._log.addHandler(ch)

        if log_dir is not None:
            log_dir = Path(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(log_dir / f"pipeline_{self.session_id}.log")
            fh.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
            self._log.addHandler(fh)

        self.debug    = self._log.debug
        self.info     = self._log.info
        self.warning  = self._log.warning
        self.error    = self._log.error
        self.critical = self._log.critical

    def stage_start(self, name: str):
        self._stage_start[name] = time.time()
        self.info(f"▶  {name} started")

    def stage_done(self, name: str):
        elapsed = time.time() - self._stage_start.get(name, time.time())
        self.info(f"✓  {name} done  ({elapsed:.1f}s)")

    def flush(self):
        for h in self._log.handlers:
            h.flush()


def get_logger(name: str = "pipeline", **kwargs) -> "Logger":
    """Convenience factory — returns a configured Logger instance."""
    return Logger(name=name, **kwargs)
