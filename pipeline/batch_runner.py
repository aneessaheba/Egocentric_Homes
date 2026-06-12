"""batch_runner.py — Parallel batch processing for the HomeHands pipeline.

Wraps individual stage scripts in a concurrent.futures executor.

Usage:
  python pipeline/batch_runner.py assets/videos/ --workers 2
  python pipeline/batch_runner.py assets/videos/ --resume
"""
import sys
import concurrent.futures
from pathlib import Path
