"""exporter.py — Export pipeline annotations to multiple formats.

Supports JSON (default), flat CSV, and COCO-style JSON.
"""
import csv
import json
from pathlib import Path
from typing import Any


def _ensure(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path

def to_json(data, path: Path, indent: int = 2) -> Path:
    """Write annotation data to a JSON file."""
    _ensure(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)
    return path
