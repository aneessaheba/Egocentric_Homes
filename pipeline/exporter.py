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

def to_csv(frames: list, path: Path) -> Path:
    """Flatten per-frame annotations to a CSV (one row per frame)."""
    if not frames:
        raise ValueError("Cannot export empty frame list to CSV")
    _ensure(path)
    keys: list = []
    for frame in frames:
        for k in frame:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for frame in frames:
            writer.writerow(frame)
    return path
