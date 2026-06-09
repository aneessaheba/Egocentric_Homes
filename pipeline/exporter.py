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

def to_coco(frames: list, clip_name: str, path: Path,
            category: str = "hand_interaction") -> Path:
    """Export to a minimal COCO-compatible JSON structure."""
    _ensure(path)
    coco: dict[str, Any] = {
        "info": {"description": f"HomeHands — {clip_name}", "version": "1.0"},
        "categories": [{"id": 1, "name": category}],
        "images": [],
        "annotations": [],
    }
    ann_id = 1
    for frame in frames:
        fid = frame.get("frame_id", 0)
        coco["images"].append({"id": fid, "file_name": f"{clip_name}_{fid:06d}.jpg"})
        for hand in frame.get("hands", []):
            coco["annotations"].append({
                "id": ann_id, "image_id": fid, "category_id": 1,
                "keypoints": hand.get("landmarks", []),
            })
            ann_id += 1
    with open(path, "w", encoding="utf-8") as f:
        json.dump(coco, f, indent=2)
    return path

def summary_stats(frames: list) -> dict:
    """Return aggregate statistics over the annotation frames."""
    total = len(frames)
    if total == 0:
        return {"total_frames": 0}
    hand_frames = sum(1 for f in frames if f.get("hands"))
    scores = [f["quality_score"] for f in frames if "quality_score" in f]
    return {
        "total_frames":      total,
        "hand_frames":       hand_frames,
        "hand_visibility_%": round(hand_frames / total * 100, 1),
        "avg_quality_score": round(sum(scores) / len(scores), 2) if scores else None,
    }

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python pipeline/exporter.py <input.json> <json|csv|coco>")
        sys.exit(1)
    src = Path(sys.argv[1])
    fmt = sys.argv[2].lower()
    with open(src) as f:
        data = json.load(f)
    out = src.with_suffix(f".export.{fmt}")
    if fmt == "json":
        to_json(data, out)
    elif fmt == "csv":
        to_csv(data if isinstance(data, list) else [data], out)
    elif fmt == "coco":
        to_coco(data if isinstance(data, list) else [data], src.stem, out)
    else:
        print(f"Unknown format: {fmt}"); sys.exit(1)
    print(f"Exported → {out}")
