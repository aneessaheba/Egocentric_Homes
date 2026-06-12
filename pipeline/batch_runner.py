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


def collect_clips(videos_dir: Path, resume: bool = False) -> list:
    """Return sorted list of .mp4 clips to process, skipping done clips if resume."""
    clips = sorted(videos_dir.glob("*.mp4"))
    if resume:
        clips = [
            c for c in clips
            if not (ANNOTATIONS_DIR / f"{c.stem}_full.json").exists()
        ]
    return clips


def process_clip(clip: Path) -> tuple:
    """Process a single clip; return (name, success, message)."""
    try:
        import run_pipeline
        run_pipeline.main()
        return clip.stem, True, "OK"
    except Exception as exc:
        return clip.stem, False, str(exc)[:120]


def run_batch(videos_dir: Path, workers: int = MAX_WORKERS,
              resume: bool = False, batch_size: int = BATCH_SIZE):
    """Process all clips in *videos_dir* using *workers* parallel processes."""
    ensure_dirs(ANNOTATIONS_DIR, REJECTED_DIR)
    clips = collect_clips(videos_dir, resume=resume)
    if not clips:
        print("  No clips to process.")
        return
    print_header(f"Batch runner  |  {len(clips)} clip(s)  |  {workers} worker(s)")
    timer   = Timer()
    results = []
    for i in range(0, len(clips), batch_size):
        chunk = clips[i : i + batch_size]
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as exe:
            futures = {exe.submit(process_clip, c): c for c in chunk}
            for fut in concurrent.futures.as_completed(futures):
                name, ok, msg = fut.result()
                print(f"  [{'OK' if ok else 'FAIL'}]  {name}  —  {msg}")
                results.append((name, ok))
    passed = sum(1 for _, ok in results if ok)
    print(f"\n  {len(results)} clips  |  {passed} passed  |  {len(results)-passed} failed")
    print_done("Batch runner", timer)
