"""
depth_preview.py
─────────────────
On-demand colorized depth visualization for spot-checking a clip.

depth.py intentionally does not save a colorized preview for every frame of
every clip — at native 4K resolution that cost ~3.5GB/clip for a debug
visualization with no schema role (see depth.py's docstring). Use this
script instead whenever you actually want to look at a clip's depth quality:
it runs the same model on just the frames you ask for and saves a colorized
PNG for each, without touching the real pipeline outputs.

Usage:
  python pipeline/depth_preview.py assets/videos/WashingCup.mp4 --frames 0 150 600
"""

import argparse
from pathlib import Path

import cv2

from depth import load_depth_model, estimate_depth_meters, colorize_depth

OUT_ROOT = Path("assets/processed/depth_preview_spotcheck")


def main():
    parser = argparse.ArgumentParser(description="Spot-check depth quality on specific frames of a clip")
    parser.add_argument("video", help="Path to input video file")
    parser.add_argument("--frames", type=int, nargs="+", required=True, help="Frame indices to preview")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"[ERROR] Video not found: {video_path}")
        return

    out_dir = OUT_ROOT / video_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    pipe = load_depth_model()
    cap = cv2.VideoCapture(str(video_path))

    for frame_id in args.frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()
        if not ret:
            print(f"  Could not read frame {frame_id}")
            continue

        depth_m = estimate_depth_meters(pipe, frame)
        preview = colorize_depth(depth_m)

        out_path = out_dir / f"frame_{frame_id:06d}.png"
        cv2.imwrite(str(out_path), preview)
        print(f"  Saved {out_path}  (min={depth_m.min():.2f}m max={depth_m.max():.2f}m)")

    cap.release()


if __name__ == "__main__":
    main()
