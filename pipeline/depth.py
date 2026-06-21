"""
depth.py
────────
Per-frame metric depth estimation for egocentric household activity videos,
using Depth-Anything-V2-Metric-Indoor-Small (Hypersim fine-tune) via
Hugging Face `transformers`.

This is the metric-depth variant, not the plain relative-depth model — it
outputs real depth in meters (fine-tuned on indoor scenes), so the dataset's
public "Avg Depth (m)" stat is an actual measured quantity rather than an
unscaled relative value.

Inference runs on a downscaled copy of each frame (fixed width, aspect
ratio preserved) for speed, then the predicted depth map is resized back up
to the frame's native resolution before saving — this is the standard
upsample-back-to-input-res convention used by every published monocular
depth pipeline; it does not add real sub-pixel precision, just makes the
depth map directly pixel-aligned with the RGB frame (and with segmentation
masks from segmentation.py) for downstream lookups.

Saved as 16-bit grayscale PNGs, depth in millimeters (matches the project's
already-public "16-bit / PNG" depth format claim): meters = pixel_value / 1000.

No colorized preview is written here — at native 4K resolution a per-frame
preview cost ~3.5GB/clip for a debug visualization that isn't referenced
anywhere in the schema. Use depth_preview.py to spot-check a clip on demand
instead of paying that cost permanently for every frame of every clip.

Outputs:
  • 16-bit depth PNG per frame    →  assets/processed/depth/<clip_name>/
  • updated hand pose JSON         →  assets/processed/hand_pose/<clip_name>.json
    ("depth" block added to every frame entry)

Usage:
  python pipeline/depth.py assets/videos/WashingCup.mp4
"""

import sys
import json
import time
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import pipeline as hf_pipeline


# ── Paths ─────────────────────────────────────────────────────────────────────

HAND_POSE_ROOT = Path("assets/processed/hand_pose")
DEPTH_ROOT     = Path("assets/processed/depth")    # real 16-bit depth PNGs

# ── Model ─────────────────────────────────────────────────────────────────────

MODEL_ID    = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
METHOD_NAME = "Depth-Anything-V2-Metric-Indoor-Small"

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# Frames are downscaled to this width (aspect ratio preserved) before being
# fed to the model — full native-resolution inference measured ~19x slower
# on 4K input for no real gain (the network's effective receptive detail
# doesn't scale with input pixels at this size).
INFERENCE_WIDTH = 640

# Depth values are stored as millimeters in a 16-bit PNG: meters = px / DEPTH_SCALE.
DEPTH_SCALE = 1000
DEPTH_MAX_MM = 65535   # uint16 ceiling — caps representable depth at 65.535m, far beyond any indoor scene


# ── Model loader ──────────────────────────────────────────────────────────────

def load_depth_model():
    print(f"  [model] Loading {METHOD_NAME} ({DEVICE}) ...")
    pipe = hf_pipeline(task="depth-estimation", model=MODEL_ID, device=DEVICE)
    print("  [model] Depth model ready.\n")
    return pipe


# ── Per-frame depth ────────────────────────────────────────────────────────────

def estimate_depth_meters(pipe, frame_bgr: np.ndarray) -> np.ndarray:
    """Run the depth model on a downscaled copy of frame_bgr, return a
    float32 depth map (meters) resized back to the frame's native resolution."""
    h, w = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    infer_h = max(1, round(INFERENCE_WIDTH * h / w))
    small = cv2.resize(rgb, (INFERENCE_WIDTH, infer_h), interpolation=cv2.INTER_AREA)

    out = pipe(Image.fromarray(small))
    depth_small = out["predicted_depth"].squeeze().cpu().numpy().astype(np.float32)

    return cv2.resize(depth_small, (w, h), interpolation=cv2.INTER_LINEAR)


def write_depth_png(depth_m: np.ndarray, out_path: Path) -> None:
    depth_mm = np.clip(depth_m * DEPTH_SCALE, 0, DEPTH_MAX_MM).astype(np.uint16)
    cv2.imwrite(str(out_path), depth_mm)


def colorize_depth(depth_m: np.ndarray) -> np.ndarray:
    """Normalize a depth map to 0-255 and apply a colormap, for visual QA only."""
    d_min, d_max = float(depth_m.min()), float(depth_m.max())
    span = max(d_max - d_min, 1e-6)
    norm = ((depth_m - d_min) / span * 255).astype(np.uint8)
    return cv2.applyColorMap(norm, cv2.COLORMAP_INFERNO)


def depth_entry(depth_m: np.ndarray, depth_path: Path) -> dict:
    write_depth_png(depth_m, depth_path)
    return {
        "method":      METHOD_NAME,
        "depth_path":  str(depth_path),
        "depth_format": "png16",
        "depth_unit":  "mm",     # pixel value unit inside the PNG
        "depth_scale": DEPTH_SCALE,  # meters = pixel_value / depth_scale
        "min_depth_m": round(float(depth_m.min()), 4),
        "max_depth_m": round(float(depth_m.max()), 4),
        "avg_depth_m": round(float(depth_m.mean()), 4),
    }


# ── Main pipeline ─────────────────────────────────────────────────────────────

def process_video(video_path_str: str, output_video: bool = False):
    video_path = Path(video_path_str)
    if not video_path.exists():
        print(f"[ERROR] Video not found: {video_path}")
        sys.exit(1)

    clip_name = video_path.stem

    print(f"\n{'=' * 60}")
    print(f"  Depth Pipeline  ({METHOD_NAME})")
    print(f"  Clip : {clip_name}")
    print(f"{'=' * 60}\n")

    # ── Load hand pose JSON ───────────────────────────────────
    json_path = HAND_POSE_ROOT / f"{clip_name}.json"
    if not json_path.exists():
        print(f"[ERROR] Hand pose JSON not found: {json_path}")
        print(f"        Run hand_pose.py first.")
        sys.exit(1)

    print("  [1] Loading hand pose JSON ...")
    with open(json_path) as f:
        pose_data = json.load(f)
    frame_lookup = {fr["frame_id"]: fr for fr in pose_data["frames"]}
    print(f"       {len(pose_data['frames'])} frames loaded\n")

    # ── Load depth model once ─────────────────────────────────
    print("  [2] Loading depth model ...")
    pipe = load_depth_model()

    # ── Open video ────────────────────────────────────────────
    print("  [3] Opening video ...")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] Could not open: {video_path}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"       {width}x{height} | {total_frames} frames\n")

    # ── Create output folder ───────────────────────────────────
    depth_dir = DEPTH_ROOT / clip_name
    depth_dir.mkdir(parents=True, exist_ok=True)

    # Colorized preview video is built in-memory frame by frame if requested —
    # no per-frame preview PNGs are ever written to disk (see module docstring).
    video_writer = None
    if output_video:
        out_path = Path("assets/processed") / f"{clip_name}_depth.mp4"
        video_writer = cv2.VideoWriter(
            str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )

    # ── Frame loop ────────────────────────────────────────────
    frame_id   = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        depth_m = estimate_depth_meters(pipe, frame)

        depth_path = depth_dir / f"frame_{frame_id:06d}.png"
        entry = depth_entry(depth_m, depth_path)

        if video_writer is not None:
            video_writer.write(colorize_depth(depth_m))

        if frame_id in frame_lookup:
            frame_lookup[frame_id]["depth"] = entry

        if frame_id > 0 and frame_id % 100 == 0:
            elapsed = time.time() - start_time
            print(f"  Frame {frame_id}/{total_frames} | {elapsed:.1f}s elapsed")

        frame_id += 1

    cap.release()
    if video_writer is not None:
        video_writer.release()
        print(f"\n  Depth preview video saved → assets/processed/{clip_name}_depth.mp4")

    # ── Save updated JSON ─────────────────────────────────────
    print(f"\n  Saving updated JSON → {json_path}")
    with open(json_path, "w") as f:
        json.dump(pose_data, f, indent=2)

    # ── Final summary ─────────────────────────────────────────
    total_time = time.time() - start_time
    print(f"\n{'─' * 60}")
    print(f"  Frames processed : {frame_id}")
    print(f"  Time             : {total_time/60:.1f} min")
    print(f"  Depth PNGs       → {depth_dir}/")
    print(f"{'─' * 60}\n")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Depth-Anything-V2 Metric Indoor Depth Estimation Pipeline"
    )
    parser.add_argument("video", help="Path to input video file")
    parser.add_argument("--output-video", action="store_true",
                         help="Stitch colorized depth frames into a preview video")
    args = parser.parse_args()

    process_video(args.video, output_video=args.output_video)
