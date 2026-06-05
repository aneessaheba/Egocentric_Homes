"""
arm_tracking.py
───────────────
Full arm tracking pipeline for head-mounted GoPro egocentric videos.

Algorithm (per frame):
  1. MediaPipe HolisticLandmarker → wrist pixel positions
  2. If wrist not detected this frame → reuse last known position
     (arms don't teleport; last position is a reasonable estimate)
  3. SAM2 with 3 foreground + 2 background prompts → arm mask
     Foreground: wrist + two points up the forearm toward elbow
     Background: just above wrist (object being touched) + table edge
  4. Save colored mask PNG + update tracking JSON

Output per clip:
  assets/processed/tracked/<clip>/frame_000001.png  ← colored arm frames
  assets/processed/tracked/<clip>_tracking.json     ← per-frame data

Usage:
  python pipeline/arm_tracking.py assets/videos/WashingCup.mp4

Options (edit CONFIG below):
  RESIZE_FOR_SAM2   — downsample before SAM2 for speed (keep full res for output)
  MAX_GAP_FRAMES    — stop propagating if wrist unseen for this many frames
  PROGRESS_EVERY    — print progress every N frames
"""

# ── Imports ───────────────────────────────────────────────────────────────────

import sys            # command-line argument (video path)
import json           # write tracking JSON
import time           # measure elapsed time
import urllib.request # download SAM2 checkpoint if missing
import tempfile       # temporary directory for frame extraction
import shutil         # clean up temp directory
from pathlib import Path

import cv2            # open video, read frames, save PNGs
import numpy as np    # mask math and array operations
import torch          # required by SAM2 internally

import mediapipe as mp                              # holistic landmark detection
from mediapipe.tasks import python as mp_tasks     # BaseOptions
from mediapipe.tasks.python import vision as mp_vision   # HolisticLandmarker

from sam2.build_sam import build_sam2              # build SAM2 model from config
from sam2.sam2_image_predictor import SAM2ImagePredictor   # per-image segmentation


# ── Configuration ─────────────────────────────────────────────────────────────

# Model file paths
HOLISTIC_MODEL = Path("assets/models/holistic_landmarker.task")
SAM2_CKPT      = Path("assets/models/sam2_hiera_tiny.pt")
SAM2_CFG       = "sam2_hiera_t.yaml"
SAM2_CKPT_URL  = "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt"

# Output folders
TRACKED_ROOT = Path("assets/processed/tracked")

# SAM2 processing resolution
# 4K (3840×2160) is slow for SAM2. We resize DOWN before segmentation,
# then scale the mask BACK UP to full resolution for the output PNG.
# 960×540 is 4× smaller and about 16× faster while keeping good quality.
RESIZE_W = 960    # width to resize to before SAM2
RESIZE_H = 540    # height to resize to before SAM2

# Tracking memory: how many frames to keep using the last known wrist position
# if MediaPipe loses the wrist. Beyond this, the arm is considered "gone".
MAX_GAP_FRAMES = 45   # ~1.5 seconds at 30fps

# Arm prompt offsets (as fraction of FULL frame height)
# Because GoPro is top-down, "up the arm" = LOWER in the frame (higher y value)
ARM_MID_OFFSET   = 0.15   # forearm midpoint: 15% of height below wrist
ARM_LOWER_OFFSET = 0.28   # near elbow:       28% of height below wrist
BG_ABOVE_OFFSET  = 0.04   # background point: 4%  of height ABOVE wrist

# Mask blend opacity: 0.55 = 55% color + 45% original
MASK_ALPHA = 0.55

# Colors (BGR — OpenCV is Blue, Green, Red)
COLOR_RIGHT = (0,   200, 0  )   # green  — right arm
COLOR_LEFT  = (0,   0,   200)   # red    — left arm

# How often to print progress (every N frames)
PROGRESS_EVERY = 100


# ── Utilities ─────────────────────────────────────────────────────────────────

def get_device() -> str:
    """Return best available compute device for SAM2."""
    if torch.cuda.is_available():
        return "cuda"    # NVIDIA GPU
    if torch.backends.mps.is_available():
        return "mps"     # Apple Silicon GPU
    return "cpu"          # fallback


def ensure_sam2():
    """Download SAM2 checkpoint if not present."""
    SAM2_CKPT.parent.mkdir(parents=True, exist_ok=True)
    if SAM2_CKPT.exists():
        return
    print("  [SAM2] Downloading checkpoint (~155 MB)...")
    urllib.request.urlretrieve(SAM2_CKPT_URL, SAM2_CKPT)
    print("  [SAM2] Done.\n")


def scale_point(px, py, from_w, from_h, to_w, to_h):
    """
    Scale a pixel coordinate from one resolution to another.

    We detect wrists at full resolution (3840×2160) but run SAM2 at
    a smaller resolution (960×540). This function converts between the two.
    """
    sx = int(px * to_w / from_w)   # multiply by the scale ratio
    sy = int(py * to_h / from_h)
    return (sx, sy)


# ── STEP 1: MediaPipe wrist detection ─────────────────────────────────────────

def run_mediapipe_pass(video_path: Path, total_frames: int) -> dict:
    """
    Pass 1 of 2: run MediaPipe HolisticLandmarker on every frame to
    detect wrist positions. Returns a dict:
        {frame_id: {"left": (px,py)|None, "right": (px,py)|None}}

    We do this as a dedicated pass first (not interleaved with SAM2)
    so we can compute the full tracking with memory before segmenting.
    """
    print("  [Pass 1/2] Running MediaPipe on all frames to detect wrists ...")

    options = mp_vision.HolisticLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=str(HOLISTIC_MODEL)),
        running_mode=mp_vision.RunningMode.IMAGE,
        min_pose_detection_confidence=0.5,
        min_pose_landmarks_confidence=0.5,
        min_hand_landmarks_confidence=0.5,
    )
    detector = mp_vision.HolisticLandmarker.create_from_options(options)

    cap    = cv2.VideoCapture(str(video_path))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    raw_detections = {}   # frame_id → {"left": pt|None, "right": pt|None}
    frame_id       = 0
    detected_count = 0    # how many frames had at least one wrist

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR → RGB for MediaPipe
        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result   = detector.detect(mp_image)

        # Use POSE wrist landmarks (indices 15=left, 16=right)
        # These are more reliable than hand WRIST for head-mounted cameras
        pose = result.pose_landmarks   # list of 33 body landmarks or []

        def pose_pt(idx):
            """Get (px,py) for pose landmark at idx, or None."""
            if not pose or idx >= len(pose):
                return None
            lm = pose[idx]
            return (int(lm.x * width), int(lm.y * height))

        left_pt  = pose_pt(15)   # LEFT_WRIST
        right_pt = pose_pt(16)   # RIGHT_WRIST

        # Also try hand WRIST as a fallback (more precise when detected)
        def hand_wrist(lms):
            """Return WRIST (index 0) pixel coords from hand landmarks."""
            if len(lms) == 21:
                return (int(lms[0].x * width), int(lms[0].y * height))
            return None

        left_hand_wrist  = hand_wrist(result.left_hand_landmarks)
        right_hand_wrist = hand_wrist(result.right_hand_landmarks)

        # Prefer hand wrist (more precise) over pose wrist when available
        if left_hand_wrist:
            left_pt = left_hand_wrist
        if right_hand_wrist:
            right_pt = right_hand_wrist

        raw_detections[frame_id] = {"left": left_pt, "right": right_pt}

        if left_pt or right_pt:
            detected_count += 1

        if frame_id % PROGRESS_EVERY == 0:
            print(f"    Frame {frame_id:>5} / {total_frames}  "
                  f"({frame_id/total_frames*100:5.1f}%)  "
                  f"detected in {detected_count} frames so far")

        frame_id += 1

    cap.release()
    detector.close()

    print(f"    Done. Wrists detected in {detected_count}/{frame_id} frames "
          f"({detected_count/frame_id*100:.1f}%)\n")
    return raw_detections, width, height


# ── STEP 2: Fill tracking gaps ────────────────────────────────────────────────

def build_tracking(raw: dict, total_frames: int) -> dict:
    """
    Turn raw per-frame detections into a smooth tracking dict.

    For frames where MediaPipe lost the wrist, reuse the last known position
    (up to MAX_GAP_FRAMES). This prevents flickering/gaps during brief occlusions.

    Returns: {frame_id: {"left": (px,py)|None, "right": (px,py)|None,
                         "left_fresh": bool, "right_fresh": bool}}
    "fresh" = True when the position comes from a real detection this frame
    "fresh" = False when we're reusing the last known position
    """
    print("  [Building tracking timeline] Filling gaps up to "
          f"{MAX_GAP_FRAMES} frames ...")

    tracking = {}

    last_left  = None   # last known left wrist (px, py)
    last_right = None   # last known right wrist (px, py)
    gap_left   = 0      # frames since last left detection
    gap_right  = 0      # frames since last right detection

    for fid in range(total_frames):
        det = raw.get(fid, {"left": None, "right": None})

        # ── Left wrist ────────────────────────────────────────
        if det["left"]:
            last_left   = det["left"]   # fresh detection — update
            gap_left    = 0
            left_fresh  = True
        elif last_left and gap_left < MAX_GAP_FRAMES:
            gap_left   += 1             # no detection — reuse last position
            left_fresh  = False         # mark as estimated (not fresh)
        else:
            last_left  = None           # too many frames without detection — give up
            gap_left   = 0
            left_fresh = False

        # ── Right wrist ───────────────────────────────────────
        if det["right"]:
            last_right  = det["right"]
            gap_right   = 0
            right_fresh = True
        elif last_right and gap_right < MAX_GAP_FRAMES:
            gap_right  += 1
            right_fresh = False
        else:
            last_right = None
            gap_right  = 0
            right_fresh = False

        tracking[fid] = {
            "left":        last_left,     # (px,py) or None
            "right":       last_right,
            "left_fresh":  left_fresh,    # True = real detection
            "right_fresh": right_fresh,
        }

    active = sum(1 for t in tracking.values()
                 if t["left"] or t["right"])
    print(f"    Active (at least one arm) in {active}/{total_frames} frames "
          f"({active/total_frames*100:.1f}%)\n")
    return tracking


# ── STEP 3: build SAM2 prompt points ─────────────────────────────────────────

def build_prompts(wrist_px, wrist_py, frame_h, frame_w, full_h):
    """
    Build foreground + background SAM2 point prompts for one arm.

    wrist_px/py  — wrist position in FULL resolution
    frame_h/w    — SAM2 input resolution (smaller, for speed)
    full_h       — original frame height (for computing offsets)

    Returns:
      point_coords — numpy array shape (N, 2)  pixel positions
      point_labels — numpy array shape (N,)    1=foreground, 0=background
    """

    # ── Scale wrist from full resolution to SAM2 input resolution ────────────
    # We run SAM2 at RESIZE_W × RESIZE_H for speed.
    # The wrist coordinates were detected at full resolution, so we scale them.
    sx = int(wrist_px * frame_w / RESIZE_W * (frame_w / RESIZE_W))   # simplified below
    sy = int(wrist_py * frame_h / RESIZE_H * (frame_h / RESIZE_H))

    # Cleaner: just scale by ratio
    w_ratio = frame_w / RESIZE_W   # how much smaller the SAM2 input is
    h_ratio = frame_h / RESIZE_H

    # Wrist in SAM2-resolution coordinates
    wx = int(wrist_px / w_ratio)
    wy = int(wrist_py / h_ratio)

    # Arm offset in SAM2-resolution pixels
    mid   = int((full_h * ARM_MID_OFFSET)   / h_ratio)   # forearm midpoint
    lower = int((full_h * ARM_LOWER_OFFSET) / h_ratio)   # near elbow
    above = int((full_h * BG_ABOVE_OFFSET)  / h_ratio)   # background (above wrist)

    # Clamp all y values to stay inside the frame
    wy_mid   = min(wy + mid,   frame_h - 1)
    wy_lower = min(wy + lower, frame_h - 1)
    wy_above = max(wy - above, 0)

    # ── Foreground points (label = 1, include in mask) ───────────────────────
    fg_points = [
        [wx, wy],          # 1. wrist itself
        [wx, wy_mid],      # 2. forearm midpoint (toward elbow)
        [wx, wy_lower],    # 3. near elbow
    ]

    # ── Background points (label = 0, exclude from mask) ────────────────────
    # Point just above wrist = the object being touched (cup, banana, etc.)
    # Top-center = the table/floor background
    bg_points = [
        [wx, wy_above],                    # just above wrist → exclude object
        [frame_w // 2, int(frame_h * 0.05)],  # top-center → exclude background table
    ]

    # Combine into arrays for SAM2
    all_points = fg_points + bg_points
    all_labels = [1, 1, 1, 0, 0]   # 1=foreground for first 3, 0=background for last 2

    point_coords = np.array(all_points, dtype=np.float32)   # shape (5, 2)
    point_labels = np.array(all_labels, dtype=np.int32)      # shape (5,)

    return point_coords, point_labels


# ── STEP 4: segment one arm ───────────────────────────────────────────────────

def segment_arm(predictor, frame_small, wrist_px, wrist_py,
                full_w, full_h, full_frame):
    """
    Run SAM2 on a downscaled frame and return the mask at FULL resolution.

    predictor   — loaded SAM2ImagePredictor
    frame_small — already-resized BGR frame (RESIZE_W × RESIZE_H)
    wrist_px/py — wrist position in FULL resolution
    full_w/h    — original frame dimensions (for scaling mask back up)
    full_frame  — original full-res frame (used only to get dimensions)

    Returns boolean mask at full resolution, or None on failure.
    """

    # Build prompts for this arm (in SAM2 / small-frame coordinates)
    point_coords, point_labels = build_prompts(
        wrist_px, wrist_py,
        frame_h=RESIZE_H, frame_w=RESIZE_W,
        full_h=full_h,
    )

    # Run SAM2 prediction
    with torch.no_grad():   # no_grad saves memory — we don't need gradients
        masks, scores, _ = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=False,   # return the single best mask
        )

    # masks[0] is shape (RESIZE_H, RESIZE_W) — boolean
    small_mask = masks[0].astype(bool)

    # Scale the small mask back up to the original full resolution
    # INTER_NEAREST keeps it binary (no blurring of edges)
    full_mask = cv2.resize(
        small_mask.astype(np.uint8),      # uint8 for cv2.resize
        (full_w, full_h),
        interpolation=cv2.INTER_NEAREST,  # no blurring of 0/1 values
    ).astype(bool)   # convert back to bool

    return full_mask


# ── STEP 5: draw mask overlay ─────────────────────────────────────────────────

def draw_overlay(frame, mask, color, wrist_pt, is_fresh):
    """
    Blend a colored mask over the arm pixels in-place.
    Draw a dot at the wrist position.
    Draw a small triangle marker if this is a tracked (not fresh) position.

    is_fresh — True = real MediaPipe detection; False = gap-filled estimate
    """

    # Blend colored mask onto frame
    overlay        = frame.copy()
    overlay[mask]  = color
    cv2.addWeighted(overlay, MASK_ALPHA, frame, 1.0 - MASK_ALPHA, 0, frame)

    # Draw wrist dot
    dot_color = (255, 255, 255)   # white center always
    cv2.circle(frame, wrist_pt, 16, color,     -1, cv2.LINE_AA)  # colored outer
    cv2.circle(frame, wrist_pt, 10, dot_color, -1, cv2.LINE_AA)  # white inner

    # If this position is estimated (not a fresh detection), draw a small "~"
    # indicator so you can see at a glance which frames used interpolated data
    if not is_fresh:
        cv2.circle(frame, wrist_pt, 18, (0, 200, 255), 3, cv2.LINE_AA)  # orange ring

    return int(mask.sum())   # return pixel count


# ── Main pipeline ─────────────────────────────────────────────────────────────

def process_video(video_path: str):
    """
    Full arm tracking pipeline for one video:
      Pass 1 — MediaPipe: detect wrist positions in all frames
      Pass 2 — SAM2:      segment arm regions frame by frame
               + draw colored overlays
               + save annotated PNGs
               + build tracking JSON
    """

    video_path = Path(video_path)
    if not video_path.exists():
        print(f"[ERROR] Video not found: {video_path}")
        sys.exit(1)

    if not HOLISTIC_MODEL.exists():
        print(f"[ERROR] Holistic model not found: {HOLISTIC_MODEL}")
        sys.exit(1)

    ensure_sam2()

    clip_name = video_path.stem   # e.g. "WashingCup"

    print(f"\n{'=' * 60}")
    print(f"  Full Arm Tracking Pipeline")
    print(f"  Clip   : {clip_name}")
    print(f"  Video  : {video_path}")
    print(f"{'=' * 60}\n")

    # Get basic video info
    cap          = cv2.VideoCapture(str(video_path))
    fps          = cap.get(cv2.CAP_PROP_FPS)
    full_w       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    full_h       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    print(f"  Resolution : {full_w} × {full_h}  |  {fps:.2f} fps")
    print(f"  Frames     : {total_frames}")
    print(f"  SAM2 at    : {RESIZE_W} × {RESIZE_H} (scaled up for output)\n")

    # Create output directory
    output_dir = TRACKED_ROOT / clip_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Pass 1: MediaPipe wrist detection ─────────────────────
    raw_detections, _, _ = run_mediapipe_pass(video_path, total_frames)

    # ── Build tracking timeline (fill gaps) ───────────────────
    tracking = build_tracking(raw_detections, total_frames)

    # ── Load SAM2 ─────────────────────────────────────────────
    device = get_device()
    print(f"  [SAM2] Loading model (device: {device}) ...")
    sam2_model = build_sam2(SAM2_CFG, str(SAM2_CKPT), device=device)
    predictor  = SAM2ImagePredictor(sam2_model)
    print(f"  [SAM2] Ready.\n")

    # ── Pass 2: SAM2 segmentation ─────────────────────────────
    print(f"  [Pass 2/2] Segmenting arm regions ...")
    print(f"  {'─' * 50}")

    cap        = cv2.VideoCapture(str(video_path))
    frame_id   = 0
    start_time = time.time()

    # Counters for final summary
    frames_with_right = 0   # frames where right arm was segmented
    frames_with_left  = 0   # frames where left arm was segmented
    frames_with_both  = 0   # frames where both arms were segmented
    total_right_px    = 0   # cumulative right arm pixel count
    total_left_px     = 0   # cumulative left arm pixel count

    # This list builds up the per-frame JSON entries
    json_frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Get tracking positions for this frame
        t         = tracking[frame_id]
        left_pt   = t["left"]     # (px,py) or None
        right_pt  = t["right"]    # (px,py) or None
        left_fresh  = t["left_fresh"]
        right_fresh = t["right_fresh"]

        # ── Only run SAM2 if at least one wrist is tracked ────
        right_px_count = 0   # pixels in right arm mask this frame
        left_px_count  = 0   # pixels in left arm mask this frame
        right_tracked  = False
        left_tracked   = False

        if left_pt or right_pt:
            # Resize frame for SAM2 (much faster than full 4K)
            frame_small = cv2.resize(frame, (RESIZE_W, RESIZE_H),
                                     interpolation=cv2.INTER_AREA)
            frame_rgb   = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)

            # Encode the frame into SAM2 feature space ONCE
            # (this is the expensive step; we call predict() once per arm after)
            with torch.no_grad():
                predictor.set_image(frame_rgb)

            # Annotated frame starts as a copy of the FULL resolution frame
            annotated = frame.copy()

            # ── Segment right arm ─────────────────────────────
            if right_pt:
                mask = segment_arm(predictor, frame_small,
                                   right_pt[0], right_pt[1],
                                   full_w, full_h, frame)
                if mask is not None and mask.any():
                    right_tracked    = True
                    right_px_count   = draw_overlay(
                        annotated, mask, COLOR_RIGHT, right_pt, right_fresh
                    )
                    total_right_px  += right_px_count
                    frames_with_right += 1

            # ── Segment left arm ──────────────────────────────
            if left_pt:
                mask = segment_arm(predictor, frame_small,
                                   left_pt[0], left_pt[1],
                                   full_w, full_h, frame)
                if mask is not None and mask.any():
                    left_tracked    = True
                    left_px_count   = draw_overlay(
                        annotated, mask, COLOR_LEFT, left_pt, left_fresh
                    )
                    total_left_px  += left_px_count
                    frames_with_left += 1

            if right_tracked and left_tracked:
                frames_with_both += 1

        else:
            # No arm tracked this frame — save original frame unmodified
            annotated = frame.copy()

        # ── Save annotated frame as PNG ────────────────────────
        fname = f"frame_{str(frame_id).zfill(6)}.png"
        cv2.imwrite(str(output_dir / fname), annotated)

        # ── Build JSON entry for this frame ───────────────────
        json_frames.append({
            "frame_id":      frame_id,
            "timestamp_sec": round(frame_id / fps, 4),
            "right_arm": {
                "tracked":     right_tracked,
                "wrist":       {"px": right_pt[0], "py": right_pt[1]} if right_pt else None,
                "wrist_fresh": right_fresh,
                "pixel_count": right_px_count,
            },
            "left_arm": {
                "tracked":     left_tracked,
                "wrist":       {"px": left_pt[0], "py": left_pt[1]} if left_pt else None,
                "wrist_fresh": left_fresh,
                "pixel_count": left_px_count,
            },
        })

        # ── Progress ───────────────────────────────────────────
        if frame_id % PROGRESS_EVERY == 0 and frame_id > 0:
            elapsed  = time.time() - start_time
            pct      = frame_id / total_frames * 100
            fps_proc = frame_id / elapsed
            eta_sec  = (total_frames - frame_id) / fps_proc if fps_proc > 0 else 0
            print(
                f"  Frame {frame_id:>6} / {total_frames}"
                f"  ({pct:5.1f}%)"
                f"  R:{frames_with_right}  L:{frames_with_left}"
                f"  |  {fps_proc:.1f} fps"
                f"  ETA {eta_sec/60:.1f} min"
            )

        frame_id += 1

    cap.release()

    # ── Save tracking JSON ────────────────────────────────────
    json_path = TRACKED_ROOT / f"{clip_name}_tracking.json"
    with open(json_path, "w") as f:
        json.dump({
            "clip_name":    clip_name,
            "total_frames": frame_id,
            "fps":          fps,
            "resolution":   {"width": full_w, "height": full_h},
            "sam2_resolution": {"width": RESIZE_W, "height": RESIZE_H},
            "summary": {
                "frames_right_tracked": frames_with_right,
                "frames_left_tracked":  frames_with_left,
                "frames_both_tracked":  frames_with_both,
                "avg_right_pixels":     int(total_right_px / max(frames_with_right, 1)),
                "avg_left_pixels":      int(total_left_px  / max(frames_with_left,  1)),
            },
            "frames": json_frames,
        }, f, indent=2)

    # ── Final summary ─────────────────────────────────────────
    elapsed = time.time() - start_time
    print(f"\n{'═' * 60}")
    print(f"  TRACKING COMPLETE  —  {clip_name}")
    print(f"{'═' * 60}")
    print(f"  Total frames          : {frame_id:,}")
    print(f"  Right arm tracked     : {frames_with_right:,}  "
          f"({frames_with_right/frame_id*100:.1f}%)")
    print(f"  Left  arm tracked     : {frames_with_left:,}  "
          f"({frames_with_left/frame_id*100:.1f}%)")
    print(f"  Both  arms tracked    : {frames_with_both:,}  "
          f"({frames_with_both/frame_id*100:.1f}%)")
    print(f"  Avg right arm size    : {int(total_right_px/max(frames_with_right,1)):,} px")
    print(f"  Avg left  arm size    : {int(total_left_px /max(frames_with_left, 1)):,} px")
    print(f"  Time taken            : {elapsed/60:.1f} min")
    print(f"\n  Tracked frames  →  {output_dir}/")
    print(f"  Tracking JSON   →  {json_path}")
    print(f"{'═' * 60}\n")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage  : python pipeline/arm_tracking.py <video_path>")
        print("Example: python pipeline/arm_tracking.py assets/videos/WashingCup.mp4")
        sys.exit(1)

    process_video(sys.argv[1])
