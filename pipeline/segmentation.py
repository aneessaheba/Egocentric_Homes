"""
segmentation.py
───────────────
SAM2-based hand + active-object segmentation for egocentric household
activity videos.

Unlike the old SAM3 text-prompted version, this script never invents a
prompt out of thin air — every box fed to SAM2 is derived directly from
hand_pose.py's own detected wrist/keypoint data:

  • "hand" masks          — box-prompted from the tight bounding box around
                             that hand's 21 MediaPipe keypoints.
  • "active_object" masks — box-prompted from a DILATED version of the same
                             hand box (see OBJECT_BOX_SCALE below). The
                             object being manipulated is usually adjacent to
                             or partially occluded by the hand rather than
                             strictly inside its keypoint hull, so the object
                             prompt needs extra margin around the hand box.

Both hands are segmented independently when 2 are present in a frame, each
with its own hand + active-object pair.

Processes every PROCESS_EVERY_N-th frame with SAM2; copies (re-saves under
new filenames, not symlinks) the same mask arrays for the frames in between,
so every frame still has real PNG files on disk, never a fabricated value.

Outputs:
  • colored preview PNG per frame  →  assets/processed/segmented/<clip_name>/
  • binary mask PNGs per frame     →  assets/processed/masks/<clip_name>/
  • updated hand pose JSON         →  assets/processed/hand_pose/<clip_name>.json
    ("segmentation" block added to every frame entry)

Usage:
  python pipeline/segmentation.py assets/videos/WashingCup.mp4
"""

import sys
import json
import time
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


# ── Paths ─────────────────────────────────────────────────────────────────────

HAND_POSE_ROOT = Path("assets/processed/hand_pose")
SEGMENTED_ROOT = Path("assets/processed/segmented")    # colored preview frames
MASKS_ROOT     = Path("assets/processed/masks")          # real binary mask PNGs

# ── Model ─────────────────────────────────────────────────────────────────────

# The pip "sam2" package (==1.1.0, already in requirements.txt — it was
# pinned but unused while segmentation.py ran SAM3 instead) ships SAM 2.1
# configs/checkpoints aren't bundled; the checkpoint is downloaded once into
# assets/models/. Hiera-Tiny is the smallest/fastest variant.
SAM2_CONFIG     = "configs/sam2.1/sam2.1_hiera_t.yaml"
SAM2_CHECKPOINT = Path("assets/models/sam2.1_hiera_tiny.pt")
SAM2_CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt"
SAM2_METHOD_NAME = "SAM2.1-Hiera-Tiny"   # exact checkpoint version, for the JSON "method" field

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"   # Apple Silicon GPU, else CPU

# ── Colors (BGR) — solid for hands, semi-transparent for objects ──────────────

COLOR_HAND_LEFT    = (50,  50,  230)   # red
COLOR_HAND_RIGHT   = (230, 80,  50 )   # blue
COLOR_OBJECT       = (0,   210, 255)   # amber — same hue regardless of hand side
OBJECT_ALPHA       = 0.5               # object overlay is translucent so the
                                        # hand mask underneath stays visible

# ── Sampling ──────────────────────────────────────────────────────────────────

# SAM2 box-prompting is far cheaper than SAM3's text-prompted inference was
# (~300ms encode + ~16ms/box decode on Apple Silicon MPS, measured directly),
# so frames are sampled more densely than the old SAM3 script did (every 9th).
PROCESS_EVERY_N = 3   # run SAM2 on every 3rd frame; copy result to the 2 frames in between

# ── Box geometry ──────────────────────────────────────────────────────────────

# Hand box = tight bbox around the 21 keypoints, padded a little because the
# keypoint hull tends to sit slightly inside the visible hand silhouette.
HAND_BOX_PAD = 0.15     # 15% padding on each side, relative to the box's own w/h

# Active-object box = the hand box, scaled up around its own center.
# scale=2.5 means the dilated box's width/height are 2.5x the hand box's —
# i.e. each side is pushed out by 0.75x the hand box's own width/height.
# This is a starting guess, not a measured constant — flagged for visual
# sanity-checking against real frames before relying on it further.
OBJECT_BOX_SCALE = 2.5


# ── Model loader ──────────────────────────────────────────────────────────────

def load_sam2() -> SAM2ImagePredictor:
    if not SAM2_CHECKPOINT.exists():
        print(f"[ERROR] SAM2 checkpoint not found: {SAM2_CHECKPOINT}")
        print(f"  Download with:")
        print(f"  curl -L -o {SAM2_CHECKPOINT} {SAM2_CHECKPOINT_URL}")
        sys.exit(1)

    print(f"  [model] Loading {SAM2_METHOD_NAME} ({DEVICE}) ...")
    model = build_sam2(SAM2_CONFIG, str(SAM2_CHECKPOINT), device=DEVICE)
    predictor = SAM2ImagePredictor(model)
    print("  [model] SAM2 ready.\n")
    return predictor


# ── Box geometry helpers ───────────────────────────────────────────────────────

def clip_box(box, frame_w: int, frame_h: int) -> np.ndarray:
    x1, y1, x2, y2 = box
    return np.array([
        max(0.0, x1), max(0.0, y1),
        min(float(frame_w), x2), min(float(frame_h), y2),
    ], dtype=np.float32)


def hand_box_from_wrist(wrist: dict, frame_w: int, frame_h: int) -> np.ndarray:
    """Tight, padded bbox around one hand's 21 MediaPipe keypoints."""
    xs = [kp["px"] for kp in wrist["keypoints"].values()]
    ys = [kp["py"] for kp in wrist["keypoints"].values()]
    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
    w, h = x2 - x1, y2 - y1
    x1, x2 = x1 - w * HAND_BOX_PAD, x2 + w * HAND_BOX_PAD
    y1, y2 = y1 - h * HAND_BOX_PAD, y2 + h * HAND_BOX_PAD
    return clip_box((x1, y1, x2, y2), frame_w, frame_h)


def dilate_box(box: np.ndarray, scale: float, frame_w: int, frame_h: int) -> np.ndarray:
    """Expand `box` to `scale`x its own width/height, centered on the same point."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    new_w, new_h = w * scale, h * scale
    return clip_box((cx - new_w / 2, cy - new_h / 2, cx + new_w / 2, cy + new_h / 2), frame_w, frame_h)


# ── Mask I/O ───────────────────────────────────────────────────────────────────

def write_mask_png(mask_bool: np.ndarray, out_path: Path) -> tuple:
    """Save a boolean mask as a binary grayscale PNG (255=mask, 0=background)."""
    cv2.imwrite(str(out_path), (mask_bool.astype(np.uint8) * 255))
    px  = int(mask_bool.sum())
    cov = round(px / mask_bool.size * 100, 4)
    return px, cov


def mask_entry(category: str, label: str, mask_bool, mask_path: Path) -> dict:
    """Build one entry of the unified `segmentation.masks[]` list."""
    if mask_bool is None:
        return {
            "category": category, "label": label, "found": False,
            "pixel_count": 0, "coverage_pct": 0.0,
            "mask_path": None, "mask_format": None,
        }
    px, cov = write_mask_png(mask_bool, mask_path)
    return {
        "category": category, "label": label, "found": px > 0,
        "pixel_count": px, "coverage_pct": cov,
        "mask_path": str(mask_path), "mask_format": "png",
    }


def paint_mask(img: np.ndarray, mask: np.ndarray, color_bgr: tuple, alpha: float = 1.0) -> np.ndarray:
    """Return a copy of img with mask pixels blended toward color_bgr."""
    out = img.copy()
    overlay = np.full_like(out, color_bgr, dtype=np.uint8)
    blended = cv2.addWeighted(out, 1 - alpha, overlay, alpha, 0)
    out[mask] = blended[mask]
    return out


# ── Per-frame segmentation ────────────────────────────────────────────────────

def segment_frame(predictor, frame_bgr, wrist_data: list, frame_w: int, frame_h: int,
                   masks_dir: Path, frame_id: int):
    """
    Run SAM2 box-prompted segmentation for every hand in `wrist_data`.

    Returns:
      annotated  — preview frame with hand (solid) + object (translucent) masks painted
      seg_entry  — {"method", "prompted_by", "masks": [...]} for this frame's JSON
      raw_masks  — [(category, label, mask_bool, file_suffix), ...] kept in memory
                   so the next PROCESS_EVERY_N-1 frames can re-save the same
                   arrays under their own filenames (real files, not omitted).
    """
    annotated  = frame_bgr.copy()
    masks_list = []
    raw_masks  = []

    if wrist_data:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        predictor.set_image(rgb)

    for wrist in wrist_data:
        label    = wrist["label"]   # "left" or "right"
        hand_box = hand_box_from_wrist(wrist, frame_w, frame_h)

        hand_masks, _, _ = predictor.predict(box=hand_box, multimask_output=False)
        hand_mask = hand_masks[0].astype(bool)

        object_box = dilate_box(hand_box, OBJECT_BOX_SCALE, frame_w, frame_h)
        obj_masks, _, _ = predictor.predict(box=object_box, multimask_output=False)
        obj_mask = obj_masks[0].astype(bool)

        hand_path = masks_dir / f"frame_{frame_id:06d}_hand_{label}.png"
        obj_path  = masks_dir / f"frame_{frame_id:06d}_object_{label}.png"

        masks_list.append(mask_entry("hand", label, hand_mask, hand_path))
        # "near_<label>" — SAM2 box-prompting returns a mask, not a semantic class,
        # so the object has no real name; the label records which hand it's near.
        masks_list.append(mask_entry("active_object", f"near_{label}", obj_mask, obj_path))
        raw_masks.append(("hand", label, hand_mask, f"hand_{label}"))
        raw_masks.append(("active_object", f"near_{label}", obj_mask, f"object_{label}"))

        hand_color = COLOR_HAND_LEFT if label == "left" else COLOR_HAND_RIGHT
        annotated  = paint_mask(annotated, obj_mask,  COLOR_OBJECT, alpha=OBJECT_ALPHA)
        annotated  = paint_mask(annotated, hand_mask, hand_color,   alpha=1.0)

    seg_entry = {"method": SAM2_METHOD_NAME, "prompted_by": "hand_bbox", "masks": masks_list}
    return annotated, seg_entry, raw_masks


def restamp_masks(raw_masks: list, masks_dir: Path, frame_id: int) -> dict:
    """Re-save a previously computed frame's mask arrays under this frame's own
    filenames — keeps every interpolated frame backed by a real file on disk
    instead of pointing multiple frames at one shared path."""
    masks_list = []
    for category, label, mask_bool, suffix in raw_masks:
        path = masks_dir / f"frame_{frame_id:06d}_{suffix}.png"
        masks_list.append(mask_entry(category, label, mask_bool, path))
    return {"method": SAM2_METHOD_NAME, "prompted_by": "hand_bbox", "masks": masks_list}


# ── Main pipeline ─────────────────────────────────────────────────────────────

def process_video(video_path_str: str, output_video: bool = False):
    video_path = Path(video_path_str)
    if not video_path.exists():
        print(f"[ERROR] Video not found: {video_path}")
        sys.exit(1)

    clip_name = video_path.stem

    print(f"\n{'=' * 60}")
    print(f"  Segmentation Pipeline  ({SAM2_METHOD_NAME})")
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

    # ── Load SAM2 model once ──────────────────────────────────
    print("  [2] Loading SAM2 model ...")
    predictor = load_sam2()

    # ── Open video ────────────────────────────────────────────
    print("  [3] Opening video ...")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] Could not open: {video_path}")
        sys.exit(1)

    fps          = cap.get(cv2.CAP_PROP_FPS)
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"       {width}x{height} | {fps:.1f} fps | {total_frames} frames\n")

    # ── Create output folders ──────────────────────────────────
    segmented_dir = SEGMENTED_ROOT / clip_name
    segmented_dir.mkdir(parents=True, exist_ok=True)
    masks_dir = MASKS_ROOT / clip_name
    masks_dir.mkdir(parents=True, exist_ok=True)

    # ── Frame loop ────────────────────────────────────────────
    frame_id   = 0
    start_time = time.time()

    hand_frames = 0    # frames with >=1 hand mask
    obj_frames  = 0    # frames with >=1 active-object mask

    prev_annotated = None
    prev_raw_masks = []   # [] means "no hands were present at the last processed frame"

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        wrist_data = []
        if frame_id in frame_lookup:
            wrist_data = frame_lookup[frame_id].get("hands", {}).get("wrists", [])

        if frame_id % PROCESS_EVERY_N == 0:
            annotated, seg_entry, raw_masks = segment_frame(
                predictor, frame, wrist_data, width, height, masks_dir, frame_id
            )
            prev_annotated = annotated
            prev_raw_masks = raw_masks
        else:
            annotated = prev_annotated if prev_annotated is not None else frame
            seg_entry = restamp_masks(prev_raw_masks, masks_dir, frame_id)

        cv2.imwrite(str(segmented_dir / f"frame_{str(frame_id).zfill(6)}.png"), annotated)

        if frame_id in frame_lookup:
            frame_lookup[frame_id]["segmentation"] = seg_entry

        if any(m["category"] == "hand" and m["found"] for m in seg_entry["masks"]):
            hand_frames += 1
        if any(m["category"] == "active_object" and m["found"] for m in seg_entry["masks"]):
            obj_frames += 1

        if frame_id > 0 and frame_id % 50 == 0:
            elapsed = time.time() - start_time
            print(f"  Frame {frame_id}/{total_frames} | {elapsed:.1f}s elapsed")

        frame_id += 1

    cap.release()

    # ── Save updated JSON ─────────────────────────────────────
    print(f"\n  Saving updated JSON → {json_path}")
    with open(json_path, "w") as f:
        json.dump(pose_data, f, indent=2)

    # ── Final summary ─────────────────────────────────────────
    total_time = time.time() - start_time
    hand_pct   = (hand_frames / frame_id * 100) if frame_id > 0 else 0.0
    obj_pct    = (obj_frames  / frame_id * 100) if frame_id > 0 else 0.0

    print(f"\n{'─' * 60}")
    print(f"  Frames processed        : {frame_id}")
    print(f"  Frames with hand mask   : {hand_frames} ({hand_pct:.1f}%)")
    print(f"  Frames with object mask : {obj_frames} ({obj_pct:.1f}%)")
    print(f"  Time                    : {total_time/60:.1f} min")
    print(f"  Preview frames → {segmented_dir}/")
    print(f"  Mask PNGs     → {masks_dir}/")
    print(f"{'─' * 60}\n")

    # ── Optional: stitch frames into annotated video ──────────
    if output_video:
        out_path = Path("assets/processed") / f"{clip_name}_annotated.mp4"
        png_files = sorted(segmented_dir.glob("frame_*.png"))
        if png_files:
            sample = cv2.imread(str(png_files[0]))
            h, w   = sample.shape[:2]
            writer = cv2.VideoWriter(
                str(out_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (w, h),
            )
            print(f"  Stitching {len(png_files)} frames into video ...")
            for p in png_files:
                writer.write(cv2.imread(str(p)))
            writer.release()
            print(f"  Saved → {out_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='SAM2 Hand + Active-Object Segmentation Pipeline'
    )
    parser.add_argument('video', help='Path to input video file')
    parser.add_argument('--output-video', action='store_true',
                         help='Stitch segmented frames into annotated video')
    args = parser.parse_args()

    process_video(args.video, output_video=args.output_video)
