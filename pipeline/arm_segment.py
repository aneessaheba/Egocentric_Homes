"""
arm_segment.py
──────────────
Tests arm segmentation on the first 200 frames of WashingCup.mp4.

How it works:
  1. Load wrist coordinates from the hand pose JSON (or extract them
     inline with MediaPipe if the JSON doesn't exist yet)
  2. For each frame, estimate a second "arm point" below the wrist
     (because the GoPro is top-down, arms enter from the bottom)
  3. Pass BOTH points to SAM2 as foreground prompts so it segments
     the full arm region connecting wrist to elbow
  4. Blend a colored mask over the frame (green = right, red = left)
  5. Save 5 sample frames to assets/processed/arm_test/

Run with:
  python pipeline/arm_segment.py
"""

# ── Imports ───────────────────────────────────────────────────────────────────

import sys            # used to exit with an error code if something goes wrong
import json           # read and write JSON files
import time           # measure how long each step takes
import urllib.request # download the SAM2 checkpoint if missing
from pathlib import Path   # clean cross-platform file path handling

import cv2            # OpenCV — open video, read frames, draw shapes, save PNGs
import numpy as np    # NumPy — array maths for mask operations
import torch          # PyTorch — needed internally by SAM2
import mediapipe as mp   # MediaPipe — used if we need to extract wrist coords inline

# SAM2 Tasks API
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# MediaPipe Tasks API — same as hand_pose.py uses
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision


# ── Configuration ─────────────────────────────────────────────────────────────

VIDEO_PATH     = Path("assets/videos/WashingCup.mp4")  # input video
HAND_POSE_JSON = Path("assets/processed/hand_pose/WashingCup.json")  # keypoints
HOLISTIC_MODEL = Path("assets/models/holistic_landmarker.task")      # MediaPipe model
SAM2_CKPT      = Path("assets/models/sam2_hiera_tiny.pt")            # SAM2 weights
SAM2_CFG       = "sam2_hiera_t.yaml"                                  # SAM2 config
OUTPUT_DIR     = Path("assets/processed/arm_test")                   # save PNGs here
SAM2_CKPT_URL  = "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt"

MAX_FRAMES   = 200           # only process the first 200 frames
SAVE_FRAMES  = [10, 50, 100, 150, 200]   # save annotated PNG at these frame numbers
ARM_OFFSET   = 0.20          # arm point is this fraction of frame height below wrist

# Mask colors in BGR (OpenCV uses Blue, Green, Red order)
# Right arm → green   RGB (0, 200, 0)   = BGR (0, 200, 0)
# Left arm  → red     RGB (200, 0, 0)   = BGR (0, 0, 200)
COLOR_RIGHT_BGR = (0,   200, 0  )   # green in BGR
COLOR_LEFT_BGR  = (0,   0,   200)   # red   in BGR
MASK_ALPHA      = 0.60              # 60% mask color, 40% original frame


# ── Device selection ──────────────────────────────────────────────────────────

def get_device() -> str:
    """
    Pick the fastest available compute device for SAM2.
      cuda → NVIDIA GPU (fastest)
      mps  → Apple Silicon GPU (M1/M2/M3 Mac)
      cpu  → always available, slowest
    """
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ── SAM2 checkpoint download ──────────────────────────────────────────────────

def ensure_sam2_checkpoint():
    """Download the SAM2 tiny checkpoint if it is not already on disk."""
    SAM2_CKPT.parent.mkdir(parents=True, exist_ok=True)   # create assets/models/ if needed
    if SAM2_CKPT.exists():
        print(f"  [SAM2] Checkpoint found: {SAM2_CKPT}")
        return
    print(f"  [SAM2] Downloading checkpoint (~155 MB) ...")
    urllib.request.urlretrieve(SAM2_CKPT_URL, SAM2_CKPT)   # download from Meta
    print(f"  [SAM2] Download complete.\n")


# ── STEP 1 HELPER: extract wrist coords inline with MediaPipe ─────────────────

def extract_wrists_inline(max_frames: int) -> dict:
    """
    If the hand pose JSON is missing or incomplete, run MediaPipe
    HolisticLandmarker on the video to get wrist pixel coordinates.

    Returns a dict: {frame_id: {"left": (px,py) or None,
                                "right": (px,py) or None}}
    """
    print(f"\n  [fallback] Running MediaPipe inline to extract wrist coords ...")

    if not HOLISTIC_MODEL.exists():
        print(f"  [ERROR] Holistic model not found: {HOLISTIC_MODEL}")
        sys.exit(1)

    # Open video
    cap    = cv2.VideoCapture(str(VIDEO_PATH))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Set up the MediaPipe detector
    options = mp_vision.HolisticLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=str(HOLISTIC_MODEL)),
        running_mode=mp_vision.RunningMode.IMAGE,   # one frame at a time
        min_pose_detection_confidence=0.5,
        min_hand_landmarks_confidence=0.5,
    )
    detector = mp_vision.HolisticLandmarker.create_from_options(options)

    wrists     = {}   # {frame_id: {"left": (px,py)|None, "right": (px,py)|None}}
    frame_id   = 0

    while frame_id < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR → RGB for MediaPipe
        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result   = detector.detect(mp_image)

        # ── Use POSE wrist landmarks (more reliable than hand landmarks) ──────────
        # The pose model tracks 33 body joints including wrists at index 15 (left)
        # and index 16 (right). These are detected even when the detailed hand model
        # doesn't fire, making them much more reliable for first-200-frame testing.
        pose = result.pose_landmarks   # list of 33 pose NormalizedLandmark or []

        def pose_wrist(idx):
            """Return (px, py) for pose landmark at idx, or None if not detected."""
            if not pose or idx >= len(pose):
                return None   # pose not detected
            lm = pose[idx]    # the specific body joint
            return (int(lm.x * width), int(lm.y * height))   # normalised → pixels

        wrists[frame_id] = {
            "left":  pose_wrist(15),   # index 15 = LEFT_WRIST in BlazePose
            "right": pose_wrist(16),   # index 16 = RIGHT_WRIST in BlazePose
        }

        frame_id += 1

    cap.release()
    detector.close()
    print(f"  [fallback] Wrist coords extracted for {frame_id} frames.\n")
    return wrists


# ── STEP 1: load or extract wrist coordinates ─────────────────────────────────

def load_wrists(max_frames: int, frame_height: int) -> dict:
    """
    Load wrist pixel coordinates for the first max_frames frames.

    Tries to read from the hand pose JSON first.
    Falls back to inline MediaPipe extraction if the JSON is missing.

    Returns: {frame_id: {"left": (px,py)|None, "right": (px,py)|None}}
    """
    # ── Try loading from JSON ─────────────────────────────────
    if HAND_POSE_JSON.exists():
        print(f"  [1] Loading wrist coords from JSON: {HAND_POSE_JSON}")
        with open(HAND_POSE_JSON, "r") as f:
            pose_data = json.load(f)   # load the full hand pose JSON

        wrists = {}
        for frame in pose_data.get("frames", []):
            fid = frame["frame_id"]   # which frame this entry is for
            if fid >= max_frames:
                break   # we only need the first max_frames frames

            left_px  = None   # will hold (px, py) for left wrist or None
            right_px = None   # will hold (px, py) for right wrist or None

            for hand in frame.get("hands", []):
                wrist_kp = hand.get("keypoints", {}).get("WRIST")   # WRIST keypoint dict
                if wrist_kp:
                    pt = (wrist_kp["px"], wrist_kp["py"])   # (pixel_x, pixel_y)
                    if hand["label"] == "Left":
                        left_px = pt
                    else:
                        right_px = pt

            wrists[fid] = {"left": left_px, "right": right_px}

        print(f"       {len(wrists)} frames loaded from JSON.\n")
        return wrists

    # ── JSON not found — extract inline ───────────────────────
    print(f"  [1] Hand pose JSON not found at {HAND_POSE_JSON}")
    print(f"      Running MediaPipe inline for first {max_frames} frames ...")
    return extract_wrists_inline(max_frames)


# ── STEP 2: calculate arm point below wrist ───────────────────────────────────

def calc_arm_point(wrist_px: int, wrist_py: int, frame_height: int) -> tuple:
    """
    Estimate where the forearm/elbow is.

    Because the GoPro camera is mounted top-down, the person's arms
    enter the frame from the bottom edge. So "further up the arm"
    means further toward the bottom of the image (higher y value).

    We add ARM_OFFSET × frame_height to the wrist y coordinate.
    For a 2160px tall frame: arm_point_y = wrist_py + (2160 × 0.20) = wrist_py + 432px

    Returns (arm_point_x, arm_point_y) as ints clamped to frame bounds.
    """
    arm_x = wrist_px                           # same horizontal position as wrist
    arm_y = wrist_py + int(frame_height * ARM_OFFSET)   # offset downward (toward camera bottom)
    arm_y = min(arm_y, frame_height - 1)       # clamp so we don't go off the bottom edge
    return (arm_x, arm_y)


# ── STEP 3 HELPER: segment one arm region with SAM2 ──────────────────────────

def segment_arm(
    predictor:    SAM2ImagePredictor,  # loaded SAM2 model
    frame_bgr:    np.ndarray,          # current video frame (BGR NumPy array)
    wrist_pt:     tuple,               # (px, py) of wrist — foreground prompt 1
    arm_pt:       tuple,               # (px, py) of arm  — foreground prompt 2
) -> np.ndarray | None:
    """
    Ask SAM2 to segment the arm region using two foreground point prompts.

    By giving it both the wrist point AND a point further up the arm,
    SAM2 tries to find the connected region containing both points.
    This typically gives us the full visible arm, not just the hand.

    Returns a boolean NumPy mask (True = arm pixel) or None on failure.
    """
    # Build the prompt arrays SAM2 expects
    # point_coords: shape (N, 2) — N pixel positions as [[x1,y1], [x2,y2], ...]
    # point_labels: shape (N,)   — 1 = foreground (include this in the mask)
    point_coords = np.array([
        [wrist_pt[0], wrist_pt[1]],   # point 1: wrist position
        [arm_pt[0],   arm_pt[1]  ],   # point 2: estimated arm/elbow position
    ], dtype=np.float32)              # SAM2 expects float32 coordinates

    point_labels = np.array([1, 1], dtype=np.int32)   # both points are foreground

    # Run SAM2 — wrap in no_grad to save memory (we don't need gradients here)
    with torch.no_grad():
        masks, scores, _ = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=False,   # return the single best mask
        )

    # masks[0] is shape (H, W) — convert to bool in case SAM2 returns uint8/float
    return masks[0].astype(bool)


# ── STEP 4 HELPER: draw colored mask + markers on a frame ────────────────────

def draw_arm_overlay(
    frame:    np.ndarray,   # BGR frame to draw on (modified in place)
    mask:     np.ndarray,   # boolean mask, True = arm pixel
    color:    tuple,        # BGR color for this arm
    wrist_pt: tuple,        # (px, py) of wrist — for the dot marker
    arm_pt:   tuple,        # (px, py) of arm   — for the dot marker
    side:     str,          # "R" or "L" — label to show on the arm
) -> int:
    """
    Blend a colored mask over the arm pixels, draw dot markers,
    and add an "R" or "L" label. Returns the number of arm pixels in the mask.
    """

    # ── Blend the colored mask over the original frame ────────────────────────
    # Strategy: paint the arm pixels with the color on a copy (overlay),
    # then mix: result = overlay × alpha + original × (1 - alpha).
    overlay         = frame.copy()   # start from the current frame
    overlay[mask]   = color          # paint every True pixel with arm color

    # cv2.addWeighted blends two images:
    #   dst = overlay × MASK_ALPHA + frame × (1 - MASK_ALPHA)
    cv2.addWeighted(
        overlay,     MASK_ALPHA,          # colored layer (60%)
        frame,       1.0 - MASK_ALPHA,    # original frame (40%)
        0,                                # gamma brightness offset (none)
        frame,                            # write result back into frame (in-place)
    )

    pixel_count = int(mask.sum())   # total number of arm pixels

    # ── Draw a dot at the wrist point ─────────────────────────────────────────
    cv2.circle(frame, wrist_pt, 14, (255, 255, 255), -1, cv2.LINE_AA)  # white circle
    cv2.circle(frame, wrist_pt, 14, color,            2, cv2.LINE_AA)  # colored ring
    cv2.circle(frame, wrist_pt,  7, color,           -1, cv2.LINE_AA)  # colored center

    # ── Draw a dot at the estimated arm/elbow point ───────────────────────────
    cv2.circle(frame, arm_pt, 14, (255, 255, 255), -1, cv2.LINE_AA)    # white circle
    cv2.circle(frame, arm_pt, 14, color,            2, cv2.LINE_AA)    # colored ring
    cv2.circle(frame, arm_pt,  7, (200, 200, 200), -1, cv2.LINE_AA)    # gray center (different from wrist)

    # ── Draw a thin dashed-style line from wrist to arm point ─────────────────
    # This makes it visually clear that both dots belong to the same arm
    cv2.line(frame, wrist_pt, arm_pt, color, 2, cv2.LINE_AA)

    # ── Draw the "R" or "L" label near the wrist ──────────────────────────────
    label_x = wrist_pt[0] + 20   # place label 20px to the right of the wrist dot
    label_y = wrist_pt[1] - 20   # and 20px above

    # Draw black outline first (same trick as before: draw shifted in 4 directions)
    for dx, dy in [(-2,-2),(-2,2),(2,-2),(2,2)]:
        cv2.putText(frame, side, (label_x+dx, label_y+dy),
                    cv2.FONT_HERSHEY_DUPLEX, 1.5, (0,0,0), 4, cv2.LINE_AA)

    # Draw colored label on top
    cv2.putText(frame, side, (label_x, label_y),
                cv2.FONT_HERSHEY_DUPLEX, 1.5, color, 2, cv2.LINE_AA)

    return pixel_count   # return count so we can print it in the summary


# ── Main test function ────────────────────────────────────────────────────────

def run_arm_test():
    """
    Full test pipeline:
      1. Load wrist coords for first MAX_FRAMES frames
      2. Load SAM2
      3. Loop over first MAX_FRAMES frames:
         - calculate arm point
         - segment with SAM2 (2 prompts per arm)
         - save sample frames with colored overlays
      4. Print report for each saved frame
    """

    print(f"\n{'=' * 60}")
    print(f"  Arm Segmentation Test  (SAM2 + 2-point prompts)")
    print(f"  Video  : {VIDEO_PATH}")
    print(f"  Frames : first {MAX_FRAMES}")
    print(f"  Saving : frames {SAVE_FRAMES}")
    print(f"{'=' * 60}\n")

    # Sanity checks before we do anything expensive
    if not VIDEO_PATH.exists():
        print(f"[ERROR] Video not found: {VIDEO_PATH}")
        sys.exit(1)

    # ── Open video to get dimensions ──────────────────────────
    cap    = cv2.VideoCapture(str(VIDEO_PATH))   # open the video file
    fps    = cap.get(cv2.CAP_PROP_FPS)            # frame rate (e.g. 29.97)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))   # e.g. 3840
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # e.g. 2160

    print(f"  Video  : {width} x {height}  |  {fps} fps\n")

    # ── STEP 1: get wrist coordinates ─────────────────────────
    wrists = load_wrists(MAX_FRAMES, height)
    # wrists is now: {0: {"left": (px,py)|None, "right": (px,py)|None}, 1: ..., ...}

    # ── Download SAM2 checkpoint if missing ───────────────────
    ensure_sam2_checkpoint()

    # ── Load SAM2 model ───────────────────────────────────────
    device = get_device()
    print(f"  [SAM2] Loading model  (device: {device}) ...")
    sam2_model  = build_sam2(SAM2_CFG, str(SAM2_CKPT), device=device)
    predictor   = SAM2ImagePredictor(sam2_model)
    print(f"  [SAM2] Model ready.\n")

    # Create the output folder
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── STEP 3 + 4: loop over first MAX_FRAMES frames ─────────
    print(f"  Processing frames 0 – {MAX_FRAMES - 1} ...")
    print(f"  {'─' * 50}")

    frame_id   = 0    # current frame counter
    start_time = time.time()

    while frame_id < MAX_FRAMES:

        ret, frame = cap.read()   # read the next frame from the video
        if not ret:
            break   # video ended before MAX_FRAMES — stop

        # ── Get wrist coords for this frame ───────────────────
        # wrists[frame_id] was filled in Step 1.
        # It contains {"left": (px,py)|None, "right": (px,py)|None}
        frame_wrists = wrists.get(frame_id, {"left": None, "right": None})
        left_wrist   = frame_wrists["left"]    # (px,py) or None
        right_wrist  = frame_wrists["right"]   # (px,py) or None

        # We only need to run SAM2 on frames where at least one wrist is visible
        # AND on the frames we want to save
        # display_num is 1-indexed (frame_id 9 = "frame 10")
        display_num = frame_id + 1   # human-friendly frame number

        # Check whether this is one of our 5 save frames
        should_save = display_num in SAVE_FRAMES

        # Skip SAM2 (slow) on frames we don't need to save
        if not should_save:
            frame_id += 1
            continue   # jump to next frame without running SAM2

        # ── Prepare for SAM2 ─────────────────────────────────
        # SAM2 expects RGB; OpenCV gives us BGR, so we swap channels
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Encode the frame into SAM2's feature space.
        # This runs the image encoder (ViT) — expensive, done once per frame.
        with torch.no_grad():
            predictor.set_image(frame_rgb)

        # Start building an annotated copy of the frame
        annotated = frame.copy()   # copy so we keep the original clean

        right_segmented  = False   # did SAM2 find a right arm mask?
        left_segmented   = False   # did SAM2 find a left arm mask?
        right_pixel_count = 0      # how many pixels in right arm mask
        left_pixel_count  = 0      # how many pixels in left arm mask

        # ── STEP 2 + 3: segment right arm ────────────────────
        if right_wrist:   # only if we have a right wrist position
            # Calculate the arm/elbow estimate point (below wrist = toward bottom)
            right_arm_pt = calc_arm_point(right_wrist[0], right_wrist[1], height)

            # Run SAM2 with 2 foreground prompts
            mask = segment_arm(predictor, frame, right_wrist, right_arm_pt)

            if mask is not None and mask.any():   # .any() = True if at least one pixel is True
                right_segmented   = True
                # Draw the green mask + wrist dot + arm dot + "R" label
                right_pixel_count = draw_arm_overlay(
                    frame     = annotated,
                    mask      = mask,
                    color     = COLOR_RIGHT_BGR,
                    wrist_pt  = right_wrist,
                    arm_pt    = right_arm_pt,
                    side      = "R",
                )

        # ── STEP 2 + 3: segment left arm ─────────────────────
        if left_wrist:   # only if we have a left wrist position
            left_arm_pt = calc_arm_point(left_wrist[0], left_wrist[1], height)

            mask = segment_arm(predictor, frame, left_wrist, left_arm_pt)

            if mask is not None and mask.any():
                left_segmented   = True
                left_pixel_count = draw_arm_overlay(
                    frame     = annotated,
                    mask      = mask,
                    color     = COLOR_LEFT_BGR,
                    wrist_pt  = left_wrist,
                    arm_pt    = left_arm_pt,
                    side      = "L",
                )

        # ── STEP 4: save this frame and print its report ──────
        out_path = OUTPUT_DIR / f"frame_{str(display_num).zfill(3)}.png"
        cv2.imwrite(str(out_path), annotated)   # save the annotated PNG

        elapsed = time.time() - start_time

        # Print the per-frame detection report
        yes_no = lambda b: "Yes" if b else "No"
        print(f"\n  Frame {display_num:>3}  ({elapsed:.1f}s)  →  {out_path.name}")
        print(f"    Right arm segmented : {yes_no(right_segmented):<3}  "
              f"pixels: {right_pixel_count:>6,}")
        print(f"    Left  arm segmented : {yes_no(left_segmented):<3}  "
              f"pixels: {left_pixel_count:>6,}")

        frame_id += 1   # advance frame counter

    # ── Release resources ─────────────────────────────────────
    cap.release()   # close the video file

    # ── Final summary ─────────────────────────────────────────
    total_time = time.time() - start_time
    print(f"\n  {'─' * 50}")
    print(f"  TEST COMPLETE")
    print(f"  {'─' * 50}")
    print(f"  Frames processed : {frame_id}")
    print(f"  Time taken       : {total_time:.1f}s")
    print(f"  Output PNGs      → {OUTPUT_DIR}/")
    print()


# ── Entry point ───────────────────────────────────────────────────────────────

# Only runs when this file is executed directly:
#   python pipeline/arm_segment.py
if __name__ == "__main__":
    run_arm_test()
