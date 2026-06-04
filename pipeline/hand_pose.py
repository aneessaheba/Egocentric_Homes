"""
hand_pose.py
────────────
Reads every frame of a video, detects hands AND arms using
MediaPipe HolisticLandmarker (Tasks API, MediaPipe 0.10+).

For each frame it extracts:
  • Arm keypoints  (from pose landmarks):
      shoulder, elbow, wrist — per left and right arm
  • Finger keypoints (from hand landmarks):
      all 21 points per hand (same as before)

Draws a connected arm + hand skeleton on each frame and saves:
  • Annotated PNGs  →  assets/processed/annotated/<clip>/
  • Keypoint JSON   →  assets/processed/hand_pose/<clip>.json

Usage:
  python pipeline/hand_pose.py assets/videos/WashingCup.mp4
"""

# ── Imports ───────────────────────────────────────────────────────────────────

import sys           # read the command-line argument (video path)
import json          # write Python dicts as .json files
import time          # measure elapsed time
from pathlib import Path   # cross-platform file path handling

import cv2           # OpenCV — open videos, read frames, save images
import numpy as np   # NumPy — needed for drawing and array operations

import mediapipe as mp   # MediaPipe — holistic landmark detection

# MediaPipe Tasks API — modern interface for MediaPipe 0.10+
from mediapipe.tasks import python as mp_tasks        # BaseOptions, etc.
from mediapipe.tasks.python import vision as mp_vision  # HolisticLandmarker


# ── Model path ────────────────────────────────────────────────────────────────

# Path to the holistic landmarker model file.
# Download with:
#   curl -L -o assets/models/holistic_landmarker.task \
#     https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task
MODEL_PATH = Path("assets/models/holistic_landmarker.task")


# ── Output folders ────────────────────────────────────────────────────────────

ANNOTATED_ROOT = Path("assets/processed/annotated")   # annotated frame PNGs
HAND_POSE_ROOT = Path("assets/processed/hand_pose")   # keypoint JSON files


# ── Pose landmark indices ─────────────────────────────────────────────────────

# MediaPipe's pose model returns 33 body landmarks numbered 0–32.
# We only need the 6 upper-body landmarks for arm tracking.
# These index numbers are defined by MediaPipe's BlazePose model.
POSE_LEFT_SHOULDER  = 11   # left shoulder joint
POSE_RIGHT_SHOULDER = 12   # right shoulder joint
POSE_LEFT_ELBOW     = 13   # left elbow joint
POSE_RIGHT_ELBOW    = 14   # right elbow joint
POSE_LEFT_WRIST     = 15   # left wrist (where arm meets hand)
POSE_RIGHT_WRIST    = 16   # right wrist (where arm meets hand)


# ── Hand landmark name list ───────────────────────────────────────────────────

# MediaPipe hand model returns 21 landmarks numbered 0–20.
# This list maps each index to its human-readable name.
LANDMARK_NAMES = [
    "WRIST",             # 0  — base of palm / wrist joint
    "THUMB_CMC",         # 1  — thumb base
    "THUMB_MCP",         # 2  — thumb first knuckle
    "THUMB_IP",          # 3  — thumb second knuckle
    "THUMB_TIP",         # 4  — thumb tip
    "INDEX_FINGER_MCP",  # 5  — index finger base knuckle
    "INDEX_FINGER_PIP",  # 6  — index first bend
    "INDEX_FINGER_DIP",  # 7  — index second bend
    "INDEX_FINGER_TIP",  # 8  — index fingertip
    "MIDDLE_FINGER_MCP", # 9  — middle finger base knuckle
    "MIDDLE_FINGER_PIP", # 10 — middle first bend
    "MIDDLE_FINGER_DIP", # 11 — middle second bend
    "MIDDLE_FINGER_TIP", # 12 — middle fingertip
    "RING_FINGER_MCP",   # 13 — ring finger base knuckle
    "RING_FINGER_PIP",   # 14 — ring first bend
    "RING_FINGER_DIP",   # 15 — ring second bend
    "RING_FINGER_TIP",   # 16 — ring fingertip
    "PINKY_MCP",         # 17 — pinky base knuckle
    "PINKY_PIP",         # 18 — pinky first bend
    "PINKY_DIP",         # 19 — pinky second bend
    "PINKY_TIP",         # 20 — pinky tip
]

# Which hand landmark indices to connect with lines to form the skeleton.
# Each tuple (a, b) = draw a line from landmark a to landmark b.
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),         # thumb
    (0,5),(5,6),(6,7),(7,8),         # index finger
    (0,9),(9,10),(10,11),(11,12),    # middle finger
    (0,13),(13,14),(14,15),(15,16),  # ring finger
    (0,17),(17,18),(18,19),(19,20),  # pinky
    (5,9),(9,13),(13,17),            # palm crossbars
]


# ── Drawing colors (BGR format for OpenCV) ────────────────────────────────────

# OpenCV uses BGR (Blue, Green, Red) — note the reversed order vs RGB.
COLOR_ARM_RIGHT  = (200, 100, 255)   # soft purple — right arm line
COLOR_ARM_LEFT   = (100, 100, 255)   # soft red    — left arm line
COLOR_BONE_RIGHT = (200, 180, 255)   # light purple — right hand bones
COLOR_DOT_RIGHT  = (255,   0, 255)   # bright magenta — right hand dots
COLOR_BONE_LEFT  = (100, 100, 255)   # light red    — left hand bones
COLOR_DOT_LEFT   = (  0,   0, 255)   # bright red   — left hand dots


# ── Helper: normalised landmark → pixel coords ────────────────────────────────

def lm_to_px(lm, width: int, height: int) -> tuple:
    """
    Convert a MediaPipe NormalizedLandmark to integer pixel coordinates.

    MediaPipe returns x and y as fractions of the image size (0.0 to 1.0).
    Multiplying by width/height gives the actual pixel position.

    lm     — a NormalizedLandmark object with .x and .y attributes
    width  — frame width in pixels
    height — frame height in pixels
    Returns (px_x, px_y) as a tuple of ints
    """
    return (int(lm.x * width), int(lm.y * height))


# ── Helper: draw the arm + hand skeleton on a frame ──────────────────────────

def draw_skeleton(
    frame:       np.ndarray,   # the image to draw on (modified in place)
    arm:         dict,         # arm dict: {"shoulder": (px,py), "elbow": ..., "wrist": ...}
    hand_pts:    list,         # list of (px, py) for all 21 hand landmarks
    arm_color:   tuple,        # BGR color for the arm line
    bone_color:  tuple,        # BGR color for hand bones
    dot_color:   tuple,        # BGR color for hand joint dots
):
    """
    Draw the full arm-to-fingertip skeleton in one connected visual:

    1. Arm segment:  shoulder ──── elbow ──── wrist  (thick line)
    2. Arm-to-hand:  pose wrist ──── hand WRIST       (connecting line)
    3. Hand bones:   all HAND_CONNECTIONS as thin lines
    4. Joint dots:   filled circles at every keypoint
    """

    # ── 1. Draw the arm line: shoulder → elbow → wrist ───────────────────────
    shoulder_pt = arm.get("shoulder")   # (px, py) tuple or None
    elbow_pt    = arm.get("elbow")
    wrist_pt    = arm.get("wrist")      # pose wrist (not hand WRIST)

    # Only draw if all three arm points are available
    if shoulder_pt and elbow_pt:
        # Shoulder to elbow
        cv2.line(frame, shoulder_pt, elbow_pt, arm_color, 4, cv2.LINE_AA)

    if elbow_pt and wrist_pt:
        # Elbow to wrist
        cv2.line(frame, elbow_pt, wrist_pt, arm_color, 4, cv2.LINE_AA)

    # Draw filled circles at each arm joint so they stand out
    for pt in [shoulder_pt, elbow_pt, wrist_pt]:
        if pt:
            cv2.circle(frame, pt, 8, arm_color, -1, cv2.LINE_AA)   # outer circle
            cv2.circle(frame, pt, 4, (255,255,255), -1, cv2.LINE_AA)  # white center

    # ── 2. Connect pose wrist to hand WRIST keypoint ─────────────────────────
    # hand_pts[0] is the WRIST landmark (index 0 = WRIST in LANDMARK_NAMES).
    # Connecting the arm's pose wrist to the hand's wrist makes them one skeleton.
    if wrist_pt and hand_pts and hand_pts[0]:
        cv2.line(frame, wrist_pt, hand_pts[0], arm_color, 3, cv2.LINE_AA)

    # ── 3. Draw hand bone lines ───────────────────────────────────────────────
    for start_idx, end_idx in HAND_CONNECTIONS:
        p1 = hand_pts[start_idx]   # start point of this bone
        p2 = hand_pts[end_idx]     # end point of this bone
        if p1 and p2:              # only draw if both points exist
            cv2.line(frame, p1, p2, bone_color, 2, cv2.LINE_AA)

    # ── 4. Draw joint dots at every hand keypoint ─────────────────────────────
    for pt in hand_pts:
        if pt:
            cv2.circle(frame, pt, 6, bone_color, -1, cv2.LINE_AA)   # outer ring
            cv2.circle(frame, pt, 3, dot_color,  -1, cv2.LINE_AA)   # bright center


# ── Main function ─────────────────────────────────────────────────────────────

def process_video(video_path: str):
    """
    Full pipeline for one video file.

    Steps:
      1. Open video
      2. Set up MediaPipe HolisticLandmarker
      3. Process every frame: detect arms + hands
      4. Draw connected skeleton (shoulder → finger tips)
      5. Save annotated PNG for each frame
      6. Collect all keypoints into a JSON file
      7. Print progress and final summary
    """

    video_path = Path(video_path)   # convert string path to Path object

    # Make sure the video file actually exists before we do anything
    if not video_path.exists():
        print(f"[ERROR] File not found: {video_path}")
        sys.exit(1)

    # Verify the model file is present
    if not MODEL_PATH.exists():
        print(f"[ERROR] Model not found: {MODEL_PATH}")
        print("  Download with:")
        print("  curl -L -o assets/models/holistic_landmarker.task \\")
        print("    https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task")
        sys.exit(1)

    clip_name = video_path.stem   # e.g. "WashingCup" (filename without extension)

    print(f"\n{'=' * 60}")
    print(f"  Hand Pose Pipeline  (MediaPipe Holistic)")
    print(f"  Clip : {clip_name}")
    print(f"{'=' * 60}\n")

    # ── Step 1: open the video ────────────────────────────────
    cap = cv2.VideoCapture(str(video_path))   # open the video file

    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {video_path}")
        sys.exit(1)

    fps          = cap.get(cv2.CAP_PROP_FPS)                  # e.g. 29.97
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))     # pixel width
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))    # pixel height
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))     # total frames

    print(f"  Resolution : {width} x {height}")
    print(f"  FPS        : {fps}")
    print(f"  Frames     : {total_frames}\n")

    # ── Step 2: create output folders ────────────────────────
    annotated_dir = ANNOTATED_ROOT / clip_name   # e.g. .../annotated/WashingCup/
    annotated_dir.mkdir(parents=True, exist_ok=True)
    HAND_POSE_ROOT.mkdir(parents=True, exist_ok=True)

    # ── Step 3: set up MediaPipe HolisticLandmarker ───────────
    # BaseOptions tells MediaPipe where to find the model weights file
    base_opts = mp_tasks.BaseOptions(model_asset_path=str(MODEL_PATH))

    # HolisticLandmarkerOptions configures the detector
    options = mp_vision.HolisticLandmarkerOptions(
        base_options=base_opts,
        running_mode=mp_vision.RunningMode.IMAGE,   # process one frame at a time
        min_pose_detection_confidence=0.5,    # min confidence to detect a body
        min_pose_landmarks_confidence=0.5,    # min confidence to track body landmarks
        min_hand_landmarks_confidence=0.5,    # min confidence to detect a hand
    )

    # Create the detector from our options
    detector = mp_vision.HolisticLandmarker.create_from_options(options)

    # ── Step 4: set up counters ───────────────────────────────
    frames_data   = []       # will hold one dict per frame for the JSON
    frame_id      = 0        # current frame number (0-indexed)
    count_2_hands = 0        # frames where both hands were detected
    count_1_hand  = 0        # frames where only one hand was detected
    count_0_hands = 0        # frames where no hands were detected
    start_time    = time.time()

    # ── Step 5: loop over every frame ────────────────────────
    while True:

        ret, frame = cap.read()   # read the next frame from the video
        if not ret:
            break                  # no more frames — exit the loop

        # Convert BGR (OpenCV) to RGB (MediaPipe expects RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Wrap the NumPy array in a MediaPipe Image object
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,   # standard RGB colour space
            data=frame_rgb,                      # the pixel array
        )

        # Run the holistic detector — returns pose + left/right hand landmarks
        result = detector.detect(mp_image)

        # ── Extract pose arm landmarks ────────────────────────
        # result.pose_landmarks is a list of 33 NormalizedLandmark objects.
        # It is an empty list [] if no body was detected in this frame.
        pose_lms = result.pose_landmarks   # list of 33 pose landmarks (or empty)

        # Helper: safely get one pose landmark as (px, py) or None if not detected
        def get_pose_pt(idx):
            """Return pixel coords for pose landmark at index idx, or None."""
            if not pose_lms or idx >= len(pose_lms):
                return None   # pose not detected or index out of range
            return lm_to_px(pose_lms[idx], width, height)

        # Extract the 6 arm points we need (3 per side)
        left_shoulder  = get_pose_pt(POSE_LEFT_SHOULDER)   # left shoulder joint
        left_elbow     = get_pose_pt(POSE_LEFT_ELBOW)      # left elbow joint
        left_wrist_pose = get_pose_pt(POSE_LEFT_WRIST)     # left wrist (from pose)
        right_shoulder  = get_pose_pt(POSE_RIGHT_SHOULDER) # right shoulder joint
        right_elbow     = get_pose_pt(POSE_RIGHT_ELBOW)    # right elbow joint
        right_wrist_pose = get_pose_pt(POSE_RIGHT_WRIST)   # right wrist (from pose)

        # Pack arm data into dicts for easy use later
        left_arm  = {"shoulder": left_shoulder,  "elbow": left_elbow,  "wrist": left_wrist_pose}
        right_arm = {"shoulder": right_shoulder, "elbow": right_elbow, "wrist": right_wrist_pose}

        # ── Extract hand finger landmarks ─────────────────────
        # HolisticLandmarker separates left and right hands directly —
        # no handedness classification needed like in HandLandmarker.
        # Each is a list of 21 NormalizedLandmark objects, or [] if not detected.
        left_hand_lms  = result.left_hand_landmarks    # 21 pts or []
        right_hand_lms = result.right_hand_landmarks   # 21 pts or []

        # Helper: convert a list of 21 hand NormalizedLandmarks to pixel tuples
        def hand_to_pts(lms):
            """Convert 21 hand NormalizedLandmarks → list of (px,py) tuples."""
            if not lms:
                return []   # no hand detected — return empty list
            return [lm_to_px(lm, width, height) for lm in lms]

        left_pts  = hand_to_pts(left_hand_lms)    # list of 21 (px,py) or []
        right_pts = hand_to_pts(right_hand_lms)   # list of 21 (px,py) or []

        # Count how many hands were actually detected this frame
        left_detected  = len(left_pts) == 21    # True if left hand found
        right_detected = len(right_pts) == 21   # True if right hand found
        num_hands = int(left_detected) + int(right_detected)   # 0, 1, or 2

        # Update running hand counts for the final summary
        if num_hands == 2:
            count_2_hands += 1
        elif num_hands == 1:
            count_1_hand += 1
        else:
            count_0_hands += 1

        # ── Draw skeleton on a copy of the frame ─────────────
        annotated_frame = frame.copy()   # copy so we don't modify the original

        # Draw LEFT arm + hand skeleton
        if left_detected:
            draw_skeleton(
                frame      = annotated_frame,
                arm        = left_arm,
                hand_pts   = left_pts,
                arm_color  = COLOR_ARM_LEFT,
                bone_color = COLOR_BONE_LEFT,
                dot_color  = COLOR_DOT_LEFT,
            )

        # Draw RIGHT arm + hand skeleton
        if right_detected:
            draw_skeleton(
                frame      = annotated_frame,
                arm        = right_arm,
                hand_pts   = right_pts,
                arm_color  = COLOR_ARM_RIGHT,
                bone_color = COLOR_BONE_RIGHT,
                dot_color  = COLOR_DOT_RIGHT,
            )

        # ── Save annotated frame as PNG ───────────────────────
        fname = f"frame_{str(frame_id).zfill(6)}.png"   # e.g. frame_000042.png
        cv2.imwrite(str(annotated_dir / fname), annotated_frame)

        # ── Build keypoint JSON for this frame ────────────────
        hands_list = []   # will hold one dict per detected hand

        # Helper: build the 21-keypoint dict from a list of (px,py) tuples
        def build_keypoints(pts, lms):
            """
            Create a dict mapping landmark name → coordinate data.
            pts  = list of (px, py) pixel tuples
            lms  = list of NormalizedLandmark objects (for x, y, z)
            """
            kp = {}
            for idx, (pt, lm) in enumerate(zip(pts, lms)):
                name = LANDMARK_NAMES[idx]   # human-readable landmark name
                kp[name] = {
                    "x":  round(lm.x, 6),   # normalised x (0.0–1.0)
                    "y":  round(lm.y, 6),   # normalised y (0.0–1.0)
                    "z":  round(lm.z, 6),   # relative depth
                    "px": pt[0],             # pixel x coordinate
                    "py": pt[1],             # pixel y coordinate
                }
            return kp

        # Helper: build the arm dict (pixel coords only — no normalised needed)
        def build_arm(arm_dict):
            """
            Convert arm dict of (px,py) tuples to JSON-safe dicts.
            Returns {"shoulder": {"px":..,"py":..}, "elbow":..., "wrist":...}
            """
            result_arm = {}
            for key, pt in arm_dict.items():   # key = "shoulder"/"elbow"/"wrist"
                if pt:
                    result_arm[key] = {"px": pt[0], "py": pt[1]}
                else:
                    result_arm[key] = None   # point not detected
            return result_arm

        # Add left hand if detected
        if left_detected:
            hands_list.append({
                "label":      "Left",
                "confidence": round(float(left_hand_lms[0].visibility or 1.0), 4),
                "arm":        build_arm(left_arm),
                "keypoints":  build_keypoints(left_pts, left_hand_lms),
            })

        # Add right hand if detected
        if right_detected:
            hands_list.append({
                "label":      "Right",
                "confidence": round(float(right_hand_lms[0].visibility or 1.0), 4),
                "arm":        build_arm(right_arm),
                "keypoints":  build_keypoints(right_pts, right_hand_lms),
            })

        # Add this frame's data to the master list
        frames_data.append({
            "frame_id":       frame_id,
            "timestamp_sec":  round(frame_id / fps, 4),   # time in seconds
            "hands_detected": num_hands,
            "hands":          hands_list,
        })

        # ── Print progress every 100 frames ──────────────────
        if frame_id % 100 == 0:
            elapsed  = time.time() - start_time
            pct_done = (frame_id / total_frames) * 100
            print(
                f"  Frame {frame_id:>6} / {total_frames}"
                f"  ({pct_done:5.1f}%)"
                f"  |  hands: 2={count_2_hands}  1={count_1_hand}  0={count_0_hands}"
                f"  |  {elapsed:.1f}s"
            )

        frame_id += 1   # advance frame counter

    # ── Step 6: release resources ─────────────────────────────
    cap.release()      # close the video file
    detector.close()   # free MediaPipe model memory

    # ── Step 7: write the keypoint JSON ───────────────────────
    output_json = {
        "clip_name":    clip_name,       # e.g. "WashingCup"
        "total_frames": frame_id,        # number of frames processed
        "fps":          fps,             # original frame rate
        "resolution": {
            "width":  width,
            "height": height,
        },
        "frames": frames_data,           # list of per-frame dicts
    }

    json_path = HAND_POSE_ROOT / f"{clip_name}.json"
    with open(json_path, "w") as f:
        json.dump(output_json, f, indent=2)   # indent=2 makes it readable

    # ── Step 8: print final summary ───────────────────────────
    total_time        = time.time() - start_time
    frames_with_hands = count_1_hand + count_2_hands
    detection_rate    = (frames_with_hands / frame_id) * 100

    print(f"\n{'─' * 60}")
    print(f"  SUMMARY  —  {clip_name}")
    print(f"{'─' * 60}")
    print(f"  Total frames processed : {frame_id}")
    print(f"  Frames with hands      : {frames_with_hands}")
    print(f"  Frames with 2 hands    : {count_2_hands}")
    print(f"  Frames with 1 hand     : {count_1_hand}")
    print(f"  Frames with 0 hands    : {count_0_hands}")
    print(f"  Detection rate         : {detection_rate:.1f}%")
    print(f"  Time taken             : {total_time:.1f}s")
    print(f"\n  Annotated frames  ->  {annotated_dir}/")
    print(f"  Keypoint JSON     ->  {json_path}")
    print()


# ── Entry point ───────────────────────────────────────────────────────────────

# This block only runs when you execute this file directly:
#   python pipeline/hand_pose.py assets/videos/WashingCup.mp4
# It does NOT run when another script does "import hand_pose".
if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage  : python pipeline/hand_pose.py <path_to_video>")
        print("Example: python pipeline/hand_pose.py assets/videos/WashingCup.mp4")
        sys.exit(1)

    process_video(sys.argv[1])
