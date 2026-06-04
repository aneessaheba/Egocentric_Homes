"""
hand_pose.py
────────────
Reads every frame of a video, detects hands using MediaPipe,
draws the 21-keypoint skeleton on each frame, and saves:
  • annotated frame PNGs  →  assets/processed/annotated/<clip_name>/
  • keypoint JSON         →  assets/processed/hand_pose/<clip_name>.json

Usage:
  python pipeline/hand_pose.py assets/videos/WashingCup.mp4
"""

# ── Imports ───────────────────────────────────────────────────────────────────

import sys                          # lets us read the command-line argument (the video path)
import json                         # lets us write Python dicts as .json files
import time                         # used to measure how long the script takes
from pathlib import Path            # modern way to work with file/folder paths

import cv2                          # OpenCV — opens videos, reads frames, saves images
import mediapipe as mp              # MediaPipe — hand detection and landmark tracking


# ── MediaPipe setup ───────────────────────────────────────────────────────────

# mp.solutions.hands gives us the Hands detector class
mp_hands = mp.solutions.hands

# mp.solutions.drawing_utils gives us draw_landmarks() to draw the skeleton
mp_draw = mp.solutions.drawing_utils

# mp.solutions.drawing_styles gives us pre-built colors for the hand skeleton
mp_styles = mp.solutions.drawing_styles


# ── Landmark name lookup ──────────────────────────────────────────────────────

# MediaPipe returns 21 landmarks numbered 0–20.
# This list maps each number to its human-readable name.
# Index position in the list = landmark ID (so index 0 = "WRIST", etc.)
LANDMARK_NAMES = [
    "WRIST",                # 0  — base of the hand
    "THUMB_CMC",            # 1  — thumb base (carpometacarpal)
    "THUMB_MCP",            # 2  — thumb first knuckle (metacarpophalangeal)
    "THUMB_IP",             # 3  — thumb second knuckle (interphalangeal)
    "THUMB_TIP",            # 4  — thumb fingertip
    "INDEX_FINGER_MCP",     # 5  — index knuckle at palm
    "INDEX_FINGER_PIP",     # 6  — index first bend
    "INDEX_FINGER_DIP",     # 7  — index second bend
    "INDEX_FINGER_TIP",     # 8  — index fingertip
    "MIDDLE_FINGER_MCP",    # 9  — middle knuckle at palm
    "MIDDLE_FINGER_PIP",    # 10 — middle first bend
    "MIDDLE_FINGER_DIP",    # 11 — middle second bend
    "MIDDLE_FINGER_TIP",    # 12 — middle fingertip
    "RING_FINGER_MCP",      # 13 — ring knuckle at palm
    "RING_FINGER_PIP",      # 14 — ring first bend
    "RING_FINGER_DIP",      # 15 — ring second bend
    "RING_FINGER_TIP",      # 16 — ring fingertip
    "PINKY_MCP",            # 17 — pinky knuckle at palm
    "PINKY_PIP",            # 18 — pinky first bend
    "PINKY_DIP",            # 19 — pinky second bend
    "PINKY_TIP",            # 20 — pinky fingertip
]


# ── Output folder roots ───────────────────────────────────────────────────────

# Base folder for annotated frame images
ANNOTATED_ROOT = Path("assets/processed/annotated")

# Base folder for keypoint JSON files
HAND_POSE_ROOT = Path("assets/processed/hand_pose")


# ── Main function ─────────────────────────────────────────────────────────────

def process_video(video_path: str):
    """
    Full pipeline for one video file:
      1. Open video
      2. Loop over every frame
      3. Detect hands with MediaPipe
      4. Draw skeleton on frame and save as PNG
      5. Collect keypoints into a list
      6. Write everything to JSON
      7. Print progress + final summary
    """

    # Convert the string path to a Path object so we can use .stem, .name, etc.
    video_path = Path(video_path)

    # Make sure the file actually exists before we try to open it
    if not video_path.exists():
        print(f"[ERROR] File not found: {video_path}")
        sys.exit(1)                  # exit with error code 1

    # The clip name is the filename without extension, e.g. "WashingCup"
    clip_name = video_path.stem

    print(f"\n{'=' * 60}")
    print(f"  Hand Pose Pipeline")
    print(f"  Clip : {clip_name}")
    print(f"{'=' * 60}\n")

    # ── Step 1: open the video with OpenCV ───────────────────

    cap = cv2.VideoCapture(str(video_path))  # open the video file

    # If OpenCV can't open the file it returns False here
    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {video_path}")
        sys.exit(1)

    # Read video metadata — these properties live inside the VideoCapture object
    fps          = cap.get(cv2.CAP_PROP_FPS)                  # frames per second (e.g. 30.0)
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))     # pixel width  (e.g. 1920)
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))    # pixel height (e.g. 1080)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))     # total number of frames

    print(f"  Resolution : {width} x {height}")
    print(f"  FPS        : {fps}")
    print(f"  Frames     : {total_frames}")
    print()

    # ── Step 2: create output folders ────────────────────────

    # Folder for annotated frames, e.g. assets/processed/annotated/WashingCup/
    annotated_dir = ANNOTATED_ROOT / clip_name
    annotated_dir.mkdir(parents=True, exist_ok=True)   # create all missing parent folders

    # Folder for the JSON output (already exists but mkdir is safe to call again)
    HAND_POSE_ROOT.mkdir(parents=True, exist_ok=True)

    # ── Step 3: set up MediaPipe Hands ───────────────────────

    # We create the Hands detector inside a 'with' block so MediaPipe
    # automatically frees GPU/CPU resources when we're done.
    hands_detector = mp_hands.Hands(
        static_image_mode=False,          # False = video mode (uses tracking between frames, faster)
        max_num_hands=2,                  # detect up to 2 hands per frame
        min_detection_confidence=0.5,     # minimum confidence to call something a hand (0-1)
        min_tracking_confidence=0.5,      # minimum confidence to keep tracking an existing hand
    )

    # ── Step 4: set up counters and storage ──────────────────

    frames_data   = []             # list that will hold one dict per frame (for the JSON)
    frame_id      = 0              # current frame number (starts at 0)
    count_2_hands = 0              # frames where both hands were visible
    count_1_hand  = 0              # frames where only one hand was visible
    count_0_hands = 0              # frames where no hand was detected
    start_time    = time.time()    # record when we started (for the speed stat at the end)

    # ── Step 5: loop over every frame ────────────────────────

    while True:                         # keep reading frames until the video ends

        ret, frame = cap.read()         # ret = True if a frame was read successfully
                                        # frame = the image as a NumPy array (height x width x 3)

        if not ret:                     # ret is False when there are no more frames
            break                       # exit the while loop

        # ── Convert colour from BGR to RGB ───────────────────
        # OpenCV loads frames in BGR colour order (Blue, Green, Red).
        # MediaPipe expects RGB (Red, Green, Blue), so we swap the channels here.
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ── Run MediaPipe hand detection ─────────────────────
        # Setting writeable=False before processing is a small speed optimisation:
        # it tells NumPy not to make a copy of the array inside MediaPipe.
        frame_rgb.flags.writeable = False               # mark as read-only
        results = hands_detector.process(frame_rgb)     # run detection — this is the main call
        frame_rgb.flags.writeable = True                # allow writing again for later steps

        # ── Count how many hands were found ──────────────────
        # results.multi_hand_landmarks is None when no hands are found,
        # or a list (one entry per hand) when hands are detected.
        num_hands = len(results.multi_hand_landmarks) if results.multi_hand_landmarks else 0

        # Update running counts
        if num_hands == 2:
            count_2_hands += 1
        elif num_hands == 1:
            count_1_hand += 1
        else:
            count_0_hands += 1

        # ── Draw the skeleton on a copy of the frame ─────────
        # We copy the original BGR frame so we draw on top of real video colours.
        annotated_frame = frame.copy()    # .copy() so we don't modify the original

        if results.multi_hand_landmarks:   # only try to draw if hands were found
            for hand_landmarks in results.multi_hand_landmarks:
                # draw_landmarks draws:
                #   • a circle at each of the 21 keypoints
                #   • lines between connected keypoints (the "bones")
                mp_draw.draw_landmarks(
                    annotated_frame,                                  # image to draw on
                    hand_landmarks,                                   # 21 landmark points
                    mp_hands.HAND_CONNECTIONS,                        # which points to connect
                    mp_styles.get_default_hand_landmarks_style(),     # dot colours/sizes
                    mp_styles.get_default_hand_connections_style(),   # line colours/thickness
                )

        # ── Save the annotated frame as PNG ──────────────────
        # zfill(6) pads the number with leading zeros: 1 → "000001"
        # This keeps files sorted correctly in any file browser.
        frame_filename  = f"frame_{str(frame_id).zfill(6)}.png"
        frame_save_path = annotated_dir / frame_filename
        cv2.imwrite(str(frame_save_path), annotated_frame)   # write the image to disk

        # ── Collect keypoint data for the JSON ───────────────
        hands_list = []   # will hold one dict per detected hand for this frame

        if results.multi_hand_landmarks and results.multi_handedness:
            # zip() pairs up:
            #   results.multi_hand_landmarks[i] — the 21 landmarks of hand i
            #   results.multi_handedness[i]      — the label ("Left"/"Right") of hand i
            for hand_landmarks, handedness in zip(
                results.multi_hand_landmarks,
                results.multi_handedness,
            ):
                # classification[0] holds the best match: label and confidence score
                hand_label      = handedness.classification[0].label         # "Left" or "Right"
                hand_confidence = round(handedness.classification[0].score, 4)  # e.g. 0.9732

                # Build a dict of all 21 named keypoints for this hand
                keypoints_dict = {}
                for idx, landmark in enumerate(hand_landmarks.landmark):
                    # landmark.x, .y are normalised (0.0 to 1.0 relative to frame dimensions)
                    # landmark.z is relative depth — negative means closer to camera
                    name = LANDMARK_NAMES[idx]   # look up the human-readable name

                    keypoints_dict[name] = {
                        "x":  round(landmark.x, 6),        # normalised x  (0 = left edge)
                        "y":  round(landmark.y, 6),        # normalised y  (0 = top edge)
                        "z":  round(landmark.z, 6),        # relative depth
                        "px": int(landmark.x * width),     # actual pixel x on the frame
                        "py": int(landmark.y * height),    # actual pixel y on the frame
                    }

                # Add this hand's complete data to the list
                hands_list.append({
                    "label":      hand_label,        # "Left" or "Right"
                    "confidence": hand_confidence,   # how sure MediaPipe is
                    "keypoints":  keypoints_dict,    # all 21 named landmark positions
                })

        # ── Build the frame dict and add to master list ───────
        frames_data.append({
            "frame_id":       frame_id,                    # e.g. 0, 1, 2 ...
            "timestamp_sec":  round(frame_id / fps, 4),   # e.g. 0.0333 seconds
            "hands_detected": num_hands,                  # 0, 1, or 2
            "hands":          hands_list,                 # list of hand dicts (may be empty)
        })

        # ── Print progress every 100 frames ──────────────────
        if frame_id % 100 == 0:
            elapsed  = time.time() - start_time            # seconds elapsed so far
            pct_done = (frame_id / total_frames) * 100     # how far through the video we are
            print(
                f"  Frame {frame_id:>6} / {total_frames}"
                f"  ({pct_done:5.1f}%)"
                f"  |  hands: 2={count_2_hands}  1={count_1_hand}  0={count_0_hands}"
                f"  |  {elapsed:.1f}s elapsed"
            )

        frame_id += 1    # increment frame counter before the next loop iteration

    # ── Step 6: release resources ─────────────────────────────
    cap.release()             # close the video file and free OpenCV's file handle
    hands_detector.close()   # release MediaPipe's internal model and memory

    # ── Step 7: write the keypoint JSON ───────────────────────

    output_json = {
        "clip_name":    clip_name,       # e.g. "WashingCup"
        "total_frames": frame_id,        # number of frames we actually processed
        "fps":          fps,             # original video frame rate
        "resolution": {
            "width":  width,             # frame width in pixels
            "height": height,            # frame height in pixels
        },
        "frames": frames_data,           # the big list — one entry per frame
    }

    # Build the file path, e.g. assets/processed/hand_pose/WashingCup.json
    json_path = HAND_POSE_ROOT / f"{clip_name}.json"

    # Write the dict to disk as a nicely formatted JSON file
    with open(json_path, "w") as f:
        json.dump(output_json, f, indent=2)   # indent=2 adds spaces so it's readable

    # ── Step 8: print final summary ───────────────────────────

    total_time         = time.time() - start_time               # total run time in seconds
    frames_with_hands  = count_1_hand + count_2_hands           # frames that had at least one hand
    detection_rate     = (frames_with_hands / frame_id) * 100   # as a percentage

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
#
# It does NOT run when another script does  "import hand_pose"
if __name__ == "__main__":

    # sys.argv is a list of everything you typed on the command line.
    # sys.argv[0] is always the script name itself ("pipeline/hand_pose.py").
    # sys.argv[1] should be the video path the user provided.
    if len(sys.argv) != 2:
        # Wrong number of arguments — show usage instructions and exit
        print("Usage  : python pipeline/hand_pose.py <path_to_video>")
        print("Example: python pipeline/hand_pose.py assets/videos/WashingCup.mp4")
        sys.exit(1)   # non-zero exit code signals an error to the shell

    # Call our main function with the path the user typed
    process_video(sys.argv[1])
