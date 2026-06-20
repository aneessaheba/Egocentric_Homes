"""
run_pipeline.py
───────────────
Master script that runs the full HomeHands dataset pipeline on every video.

For each .mp4 in assets/videos/ it runs, in order:
  1. hand_pose.py     — detect hands, save keypoint JSON + annotated frames
  2. segmentation.py  — SAM3 text-prompted arm segmentation
  3. transcribe.py    — local Whisper audio transcription + subtitle burning

Then it combines all outputs into one final annotation JSON per clip:
  assets/processed/annotations/[clip_name]_full.json

Usage:
  python pipeline/run_pipeline.py
"""

# ── Imports ───────────────────────────────────────────────────────────────────

import sys          # used to exit with an error code if something goes wrong
import csv          # read assets/clip_metadata.csv
import json         # read and write JSON files
import time         # measure elapsed time
import traceback    # print full error details when a module fails
from pathlib import Path   # clean cross-platform file path handling

# Import the three pipeline modules directly so we can call their functions.
# Because the pipeline/ folder contains these files, we need to make sure Python
# can find them. We add the pipeline folder to sys.path below.
# sys.path is the list of folders Python searches when you do "import something".
sys.path.insert(0, str(Path(__file__).parent))   # add pipeline/ directory to search path

# Now we can import our three modules as if they were installed libraries
import hand_pose     # hand_pose.process_video()
import segmentation  # segmentation.process_video()
import transcribe    # transcribe.process_video()


# ── Folder paths ──────────────────────────────────────────────────────────────

# Folder containing the raw .mp4 video files
VIDEOS_DIR = Path("assets/videos")

# Folder where hand pose JSONs are stored (written by hand_pose.py)
HAND_POSE_DIR = Path("assets/processed/hand_pose")

# Folder where standalone arm pose JSONs are stored (written by arm_pose.py)
ARM_POSE_DIR = Path("assets/processed/arm_pose")

# Folder where narration JSONs are stored (written by transcribe.py)
NARRATIONS_DIR = Path("assets/processed/narrations")

# Folder where the final combined annotation JSONs will be saved
ANNOTATIONS_DIR = Path("assets/processed/annotations")

# Per-clip metadata (activity/participant/home/room/camera), keyed by filename stem
CLIP_METADATA_CSV = Path("assets/clip_metadata.csv")


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_clip_metadata() -> dict:
    """
    Load assets/clip_metadata.csv into a dict keyed by filename stem.

    Columns: filename, activity_category, activity_label, participant_id,
             home_id, room, camera_model.

    Returns {} if the CSV doesn't exist — callers must handle missing
    metadata explicitly (null fields + a printed warning), not guess.
    """
    metadata = {}
    if not CLIP_METADATA_CSV.exists():
        print(f"  [metadata] No CSV found at {CLIP_METADATA_CSV} — "
              f"all clips will have null activity/participant/home/room/camera fields")
        return metadata
    with open(CLIP_METADATA_CSV, newline="") as f:
        for row in csv.DictReader(f):
            metadata[Path(row["filename"]).stem] = row
    return metadata


def find_narration_id_for_frame(timestamp_sec: float, narrations: list):
    """
    Given a frame's timestamp and the list of narration segments,
    return the id of the narration segment that was being spoken at that
    moment, or None if no segment covers this timestamp.

    A narration covers a frame if:
      segment["start"] <= timestamp_sec < segment["end"]
    """
    for seg in narrations:
        if seg["start"] <= timestamp_sec < seg["end"]:
            return seg["id"]
    return None    # no segment covered this timestamp (silence or gap)


def clip_id_from_index(index: int) -> str:
    """
    Create a short dataset ID string for a clip.
    index 0 → "HH_001"
    index 1 → "HH_002"
    etc.

    "HH" stands for HomeHands.
    zfill(3) pads the number with leading zeros to three digits.
    """
    return f"HH_{str(index + 1).zfill(3)}"    # index+1 so IDs start at 001 not 000


# ── Module runner ─────────────────────────────────────────────────────────────

def run_module(name: str, func, video_path: Path) -> bool:
    """
    Run a single pipeline function (hand_pose, segmentation, or transcribe)
    on one video, catch any errors, and print a status line.

    Returns True if the module succeeded, False if it raised an exception.

    Parameters:
      name       — human-readable module name for the status line, e.g. "Hand pose"
      func       — the process_video function to call, e.g. hand_pose.process_video
      video_path — Path object pointing to the .mp4 file
    """
    try:
        func(str(video_path))    # call the module's process_video() with the video path
        # If we reach this line, no exception was raised — success
        print(f"  ✅ {name:<20} — {video_path.stem}")
        return True    # signal success to the caller

    except Exception as e:
        # Something went wrong inside the module — catch the exception,
        # print it, and return False so the pipeline continues to the next step
        print(f"  ❌ {name:<20} FAILED — {video_path.stem}")
        print(f"     Error: {e}")
        # Print the full stack trace so the user can see exactly where it failed
        traceback.print_exc()
        return False    # signal failure to the caller


# ── Combine outputs into final annotation JSON ────────────────────────────────

def build_annotation(video_path: Path, clip_index: int, clip_metadata: dict) -> dict | None:
    """
    Load the hand pose, arm pose, and narration JSON for one clip, merge them
    together under the unified schema, and return the combined annotation dict.

    Returns None if the required input files are missing.

    Key rule: every frame gets the same key set every time — "hands", "arm",
    "segmentation", "depth", "narration_id" — defaulting to null/empty rather
    than being omitted when a given modality wasn't run for this clip.
    """
    clip_name = video_path.stem    # e.g. "WashingCup"

    # Paths to the JSON files produced by hand_pose.py, arm_pose.py, transcribe.py
    hand_pose_json_path = HAND_POSE_DIR   / f"{clip_name}.json"
    arm_pose_json_path  = ARM_POSE_DIR    / f"{clip_name}.json"
    narration_json_path = NARRATIONS_DIR  / f"{clip_name}.json"

    # ── Load hand pose JSON ───────────────────────────────────
    if not hand_pose_json_path.exists():
        print(f"     [merge] Skipping — hand pose JSON not found: {hand_pose_json_path}")
        return None    # can't merge without this file

    with open(hand_pose_json_path, "r") as f:
        pose_data = json.load(f)    # load the full hand pose + segmentation data

    # ── Load arm pose JSON ─────────────────────────────────────
    # arm_pose.py is not yet auto-invoked by this script's module-running loop
    # (that's Stage 7) — if its output doesn't exist for this clip, every
    # frame's "arm" block is explicitly null, never omitted.
    arm_by_frame_id = {}
    if arm_pose_json_path.exists():
        with open(arm_pose_json_path, "r") as f:
            arm_data = json.load(f)
        arm_by_frame_id = {fr["frame_id"]: fr.get("arm") for fr in arm_data.get("frames", [])}
    else:
        print(f"     [merge] No arm pose JSON found ({arm_pose_json_path}) — \"arm\" will be null")

    null_arm = {"left_wrist": None, "left_elbow": None, "right_wrist": None, "right_elbow": None}

    # ── Load narration JSON ───────────────────────────────────
    # The narration file is optional — if Whisper found nothing, it may not exist
    narrations = []    # default to empty list if file is missing
    if narration_json_path.exists():
        with open(narration_json_path, "r") as f:
            narration_data = json.load(f)
        narrations = narration_data.get("narrations", [])   # list of timed segments
    else:
        print(f"     [merge] No narration JSON found — narrations will be empty")

    # ── Pull metadata from the hand pose JSON ─────────────────
    total_frames = pose_data.get("total_frames", 0)   # how many frames in the video
    fps          = pose_data.get("fps", 0)             # frames per second, exact as reported by the source video
    res          = pose_data.get("resolution", {})    # {"width": ..., "height": ...}
    resolution   = f"{res.get('width', 0)}x{res.get('height', 0)}"   # "WxH" string per unified schema

    # ── Calculate hand detection rate ─────────────────────────
    # Count frames where at least one hand was detected
    frames_with_hands = sum(
        1 for f in pose_data["frames"]
        if f.get("hands", {}).get("hands_found", 0) > 0
    )
    # Detection rate as a 0-1 fraction, matching the unified schema
    detection_rate = round(frames_with_hands / total_frames, 4) if total_frames > 0 else 0.0

    # ── Get total duration from narration (or calculate from frames/fps) ──────
    if narrations:
        # Duration = end of the last narration segment
        total_duration = narrations[-1]["end"]
    elif fps and total_frames:
        # Fallback: calculate from frame count and FPS
        total_duration = round(total_frames / fps, 3)
    else:
        total_duration = 0.0    # unknown

    # ── Build the merged frames list ──────────────────────────
    merged_frames = []

    for frame in pose_data["frames"]:
        frame_id  = frame.get("frame_id")
        timestamp = frame.get("timestamp_sec", 0.0)

        merged_frame = {
            "frame_id":      frame_id,
            "timestamp_sec": timestamp,
            "hands":         frame.get("hands", {"hands_found": 0, "wrists": []}),
            "arm":           arm_by_frame_id.get(frame_id, null_arm) if arm_by_frame_id else null_arm,
            "segmentation":  frame.get("segmentation"),    # None if segmentation.py hasn't run for this clip
            "depth":         None,                          # populated once depth.py exists (Stage 5)
            "narration_id":  find_narration_id_for_frame(timestamp, narrations),
        }
        merged_frames.append(merged_frame)

    # ── Per-clip metadata (activity/participant/home/room/camera) from CSV ────
    meta = clip_metadata.get(clip_name)
    if meta is None:
        print(f"     [metadata] No CSV row for \"{clip_name}\" — "
              f"activity/participant/home/room/camera fields will be null")

    def meta_get(key):
        return (meta.get(key) or None) if meta else None

    # ── Build the top-level annotation dict ───────────────────
    annotation = {
        "clip_id":             clip_id_from_index(clip_index),   # e.g. "HH_001" — cosmetic display ID only
        "filename":            video_path.name,                  # e.g. "WashingCup.mp4"
        "activity_category":   meta_get("activity_category"),
        "activity_label":      meta_get("activity_label"),
        "participant_id":      meta_get("participant_id"),
        "home_id":             meta_get("home_id"),
        "room":                meta_get("room"),
        "camera_model":        meta_get("camera_model"),
        "duration_sec":        total_duration,                   # video length in seconds
        "total_frames":        total_frames,                     # total frame count
        "fps":                 fps,                              # frames per second
        "resolution":          resolution,                       # "WxH" string
        "quality_control":     None,                              # populated once the QC gate exists (Stage 3/6)
        "hand_detection_rate": detection_rate,                   # fraction (0-1) of frames with hands
        "narrations":          narrations,                       # list of narration segments
        "frames":              merged_frames,                    # merged per-frame data
    }

    return annotation    # return the dict to the caller


# ── Main pipeline ─────────────────────────────────────────────────────────────

def main():
    """
    Entry point — discovers all videos, runs the three pipeline modules on each,
    merges outputs into final annotations, and prints the summary.
    """

    print(f"\n{'═' * 54}")
    print(f"       HomeHands Dataset Pipeline")
    print(f"{'═' * 54}\n")

    # ── Step 1: find all .mp4 videos ─────────────────────────

    # sorted() ensures videos are always processed in alphabetical order
    # This makes the clip IDs (HH_001, HH_002, ...) consistent between runs
    videos = sorted(
        p for p in VIDEOS_DIR.iterdir()          # iterate over everything in assets/videos/
        if p.suffix.lower() == ".mp4"            # keep only .mp4 files
        and "_subtitled" not in p.stem           # skip subtitled copies we already made
        and p.parent == VIDEOS_DIR               # don't recurse into sub-folders
    )

    # Stop early if there are no videos to process
    if not videos:
        print(f"[ERROR] No .mp4 files found in {VIDEOS_DIR}")
        print(f"        Put your video files there and re-run.")
        sys.exit(1)

    # Print the numbered list of discovered videos
    print(f"  Found {len(videos)} video(s):\n")
    for i, v in enumerate(videos):
        print(f"    {i + 1:>2}. {v.name}")   # right-align the number for clean alignment
    print()

    # Create the annotations output folder if it doesn't exist
    ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)

    # Load per-clip metadata (activity/participant/home/room/camera) once
    clip_metadata = load_clip_metadata()

    # ── Step 2–5: process each video ─────────────────────────

    # These counters are used for the final summary
    total_processed   = 0    # videos that completed without any module failure
    total_failed      = 0    # videos where at least one module failed
    grand_total_frames   = 0    # sum of total_frames across all clips
    grand_frames_hands   = 0    # sum of frames_with_hands across all clips
    grand_narrations     = 0    # sum of narration segments across all clips
    failed_videos     = []   # names of videos that had failures

    # Track per-video module results for the summary
    results = []    # list of dicts, one per video, with keys "video", "ok", "failed_modules"

    overall_start = time.time()    # record when we started the whole pipeline

    for clip_index, video_path in enumerate(videos):
        # ── Print a header for this video ────────────────────
        clip_name = video_path.stem   # filename without extension
        print(f"\n{'─' * 54}")
        print(f"  [{clip_index + 1}/{len(videos)}]  {video_path.name}")
        print(f"{'─' * 54}")

        clip_start    = time.time()     # start timer for this clip
        failed_any    = False           # becomes True if any module fails
        failed_modules = []             # list of module names that failed for this clip

        # ── Step 2a: run hand_pose.py ─────────────────────────
        ok_pose = run_module(
            "Hand pose",               # display name for status line
            hand_pose.process_video,   # the function to call
            video_path,                # the video to process
        )
        if not ok_pose:
            failed_any = True
            failed_modules.append("hand_pose")

        # ── Step 2b: run segmentation.py ─────────────────────
        # Segmentation reads the hand pose JSON, so we only run it if hand pose succeeded.
        # If hand pose failed there are no wrist coordinates to use as SAM2 prompts.
        if ok_pose:
            ok_seg = run_module(
                "Segmentation",
                segmentation.process_video,
                video_path,
            )
            if not ok_seg:
                failed_any = True
                failed_modules.append("segmentation")
        else:
            # Skip segmentation and mark it as failed (dependency not met)
            print(f"  ⏭  {'Segmentation':<20} — skipped (hand pose failed)")
            ok_seg = False
            failed_modules.append("segmentation (skipped)")

        # ── Step 2c: run transcribe.py ────────────────────────
        # Transcription is independent of hand pose and segmentation,
        # so we always run it regardless of whether the above succeeded.
        ok_trans = run_module(
            "Transcription",
            transcribe.process_video,
            video_path,
        )
        if not ok_trans:
            failed_any = True
            failed_modules.append("transcription")

        clip_elapsed = time.time() - clip_start    # how long this clip took

        # ── Step 5: combine outputs into final annotation JSON ─
        print(f"\n  Building final annotation JSON ...")
        annotation = build_annotation(video_path, clip_index, clip_metadata)

        if annotation is not None:
            # Write the combined annotation to disk
            out_path = ANNOTATIONS_DIR / f"{clip_name}_full.json"
            with open(out_path, "w") as f:
                json.dump(annotation, f, indent=2)   # save as readable JSON
            print(f"  💾 Saved → {out_path}")

            # Accumulate global statistics for the final summary
            grand_total_frames += annotation["total_frames"]
            grand_frames_hands += int(
                annotation["total_frames"] * annotation["hand_detection_rate"]
            )    # reverse-calculate frames_with_hands from the rate fraction
            grand_narrations   += len(annotation["narrations"])
        else:
            print(f"  ⚠️  Could not build annotation — required JSON files missing.")

        # ── Print per-clip completion line ────────────────────
        status_icon = "✅" if not failed_any else "⚠️ "
        print(f"\n  {status_icon} {clip_name} complete  ({clip_elapsed:.1f}s)")

        if failed_any:
            total_failed += 1
            failed_videos.append(clip_name)
        else:
            total_processed += 1

    # ── Step 6: final summary ─────────────────────────────────

    total_time = time.time() - overall_start    # total wall-clock time for everything

    print(f"\n\n{'═' * 54}")
    print(f"       HomeHands Pipeline Complete")
    print(f"{'═' * 54}")
    print(f"  Videos processed     : {total_processed}")
    print(f"  Videos with failures : {total_failed}")
    print(f"  Total frames         : {grand_total_frames:,}")   # comma separator e.g. 12,450
    print(f"  Frames with hands    : {grand_frames_hands:,}")
    print(f"  Narration segments   : {grand_narrations}")
    print(f"  Annotations saved to : {ANNOTATIONS_DIR}/")
    print(f"  Total time           : {total_time:.1f}s")

    # If any videos had failures, list them so the user knows what to check
    if failed_videos:
        print(f"\n  Videos with errors:")
        for name in failed_videos:
            print(f"    ✗  {name}")   # list each failed video name

    print(f"{'═' * 54}\n")


# ── Entry point ───────────────────────────────────────────────────────────────

# This block only runs when you execute the file directly:
#   python pipeline/run_pipeline.py
#
# It does NOT run when another script does "import run_pipeline".
if __name__ == "__main__":
    main()    # call the main function to start everything
