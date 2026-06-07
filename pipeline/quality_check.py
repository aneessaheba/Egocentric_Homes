"""
quality_check.py
────────────────
Video quality checker for egocentric household activity recordings.

Runs 5 checks per video (basic properties, hand detection, blur, brightness,
audio narration) and produces a scored quality report saved as JSON.

Usage:
  Single video : python pipeline/quality_check.py assets/videos/WashingCup.mp4
  All videos   : python pipeline/quality_check.py assets/videos/
"""

# ── Standard library imports ──────────────────────────────────────────────────
import sys                    # command-line arguments
import json                   # write JSON reports
import time                   # measure elapsed time per video
import subprocess             # run ffprobe as a shell command
from pathlib import Path      # cross-platform file and folder paths

# ── Third-party imports ───────────────────────────────────────────────────────
import cv2                    # OpenCV — open video, read frames, compute blur
import numpy as np            # NumPy — array maths for brightness/blur averages
import mediapipe as mp        # MediaPipe — hand detection on sampled frames

# ── Optional Whisper import (graceful fallback if not installed) ───────────────
try:
    import whisper            # OpenAI Whisper — transcribe audio for narration check
    WHISPER_AVAILABLE = True  # flag so we know Whisper loaded successfully
except ImportError:
    WHISPER_AVAILABLE = False # Whisper not installed; skip narration check


# ── Paths ─────────────────────────────────────────────────────────────────────
QUALITY_ROOT = Path("assets/processed/quality")   # output folder for JSON reports

# ── Sampling interval ─────────────────────────────────────────────────────────
SAMPLE_EVERY = 10   # analyse every 10th frame — keeps runtime under 2 min per video

# ── Blur threshold ────────────────────────────────────────────────────────────
BLUR_THRESHOLD = 200   # Laplacian variance below this = frame is blurry

# ── Brightness thresholds ─────────────────────────────────────────────────────
BRIGHT_LOW  = 80    # below this = too dark
BRIGHT_HIGH = 180   # above this = too bright


# ── MediaPipe hands model ─────────────────────────────────────────────────────
# Load once at module level so every video reuses the same detector instance
_mp_hands    = mp.solutions.hands                               # hands solution namespace
_hand_detect = _mp_hands.Hands(                                 # create detector
    static_image_mode=True,                                     # one frame at a time (no tracking)
    max_num_hands=2,                                            # look for up to 2 hands
    min_detection_confidence=0.5,                               # confidence threshold
)


# ── Helper: run ffprobe and return parsed JSON ────────────────────────────────

def ffprobe(video_path: Path) -> dict:
    """Run ffprobe on a video and return the parsed JSON output."""
    cmd = [                                        # build the ffprobe command
        "ffprobe",                                 # CLI tool bundled with ffmpeg
        "-v", "quiet",                             # suppress info messages
        "-print_format", "json",                   # output as JSON
        "-show_streams",                           # include stream details
        "-show_format",                            # include container/format info
        str(video_path),                           # path to the video file
    ]
    try:
        result = subprocess.run(                   # run the command
            cmd,
            capture_output=True,                   # capture stdout and stderr
            text=True,                             # decode bytes to string
            timeout=30,                            # abort if it hangs for 30 s
        )
        return json.loads(result.stdout)           # parse JSON output into a dict
    except Exception:
        return {}                                  # return empty dict on any error


# ── Check 1: Basic properties ─────────────────────────────────────────────────

def check_basic(video_path: Path) -> dict:
    """Return basic video properties using OpenCV and ffprobe."""
    cap       = cv2.VideoCapture(str(video_path))         # open the video file
    width     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))    # frame width in pixels
    height    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))   # frame height in pixels
    fps       = cap.get(cv2.CAP_PROP_FPS)                 # frames per second
    n_frames  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))    # total frame count
    duration  = round(n_frames / fps, 2) if fps > 0 else 0   # duration in seconds
    cap.release()                                          # close the video handle

    probe     = ffprobe(video_path)                        # run ffprobe for audio info
    streams   = probe.get("streams", [])                   # list of stream dicts
    has_audio = any(                                       # True if any stream is audio
        s.get("codec_type") == "audio" for s in streams
    )

    return {                                               # pack results into a dict
        "width":     width,
        "height":    height,
        "fps":       round(fps, 2),
        "n_frames":  n_frames,
        "duration":  duration,
        "has_audio": has_audio,
    }


# ── Check 2: Hand detection ───────────────────────────────────────────────────

def check_hands(video_path: Path) -> dict:
    """Sample every 10th frame, detect hands, return detection statistics."""
    cap         = cv2.VideoCapture(str(video_path))    # open video
    frame_id    = 0                                    # current frame index
    sampled     = 0                                    # how many frames we analysed
    two_hands   = 0                                    # frames where 2 hands found
    one_hand    = 0                                    # frames where 1 hand found
    zero_hands  = 0                                    # frames where 0 hands found
    conf_scores = []                                   # list of per-detection confidence values

    while True:
        ret, frame = cap.read()                        # read next frame
        if not ret:                                    # end of video
            break

        if frame_id % SAMPLE_EVERY == 0:              # only analyse every 10th frame
            sampled += 1                               # count this sample
            small   = cv2.resize(frame, (640, 360))   # downscale for faster inference
            rgb     = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)   # MediaPipe needs RGB
            result  = _hand_detect.process(rgb)                # run hand detection

            if result.multi_hand_landmarks:            # at least one hand found
                n = len(result.multi_hand_landmarks)   # number of hands detected
                if n >= 2:                             # both hands visible
                    two_hands += 1
                else:                                  # only one hand visible
                    one_hand  += 1

                # collect confidence scores for each detected hand
                if result.multi_handedness:
                    for h in result.multi_handedness:              # loop over each hand
                        conf_scores.append(h.classification[0].score)   # confidence 0-1
            else:
                zero_hands += 1                        # no hands in this frame

        frame_id += 1                                  # advance frame counter

    cap.release()                                      # close video

    detected    = two_hands + one_hand                 # frames with ≥1 hand
    detect_rate = round(detected / sampled * 100, 1) if sampled else 0.0    # % detected
    both_rate   = round(two_hands / sampled * 100, 1) if sampled else 0.0   # % with 2 hands
    avg_conf    = round(float(np.mean(conf_scores)), 3) if conf_scores else 0.0  # mean conf

    return {                                           # return all statistics
        "sampled_frames":   sampled,
        "two_hands":        two_hands,
        "one_hand":         one_hand,
        "zero_hands":       zero_hands,
        "detection_rate":   detect_rate,
        "both_hands_rate":  both_rate,
        "avg_confidence":   avg_conf,
    }


# ── Check 3: Blur score ───────────────────────────────────────────────────────

def check_blur(video_path: Path) -> dict:
    """Sample every 10th frame, compute Laplacian variance (blur score)."""
    cap          = cv2.VideoCapture(str(video_path))   # open video
    frame_id     = 0                                   # current frame index
    scores       = []                                  # blur score per sampled frame
    blurry_count = 0                                   # frames below blur threshold

    while True:
        ret, frame = cap.read()                        # read next frame
        if not ret:                                    # end of video
            break

        if frame_id % SAMPLE_EVERY == 0:              # every 10th frame only
            small  = cv2.resize(frame, (640, 360))     # resize for speed
            gray   = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)   # convert to grayscale
            score  = cv2.Laplacian(gray, cv2.CV_64F).var()     # Laplacian variance = sharpness
            scores.append(score)                       # store this frame's score
            if score < BLUR_THRESHOLD:                 # score below threshold = blurry
                blurry_count += 1

        frame_id += 1                                  # advance frame counter

    cap.release()                                      # close video

    sampled    = len(scores)                           # total frames sampled
    avg_blur   = round(float(np.mean(scores)), 1) if scores else 0.0       # average score
    blurry_pct = round(blurry_count / sampled * 100, 1) if sampled else 0.0  # % blurry

    return {                                           # return blur statistics
        "avg_blur_score":  avg_blur,
        "blurry_frames":   blurry_count,
        "blurry_pct":      blurry_pct,
        "sampled_frames":  sampled,
    }


# ── Check 4: Brightness ───────────────────────────────────────────────────────

def check_brightness(video_path: Path) -> dict:
    """Sample every 10th frame, compute average pixel brightness."""
    cap          = cv2.VideoCapture(str(video_path))   # open video
    frame_id     = 0                                   # current frame index
    brightness   = []                                  # mean brightness per frame

    while True:
        ret, frame = cap.read()                        # read next frame
        if not ret:                                    # end of video
            break

        if frame_id % SAMPLE_EVERY == 0:              # every 10th frame only
            small  = cv2.resize(frame, (320, 180))     # small resize — brightness doesn't need detail
            gray   = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)   # grayscale
            brightness.append(float(gray.mean()))      # mean pixel value 0-255

        frame_id += 1                                  # advance frame counter

    cap.release()                                      # close video

    avg   = round(float(np.mean(brightness)), 1) if brightness else 0.0    # overall average
    label = (                                          # classify into dark/good/bright
        "too dark"   if avg < BRIGHT_LOW  else
        "too bright" if avg > BRIGHT_HIGH else
        "good"
    )

    return {                                           # return brightness statistics
        "avg_brightness": avg,
        "label":          label,
    }


# ── Check 5: Audio narration ──────────────────────────────────────────────────

def check_narration(video_path: Path, has_audio: bool) -> dict:
    """If video has audio, extract it and run Whisper to detect narration."""
    if not has_audio:                                  # no audio stream — skip immediately
        return {"narration_found": False, "segments": 0, "note": "no audio track"}

    if not WHISPER_AVAILABLE:                          # Whisper not installed — skip
        return {"narration_found": False, "segments": 0, "note": "whisper not installed"}

    try:
        audio_path = Path("/tmp") / f"{video_path.stem}_audio.wav"   # temp WAV file

        # extract audio from the video using ffmpeg
        subprocess.run([
            "ffmpeg", "-y",                            # overwrite output if exists
            "-i", str(video_path),                     # input video
            "-vn",                                     # no video stream in output
            "-ac", "1",                                # mono audio (faster for Whisper)
            "-ar", "16000",                            # 16kHz sample rate (Whisper standard)
            "-t", "60",                                # only take first 60 seconds
            str(audio_path),                           # output path
        ], capture_output=True, timeout=60)            # capture output, 60 s timeout

        model   = whisper.load_model("base")           # load Whisper base model
        result  = model.transcribe(str(audio_path))    # transcribe the audio
        segs    = [s for s in result.get("segments", [])   # filter segments with real speech
                   if s.get("no_speech_prob", 1.0) < 0.5]  # no_speech_prob < 0.5 = real speech

        audio_path.unlink(missing_ok=True)             # delete the temp WAV file

        return {                                       # return narration statistics
            "narration_found": len(segs) > 0,
            "segments":        len(segs),
            "note":            "whisper base",
        }
    except Exception as e:
        return {"narration_found": False, "segments": 0, "note": str(e)}   # graceful error


# ── Scoring ───────────────────────────────────────────────────────────────────

def compute_score(basic: dict, hands: dict, blur: dict, brightness: dict, narr: dict) -> tuple:
    """Compute a 0-100 quality score and return (score, list_of_issues)."""
    score  = 0     # start at zero, add points for good results
    issues = []    # collect human-readable issue descriptions

    # +25 — hand detection rate above 80%
    if hands["detection_rate"] >= 80:
        score += 25
    # -20 — hand detection rate below 60%
    elif hands["detection_rate"] < 60:
        score -= 20
        issues.append("low hands")
    # -10 — additionally penalise if detection rate is below 40%
    if hands["detection_rate"] < 40:
        score -= 10
        issues.append("very low hands")

    # +20 — sharp video (avg blur score above 500)
    if blur["avg_blur_score"] >= 500:
        score += 20
    # -25 — blurry video (avg blur score below 200)
    elif blur["avg_blur_score"] < 200:
        score -= 25
        issues.append("low blur")

    # +15 — brightness in the good range
    if brightness["label"] == "good":
        score += 15
    else:
        issues.append(brightness["label"])             # record "too dark" or "too bright"

    # +15 — both hands visible in more than half of sampled frames
    if hands["both_hands_rate"] >= 50:
        score += 15

    # +10 — audio narration speech detected
    if narr["narration_found"]:
        score += 10
    # -15 — no audio track at all
    elif not basic["has_audio"]:
        score -= 15
        issues.append("no audio")

    # +10 — FPS is 30 (expected for GoPro footage)
    if basic["fps"] >= 29.9:
        score += 10

    # +5 — resolution is at least 1080p (height >= 1080)
    if basic["height"] >= 1080:
        score += 5

    score = max(0, min(100, score))                    # clamp score to 0-100 range

    return score, issues                               # return final score and issue list


# ── Verdict label ─────────────────────────────────────────────────────────────

def verdict(score: int) -> str:
    """Return a verdict string based on the score."""
    if score >= 80:                                    # 80-100 = good
        return "✅ GOOD"
    elif score >= 60:                                  # 60-79 = okay
        return "⚠️ OKAY"
    else:                                              # 0-59 = bad
        return "❌ BAD"


# ── Terminal report printer ───────────────────────────────────────────────────

def print_report(video_path: Path, basic: dict, hands: dict, blur: dict,
                 brightness: dict, narr: dict, score: int):
    """Print a formatted quality report for one video to the terminal."""

    v    = verdict(score)                              # compute verdict label
    w    = basic["width"]                              # frame width
    h    = basic["height"]                             # frame height
    tick = lambda b: "✅" if b else "❌"               # helper: green tick or red cross

    print(f"\n{'═' * 48}")
    print(f"  Quality Report: {video_path.name}")
    print(f"{'═' * 48}")
    print(f"  Resolution      : {w}x{h}")             # e.g. 3840x2160
    print(f"  FPS             : {basic['fps']}")       # e.g. 29.97
    print(f"  Duration        : {basic['duration']}s") # e.g. 76.1s
    print()
    print(f"  Hand detection  : {hands['detection_rate']}%  {tick(hands['detection_rate'] >= 80)}")
    print(f"  Both hands      : {hands['both_hands_rate']}%  {tick(hands['both_hands_rate'] >= 50)}")
    print(f"  Avg confidence  : {hands['avg_confidence']}  {tick(hands['avg_confidence'] >= 0.7)}")
    print()
    print(f"  Blur score      : {blur['avg_blur_score']}  {tick(blur['avg_blur_score'] >= 500)}")
    print(f"  Blurry frames   : {blur['blurry_pct']}%  {tick(blur['blurry_pct'] < 10)}")
    print()
    bl = brightness["label"]                           # "good" / "too dark" / "too bright"
    print(f"  Brightness      : {brightness['avg_brightness']} ({bl})  {tick(bl == 'good')}")
    print()
    print(f"  Has audio       : {'Yes' if basic['has_audio'] else 'No'}  {tick(basic['has_audio'])}")
    seg_txt = f"Yes ({narr['segments']} segments)" if narr["narration_found"] else "No"
    print(f"  Narration found : {seg_txt}  {tick(narr['narration_found'])}")
    print()
    print(f"  SCORE           : {score}/100")
    print(f"  VERDICT         : {v}")
    print(f"{'═' * 48}")


# ── Per-video pipeline ────────────────────────────────────────────────────────

def analyse_video(video_path: Path) -> dict:
    """Run all checks on one video and return the full result dict."""
    t0 = time.time()                                   # record start time

    print(f"\n  Analysing: {video_path.name} ...")

    basic      = check_basic(video_path)               # check 1: resolution, fps, audio
    hands      = check_hands(video_path)               # check 2: hand detection
    blur       = check_blur(video_path)                # check 3: blur score
    brightness = check_brightness(video_path)          # check 4: brightness
    narr       = check_narration(video_path, basic["has_audio"])   # check 5: narration

    score, issues = compute_score(basic, hands, blur, brightness, narr)   # final score

    print_report(video_path, basic, hands, blur, brightness, narr, score) # terminal output

    # ── Save JSON report ──────────────────────────────────────────────────────
    QUALITY_ROOT.mkdir(parents=True, exist_ok=True)    # create output folder if needed
    json_path = QUALITY_ROOT / f"{video_path.stem}_quality.json"   # output path

    report = {                                         # build the full report dict
        "video":      str(video_path),
        "clip_name":  video_path.stem,
        "score":      score,
        "verdict":    verdict(score),
        "issues":     issues,
        "basic":      basic,
        "hands":      hands,
        "blur":       blur,
        "brightness": brightness,
        "narration":  narr,
        "elapsed_s":  round(time.time() - t0, 1),     # total time to process this video
    }

    with open(json_path, "w") as f:                    # write JSON to disk
        json.dump(report, f, indent=2)

    print(f"  Report saved → {json_path}")

    return report                                      # return for summary table


# ── Summary table ─────────────────────────────────────────────────────────────

def print_summary(reports: list):
    """Print a summary table of all processed videos."""
    good = sum(1 for r in reports if r["score"] >= 80)    # count GOOD videos
    okay = sum(1 for r in reports if 60 <= r["score"] < 80)   # count OKAY videos
    bad  = sum(1 for r in reports if r["score"] < 60)    # count BAD videos

    print(f"\n{'═' * 63}")
    print(f"  Quality Summary — All Videos")
    print(f"{'═' * 63}")
    print(f"  {'Video':<28} {'Score':<7} {'Verdict':<12} Issues")
    print(f"  {'─'*28} {'─'*6} {'─'*11} {'─'*20}")

    for r in reports:                                  # one row per video
        name    = r["clip_name"][:27]                  # truncate long names to fit column
        sc      = r["score"]                           # numeric score
        vd      = r["verdict"]                         # verdict string with emoji
        iss     = ", ".join(r["issues"]) if r["issues"] else ""   # comma-joined issues
        print(f"  {name:<28} {sc:<7} {vd:<12} {iss}")

    print(f"{'═' * 63}")
    print(f"  Total: {len(reports)} videos | Good: {good} | Okay: {okay} | Bad: {bad}")
    print(f"{'═' * 63}\n")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":

    if len(sys.argv) != 2:                             # must have exactly one argument
        print("Usage:")
        print("  Single video : python pipeline/quality_check.py assets/videos/WashingCup.mp4")
        print("  All videos   : python pipeline/quality_check.py assets/videos/")
        sys.exit(1)

    target  = Path(sys.argv[1])                        # the path the user provided
    reports = []                                       # collect all report dicts

    if target.is_dir():                                # user passed a folder
        # find all MP4 files in that folder (non-recursive, top-level only)
        videos = sorted(target.glob("*.mp4"))
        if not videos:                                 # nothing found
            print(f"[ERROR] No .mp4 files found in {target}")
            sys.exit(1)
        print(f"\n  Found {len(videos)} video(s) in {target}")
        for v in videos:                               # process each video in turn
            reports.append(analyse_video(v))

    elif target.is_file():                             # user passed a single file
        if target.suffix.lower() != ".mp4":            # must be an MP4
            print(f"[ERROR] Not an MP4 file: {target}")
            sys.exit(1)
        reports.append(analyse_video(target))          # process just this one video

    else:                                              # path doesn't exist
        print(f"[ERROR] Path not found: {target}")
        sys.exit(1)

    if len(reports) > 1:                               # only show summary for multiple videos
        print_summary(reports)
