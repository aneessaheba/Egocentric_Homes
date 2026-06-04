#!/usr/bin/env python3
"""
transcribe_videos.py
────────────────────
Produces granular action subtitles for egocentric GoPro clips by combining:

  • OpenAI Whisper  — transcribes spoken narration from audio
  • GPT-4o Vision  — describes visual actions from sampled frames
                     (e.g. "opening a cupboard", "reaching for the cleaner")

Both streams are merged into a single dense subtitle timeline, written as:
  <clip>.srt          — subtitle file (merged narration + visual actions)
  <clip>.txt          — plain transcript
  <clip>.json         — HomeHands narration annotation
  <clip>_subtitled.mp4 — video with burned-in subtitles

Usage
─────
  export OPENAI_API_KEY="sk-..."        # required for GPT-4o vision
  python transcribe_videos.py

Requirements
────────────
  brew install ffmpeg
  pip install openai openai-whisper opencv-python
"""

import base64
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


# ── CONFIG ────────────────────────────────────────────────────────────────────
VIDEOS_DIR      = Path("assets/videos")
OUTPUT_DIR      = Path("assets/videos/subtitled")
ANNOT_DIR       = Path("assets/data/narrations")

WHISPER_MODEL   = "base"      # tiny | base | small | medium | large
FRAME_INTERVAL  = 2.0         # sample one frame every N seconds for visual captioning
VISION_MODEL    = "gpt-4o"    # GPT-4o supports vision; gpt-4-turbo also works
# ──────────────────────────────────────────────────────────────────────────────


# ── DEPENDENCY CHECKS ─────────────────────────────────────────────────────────

def check_ffmpeg():
    if shutil.which("ffmpeg") is None:
        sys.exit(
            "\n[ERROR] ffmpeg not found.\n"
            "  macOS : brew install ffmpeg\n"
            "  Ubuntu: sudo apt-get install ffmpeg\n"
        )


def ensure_packages():
    """Install missing Python packages automatically."""
    packages = {
        "whisper":  "openai-whisper",
        "openai":   "openai",
        "cv2":      "opencv-python",
    }
    for module, pip_name in packages.items():
        try:
            __import__(module)
        except ImportError:
            print(f"[setup] Installing {pip_name} …")
            subprocess.run(
                [sys.executable, "-m", "pip", "install", pip_name],
                check=True,
            )


def get_openai_client():
    """Return an OpenAI client, aborting if the API key is missing."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        sys.exit(
            "\n[ERROR] OPENAI_API_KEY not set.\n"
            "  export OPENAI_API_KEY='sk-...'\n"
        )
    from openai import OpenAI
    return OpenAI(api_key=api_key)


# ── AUDIO / WHISPER ───────────────────────────────────────────────────────────

def extract_audio(video_path: Path, audio_path: Path):
    """Extract 16 kHz mono WAV — Whisper's native format."""
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(video_path),
         "-ac", "1", "-ar", "16000", "-vn", str(audio_path)],
        check=True, capture_output=True,
    )


def transcribe_audio(audio_path: Path, model) -> list[dict]:
    """
    Run Whisper on the WAV file.
    Returns segments: [{"start": float, "end": float, "text": str, "source": "whisper"}]
    word_timestamps=True gives finer per-word timing for short narrations.
    """
    result = model.transcribe(
        str(audio_path),
        language="en",
        fp16=False,
        word_timestamps=True,   # finer timing, better for short action narrations
    )
    segments = []
    for seg in result["segments"]:
        segments.append({
            "start":  round(float(seg["start"]), 3),
            "end":    round(float(seg["end"]),   3),
            "text":   seg["text"].strip(),
            "source": "whisper",
        })
    return segments


# ── FRAME SAMPLING / GPT-4o VISION ────────────────────────────────────────────

def get_video_duration(video_path: Path) -> float:
    """Use ffprobe to get the video duration in seconds."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ],
        capture_output=True, text=True, check=True,
    )
    return float(result.stdout.strip())


def extract_frame(video_path: Path, timestamp: float) -> bytes | None:
    """
    Extract a single JPEG frame at `timestamp` seconds.
    Returns raw JPEG bytes, or None if extraction fails.
    Resize to 512px wide to keep API payload small.
    """
    result = subprocess.run(
        [
            "ffmpeg", "-y",
            "-ss", str(timestamp),
            "-i", str(video_path),
            "-vframes", "1",
            "-vf", "scale=512:-1",    # resize: 512px wide, keep aspect ratio
            "-f", "image2pipe",
            "-vcodec", "mjpeg",
            "pipe:1",                  # output to stdout
        ],
        capture_output=True,
    )
    return result.stdout if result.returncode == 0 and result.stdout else None


def describe_frame(jpeg_bytes: bytes, client, timestamp: float) -> str | None:
    """
    Send a single frame to GPT-4o and ask for a short action description.
    Returns a concise phrase like "opening a cupboard" or None on failure.
    """
    b64 = base64.b64encode(jpeg_bytes).decode("utf-8")

    try:
        response = client.chat.completions.create(
            model=VISION_MODEL,
            max_tokens=30,       # keep it short — we only want a brief action label
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are labeling egocentric (first-person) household activity video. "
                        "Describe ONLY the physical action visible in the frame in 3-6 words. "
                        "Focus on hand/body movement and objects. "
                        "Examples: 'picking up the cup', 'opening the cupboard door', "
                        "'reaching for the cleaning spray', 'folding the shirt sleeve'. "
                        "Do NOT describe the scene background. "
                        "Reply with ONLY the short action phrase, no punctuation."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{b64}",
                                "detail": "low",   # low detail = faster + cheaper
                            },
                        },
                        {
                            "type": "text",
                            "text": f"Frame at {timestamp:.1f}s. What action is happening?",
                        },
                    ],
                },
            ],
        )
        text = response.choices[0].message.content.strip()
        return text if text else None
    except Exception as e:
        print(f"          [vision warn] {e}")
        return None


def caption_frames(video_path: Path, client) -> list[dict]:
    """
    Sample a frame every FRAME_INTERVAL seconds, describe each with GPT-4o.
    Returns segments: [{"start": float, "end": float, "text": str, "source": "vision"}]
    """
    duration = get_video_duration(video_path)
    timestamps = [
        round(t, 2)
        for t in [i * FRAME_INTERVAL for i in range(int(duration / FRAME_INTERVAL))]
        if t < duration
    ]

    print(f"        Sampling {len(timestamps)} frame(s) every {FRAME_INTERVAL}s …")
    segments = []

    for i, ts in enumerate(timestamps):
        jpeg = extract_frame(video_path, ts)
        if jpeg is None:
            continue

        end_ts = timestamps[i + 1] if i + 1 < len(timestamps) else round(ts + FRAME_INTERVAL, 2)
        label  = describe_frame(jpeg, client, ts)

        if label:
            segments.append({
                "start":  ts,
                "end":    end_ts,
                "text":   label,
                "source": "vision",
            })
            print(f"          [{ts:6.1f}s] {label}")

    return segments


# ── MERGE STREAMS ─────────────────────────────────────────────────────────────

def merge_segments(whisper_segs: list[dict], vision_segs: list[dict]) -> list[dict]:
    """
    Merge Whisper narration and GPT-4o visual captions into one timeline.

    Strategy:
      - Whisper narration takes priority: wherever speech is detected,
        it replaces the visual caption for that time window.
      - Visual captions fill the gaps between spoken segments.
      - Adjacent visual captions with identical text are collapsed into one.
    """
    # Build a set of time windows covered by Whisper speech
    speech_windows = [(s["start"], s["end"]) for s in whisper_segs]

    def overlaps_speech(start: float, end: float) -> bool:
        for ws, we in speech_windows:
            # overlapping if not completely before or after
            if start < we and end > ws:
                return True
        return False

    # Keep only vision segments that don't overlap with speech
    filtered_vision = [
        s for s in vision_segs
        if not overlaps_speech(s["start"], s["end"])
    ]

    # Collapse consecutive identical vision labels
    collapsed = []
    for seg in filtered_vision:
        if collapsed and collapsed[-1]["source"] == "vision" \
                     and collapsed[-1]["text"].lower() == seg["text"].lower():
            collapsed[-1]["end"] = seg["end"]   # extend existing segment
        else:
            collapsed.append(dict(seg))

    # Combine and sort by start time
    all_segs = whisper_segs + collapsed
    all_segs.sort(key=lambda s: s["start"])
    return all_segs


# ── OUTPUT WRITERS ────────────────────────────────────────────────────────────

def _srt_ts(seconds: float) -> str:
    h  = int(seconds // 3600)
    m  = int((seconds % 3600) // 60)
    s  = int(seconds % 60)
    ms = int(round((seconds - int(seconds)) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(segments: list[dict], srt_path: Path):
    lines = []
    for i, seg in enumerate(segments, 1):
        lines += [
            str(i),
            f"{_srt_ts(seg['start'])} --> {_srt_ts(seg['end'])}",
            seg["text"],
            "",
        ]
    srt_path.write_text("\n".join(lines), encoding="utf-8")


def write_txt(segments: list[dict], txt_path: Path):
    txt_path.write_text(
        "\n".join(seg["text"] for seg in segments),
        encoding="utf-8",
    )


def write_json(clip_name: str, segments: list[dict], json_path: Path):
    """
    HomeHands annotation JSON — includes source field so you can tell
    which labels came from narration vs. visual detection.
    {
      "clip": "WashingCup.mp4",
      "narrations": [
        {"start": 0.0, "end": 2.3, "text": "...", "source": "whisper"|"vision"}
      ]
    }
    """
    payload = {
        "clip": clip_name,
        "narrations": [
            {
                "start":  seg["start"],
                "end":    seg["end"],
                "text":   seg["text"],
                "source": seg.get("source", "unknown"),
            }
            for seg in segments
        ],
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


# ── SUBTITLE BURNING ──────────────────────────────────────────────────────────

def burn_subtitles(video_path: Path, srt_path: Path, out_path: Path):
    """
    Burn subtitles into the video using ffmpeg.
    White text, black 2px outline, bottom-center, Arial 20pt.
    SRT is copied to a temp path with no spaces to avoid ffmpeg filter issues.
    """
    style = (
        "FontName=Arial,FontSize=20,"
        "PrimaryColour=&H00FFFFFF,"
        "OutlineColour=&H00000000,"
        "Outline=2,Shadow=0,"
        "Alignment=2,MarginV=25"
    )
    with tempfile.NamedTemporaryFile(suffix=".srt", delete=False) as tmp:
        tmp_srt = Path(tmp.name)
    shutil.copy(srt_path, tmp_srt)
    try:
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", str(video_path),
                "-vf", f"subtitles={tmp_srt}:force_style='{style}'",
                "-c:a", "copy",
                str(out_path),
            ],
            check=True,
        )
    finally:
        tmp_srt.unlink(missing_ok=True)


# ── PER-CLIP PIPELINE ─────────────────────────────────────────────────────────

def process_clip(video_path: Path, whisper_model, openai_client) -> dict:
    stem = video_path.stem

    srt_path  = ANNOT_DIR  / f"{stem}.srt"
    txt_path  = ANNOT_DIR  / f"{stem}.txt"
    json_path = ANNOT_DIR  / f"{stem}.json"
    out_video = OUTPUT_DIR / f"{stem}_subtitled.mp4"

    print(f"\n{'─' * 60}")
    print(f"  Clip: {video_path.name}")
    print(f"{'─' * 60}")

    # ── Step 1: Whisper — spoken narration ───────────────────
    print("  [1/4] Extracting audio + transcribing narration …")
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        audio_path = Path(tmp.name)
    try:
        extract_audio(video_path, audio_path)
        whisper_segs = transcribe_audio(audio_path, whisper_model)
    finally:
        audio_path.unlink(missing_ok=True)

    print(f"        → {len(whisper_segs)} narration segment(s)")
    for seg in whisper_segs:
        print(f"          [{seg['start']:6.2f}s] \"{seg['text']}\"")

    # ── Step 2: GPT-4o — visual action captions ──────────────
    print("  [2/4] Captioning frames with GPT-4o vision …")
    vision_segs = caption_frames(video_path, openai_client)
    print(f"        → {len(vision_segs)} visual segment(s)")

    # ── Step 3: Merge both streams ───────────────────────────
    print("  [3/4] Merging narration + visual captions …")
    merged = merge_segments(whisper_segs, vision_segs)
    print(f"        → {len(merged)} total subtitle entries")

    # ── Step 4: Write annotation files ───────────────────────
    write_srt(merged, srt_path)
    write_txt(merged, txt_path)
    write_json(video_path.name, merged, json_path)

    # ── Step 5: Burn subtitles into video ────────────────────
    print("  [4/4] Burning subtitles into video …")
    burn_subtitles(video_path, srt_path, out_video)

    print(f"  ✓  Saved → {out_video.name}")
    return {"clip": video_path.name, "status": "ok", "segments": len(merged)}


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    check_ffmpeg()
    ensure_packages()

    import whisper as whisper_lib

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ANNOT_DIR.mkdir(parents=True, exist_ok=True)

    clips = sorted(
        p for p in VIDEOS_DIR.iterdir()
        if p.suffix.lower() in {".mp4", ".mov", ".avi"}
        and "_subtitled" not in p.stem
        and p.parent == VIDEOS_DIR
    )

    if not clips:
        sys.exit(f"\n[ERROR] No video files found in {VIDEOS_DIR}\n")

    print(f"\nFound {len(clips)} clip(s).")
    print(f"Loading Whisper '{WHISPER_MODEL}' model …")
    whisper_model  = whisper_lib.load_model(WHISPER_MODEL)
    openai_client  = get_openai_client()
    print("Ready.\n")

    summary = []
    for clip in clips:
        try:
            result = process_clip(clip, whisper_model, openai_client)
        except Exception as exc:
            print(f"  ✗  Error: {exc}")
            result = {"clip": clip.name, "status": "error", "error": str(exc)}
        summary.append(result)

    print(f"\n{'═' * 60}")
    print("  SUMMARY")
    print(f"{'═' * 60}")
    ok  = [s for s in summary if s["status"] == "ok"]
    err = [s for s in summary if s["status"] != "ok"]
    for s in ok:
        print(f"  ✓  {s['clip']}  —  {s['segments']} subtitle entries")
    for s in err:
        print(f"  ✗  {s['clip']}  —  {s.get('error', 'unknown error')}")
    print(f"\n  {len(ok)} succeeded · {len(err)} failed")
    print(f"  Subtitled videos : {OUTPUT_DIR}/")
    print(f"  Annotations      : {ANNOT_DIR}/\n")


if __name__ == "__main__":
    main()
