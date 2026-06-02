"""config.py — Centralised pipeline configuration.
"""
from pathlib import Path

# ── Directories ───────────────────────────────────────────────────
ASSETS_DIR    = Path("assets")
VIDEOS_DIR    = ASSETS_DIR / "videos"
PROCESSED_DIR = ASSETS_DIR / "processed"
REJECTED_DIR  = VIDEOS_DIR / "rejected"

# ── Processed sub-directories ─────────────────────────────────────
ANNOTATIONS_DIR = PROCESSED_DIR / "annotations"
HAND_DIR        = PROCESSED_DIR / "hand_pose"
ARM_DIR         = PROCESSED_DIR / "arm_pose"
SEG_DIR         = PROCESSED_DIR / "segmentation"
DEPTH_DIR       = PROCESSED_DIR / "depth"
QUALITY_DIR     = PROCESSED_DIR / "quality"
TRANSCRIPT_DIR  = PROCESSED_DIR / "transcripts"

# ── Model paths ───────────────────────────────────────────────────
YOLO_POSE_MODEL  = Path("yolov8n-pose.pt")
SAM2_CHECKPOINT  = Path("models/sam2.1_hiera_tiny.pt")
DEPTH_CHECKPOINT = Path("models/depth_anything_v2_metric_indoor_small.pth")
WHISPER_MODEL    = "base"

# ── Processing thresholds ─────────────────────────────────────────
QUALITY_REJECT_THRESHOLD = 60
HAND_CONFIDENCE_MIN      = 0.5
ARM_CONFIDENCE_MIN       = 0.4
ARM_EMA_ALPHA            = 0.3
ARM_TEMPORAL_BUFFER      = 3
SEG_FRAME_INTERVAL       = 9
DEPTH_RESIZE_WIDTH       = 518

# ── Quality score weights ─────────────────────────────────────────
QUALITY_WEIGHTS = {
    "blur":            0.30,
    "brightness":      0.15,
    "contrast":        0.15,
    "hand_visibility": 0.25,
    "motion_coverage": 0.15,
}
