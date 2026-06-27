# Models Used in the HomeHands Pipeline

## Hand Pose — MediaPipe Hands

- **Model**: MediaPipe Hands (Lite / Full)
- **Output**: 21 3D landmarks per hand, up to 2 hands per frame
- **Confidence threshold**: 0.5 (configurable in `config.py`)
- **Script**: `pipeline/hand_pose.py`

## Arm Pose — YOLOv8n-pose

- **Model**: `yolov8n-pose.pt` (ultralytics)
- **Output**: 17 COCO keypoints; pipeline uses indices 5-10 (shoulders, elbows, wrists)
- **EMA smoothing**: alpha = 0.3, temporal buffer = 3 frames
- **Confidence threshold**: 0.4 per keypoint
- **Script**: `pipeline/arm_pose.py`

## Segmentation — SAM 2.1 Hiera-Tiny

- **Model**: `sam2.1_hiera_tiny.pt`
- **Prompted by**: Hand bounding boxes from `hand_pose.py`
- **Inference interval**: Every 9th frame (configurable via `SEG_FRAME_INTERVAL`)
- **Quantisation**: 4-bit to reduce VRAM usage
- **Script**: `pipeline/segmentation.py`

## Depth — Depth-Anything-V2 Metric Indoor Small

- **Model**: `depth_anything_v2_metric_indoor_small.pth`
- **Output**: Per-pixel metric depth map (metres)
- **Resize width**: 518 px (configurable via `DEPTH_RESIZE_WIDTH`)
- **Script**: `pipeline/depth.py`

## Transcription — OpenAI Whisper

- **Model**: `base` (multilingual, ~150 MB)
- **Audio extraction**: 16 kHz mono WAV via ffmpeg
- **Language**: auto-detected by default; override with `--language CODE`
- **Script**: `pipeline/transcribe.py`

## Quality Scoring — MediaPipe + OpenCV

No pretrained model — computed from frame statistics:
- **Blur**: Laplacian variance
- **Brightness / Contrast**: Mean and std of grayscale channel
- **Hand visibility**: Fraction of frames with detected hands
- **Motion coverage**: Optical flow magnitude distribution
- **Script**: `pipeline/quality_check.py`

## Model size comparison

| Stage          | Model                             | Size    | VRAM  |
|----------------|-----------------------------------|---------|-------|
| Hand pose      | MediaPipe Hands Lite              | ~9 MB   | CPU   |
| Arm pose       | YOLOv8n-pose                      | ~6 MB   | ~0.5 GB |
| Segmentation   | SAM 2.1 Hiera-Tiny (4-bit)        | ~38 MB  | ~1 GB |
| Depth          | Depth-Anything-V2 Indoor-S        | ~99 MB  | ~1.5 GB |
| Transcription  | Whisper base                      | ~150 MB | ~1 GB |

All stages can run on a single GPU with 6 GB VRAM.

## Updating models

To swap in a newer checkpoint:

1. Download the new weights file
2. Place it at the path expected by `config.py` (e.g. `models/sam2.1_hiera_tiny.pt`)
3. Run the pipeline — no code changes needed

To use a different Whisper model size, change `WHISPER_MODEL` in `config.py`:

```python
WHISPER_MODEL = "small"   # trades accuracy for speed
WHISPER_MODEL = "medium"  # best quality, slower
```

## Licence notes

| Model                    | Licence           | Commercial use |
|--------------------------|-------------------|----------------|
| MediaPipe Hands          | Apache 2.0        | Yes            |
| YOLOv8n-pose             | AGPL-3.0          | Requires care  |
| SAM 2.1                  | Apache 2.0        | Yes            |
| Depth-Anything-V2        | Apache 2.0        | Yes            |
| Whisper                  | MIT               | Yes            |

Check the respective repositories for the latest licence terms before
using these models in a commercial product.

## Inference pipeline flow

```
raw .mp4
   │
   ├─► hand_pose.py     →  hand_pose/*.json  (MediaPipe)
   ├─► arm_pose.py      →  arm_pose/*.json   (YOLOv8n-pose)
   ├─► segmentation.py  →  segmentation/*.json (SAM 2.1)
   ├─► depth.py         →  depth/*.npy        (Depth-Anything-V2)
   ├─► transcribe.py    →  transcripts/*.json  (Whisper)
   └─► quality_check.py →  quality/*.json     (OpenCV metrics)
          │
          └─► quality_gate.py (quarantine if score < 60)
                   │
                   └─► run_pipeline.py merges all → annotations/*.json
```

## Updating to SAM 2.1 from SAM 2

If you have the older SAM 2 checkpoint (`sam2_hiera_tiny.pt`), upgrade to
SAM 2.1 for better mask quality:

1. Download `sam2.1_hiera_tiny.pt` from the Meta AI repository
2. Replace the file at `models/sam2.1_hiera_tiny.pt`
3. No code changes are needed — the pipeline uses the path from `config.py`

SAM 2.1 improves boundary precision and reduces mask leakage on cluttered
backgrounds common in kitchen environments.

## Depth model output format

Depth-Anything-V2 produces metric depth in metres (not disparity).
The pipeline saves:

- `depth/<clip>_depth.npy` — float32 array of shape `(H, W)`
- `depth/<clip>_depth.json` — metadata: `{min_depth, max_depth, mean_depth, shape}`

To load and visualise:

```python
import numpy as np
import matplotlib.pyplot as plt

depth = np.load("assets/processed/depth/MyClip_depth.npy")
plt.imshow(depth, cmap="inferno")
plt.colorbar(label="Depth (m)")
plt.show()
```
