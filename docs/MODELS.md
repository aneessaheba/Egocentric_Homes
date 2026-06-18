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
