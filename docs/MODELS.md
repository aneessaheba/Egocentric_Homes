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
