# Models Used in the HomeHands Pipeline

## Hand Pose — MediaPipe Hands

- **Model**: MediaPipe Hands (Lite / Full)
- **Output**: 21 3D landmarks per hand, up to 2 hands per frame
- **Confidence threshold**: 0.5 (configurable in `config.py`)
- **Script**: `pipeline/hand_pose.py`
