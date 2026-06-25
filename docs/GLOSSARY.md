# Glossary

**Clip**: A single `.mp4` video file submitted by a contributor.

**Quality score**: A 0–100 composite score computed by `quality_check.py`
from blur, brightness, contrast, hand visibility, and motion coverage metrics.

**Quality gate**: The automated step that quarantines clips scoring below 60
by moving them to `assets/videos/rejected/`.

**Annotation**: A JSON file describing frame-level metadata for a clip —
landmarks, keypoints, masks, depth, transcripts, and quality scores.

**Full annotation**: The merged JSON (`*_full.json`) produced by
`run_pipeline.py` combining all stage outputs into one file per clip.

**Acceptance rate**: The percentage of a contributor's submitted clips
that pass the quality gate (score ≥ 60).

**Egocentric video**: Video recorded from the first-person perspective —
the camera is worn by the person performing the activity.

**Hand visibility**: The fraction of video frames in which at least one
hand is detected. One of five metrics contributing to the quality score.

**Motion coverage**: A measure of how much of the frame contains meaningful
motion across the clip. Low motion coverage may indicate the subject was
stationary or the camera was not recording the intended activity.

**COCO format**: A standardised JSON structure for computer vision datasets
(images, categories, annotations). See https://cocodataset.org for the spec.
EgoLoop supports COCO export via `pipeline/exporter.py`.

**Landmark**: A 2D or 3D point representing a joint or anatomical location.
MediaPipe Hands produces 21 landmarks per hand; YOLOv8 produces 17 COCO
pose landmarks per person.

**EMA (Exponential Moving Average)**: A smoothing technique applied to arm
keypoints over time to reduce jitter from frame-to-frame detection noise.
