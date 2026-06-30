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

**SAM (Segment Anything Model)**: Meta's promptable segmentation model.
The pipeline uses SAM 2.1 Hiera-Tiny, prompted with hand bounding boxes,
to produce binary masks of the hands and nearby objects in each frame.

**Depth map**: A per-pixel array of distances (in metres) from the camera
produced by Depth-Anything-V2. Stored as `.npy` arrays alongside JSON metadata.

**Whisper**: OpenAI's automatic speech recognition model used to transcribe
narration or speech in submitted clips. Supports 99 languages.

**Batch runner**: `pipeline/batch_runner.py` — processes multiple clips in
parallel using Python's `ProcessPoolExecutor`.

**Resume mode**: When `--resume` is passed to `run_pipeline.py` or
`batch_runner.py`, already-processed clips (those with an existing
`*_full.json`) are skipped, allowing a run to continue after interruption.

**Dry run**: The `--dry-run` flag for `quality_gate.py` — scores clips
and prints what *would* be rejected without actually moving any files.

**Quality weight**: The fractional importance assigned to each quality metric
when computing the composite quality score. Weights are defined in
`config.py` under `QUALITY_WEIGHTS` and must sum to 1.0.

**Quarantine**: When a clip fails the quality gate, it is *quarantined* by
moving it to `assets/videos/rejected/`. The clip is not deleted — it can
be recovered and re-submitted after improvements.

**Landmark confidence**: A float in [0, 1] indicating how confident the
detector is in a keypoint's location. Arm keypoints with confidence < 0.4
are filtered out before saving (see `config.py` / `ARM_CONFIDENCE_MIN`).

**Segmentation mask**: A binary per-pixel array (0 = background, 1 = foreground)
produced by SAM 2.1. Saved as a list of run-length encoded segments in JSON,
or as a numpy bool array when loaded for downstream use.

**Run-length encoding (RLE)**: A compact format for binary masks where
consecutive identical values are stored as (value, count) pairs.
SAM 2.1 segmentation masks are stored in COCO RLE format in the JSON output.

**Optical flow**: A computer vision technique measuring per-pixel motion
between consecutive frames. Used by `quality_check.py` to compute the
motion coverage metric.

**Annotation schema**: The structure of a per-frame annotation dict.
See `docs/API.md` for the full schema with example values.

**Stage**: One step in the pipeline (e.g. `hand_pose`, `segmentation`).
Each stage is implemented as a separate script in `pipeline/` and produces
one JSON file per input clip.

**Clip stem**: The filename without extension. Used as the base name for
all output files from the pipeline. For `WashingCup.mp4`, the stem is
`WashingCup` and outputs are `WashingCup_hand_pose.json`, etc.

**Processed directory**: `assets/processed/` — the root output directory
for all pipeline annotation files, organised into sub-directories per stage.
