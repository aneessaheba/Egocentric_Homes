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
