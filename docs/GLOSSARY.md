# Glossary

**Clip**: A single `.mp4` video file submitted by a contributor.

**Quality score**: A 0–100 composite score computed by `quality_check.py`
from blur, brightness, contrast, hand visibility, and motion coverage metrics.

**Quality gate**: The automated step that quarantines clips scoring below 60
by moving them to `assets/videos/rejected/`.
