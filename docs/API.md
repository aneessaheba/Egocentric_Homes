# Pipeline API Reference

This document describes the public interface exposed by each pipeline module.

## config.py

All constants can be imported directly:

```python
from config import VIDEOS_DIR, REJECTED_DIR, QUALITY_REJECT_THRESHOLD
```

Key constants:

| Constant                  | Default             | Description                        |
|---------------------------|---------------------|------------------------------------|
| `VIDEOS_DIR`              | `assets/videos`     | Root input directory               |
| `REJECTED_DIR`            | `assets/videos/rejected` | Quarantine folder             |
| `QUALITY_REJECT_THRESHOLD`| 60                  | Minimum passing quality score      |
| `WHISPER_MODEL`           | `"base"`            | Whisper model size for transcripts |
| `SEG_FRAME_INTERVAL`      | 9                   | SAM2 inference every N frames      |

## utils.py

```python
from utils import load_json, save_json, video_info, Timer, print_progress
```

### `load_json(path) -> dict | list | None`
Load a JSON file; returns `None` if the file doesn't exist.

### `save_json(data, path, indent=2)`
Write data to JSON, creating parent directories as needed.

### `video_info(video_path) -> dict`
Return `{width, height, fps, total_frames, duration_sec}` without decoding frames.

### `Timer`
Simple wall-clock timer with `elapsed()` and `elapsed_str()` methods.

## validators.py

```python
from validators import validate_video_path, validate_score, validate_language
```

All validators **raise `ValueError`** on bad input and **return the value** on success.

| Function                        | Validates                             |
|---------------------------------|---------------------------------------|
| `validate_video_path(path)`     | File exists, supported extension      |
| `validate_video_dir(path)`      | Directory exists, contains .mp4 files |
| `validate_score(v, lo, hi)`     | Numeric in `[lo, hi]`                 |
| `validate_confidence(v)`        | Float in `[0, 1]`                     |
| `validate_fps(fps)`             | Positive float                        |
| `validate_resolution(w, h)`     | Both dimensions > 0                   |
| `validate_duration(secs)`       | 1–300 seconds                         |
| `validate_language(code)`       | ISO 639-1 code in supported set       |
| `validate_clip_name(name)`      | Letters, digits, `_`, `-` only        |

## logger.py

```python
from logger import get_logger
log = get_logger("my_module", log_dir="logs/")
log.info("Processing started")
log.stage_start("hand_pose")
# ... do work ...
log.stage_done("hand_pose")
```

### `get_logger(name, level, log_dir, verbose) -> Logger`
Factory that returns a configured `Logger` instance.

### `Logger.stage_start(name)` / `Logger.stage_done(name)`
Log stage boundaries and compute elapsed time automatically.

## exporter.py

```python
from exporter import to_json, to_csv, to_coco, summary_stats
```

| Function                              | Output                            |
|---------------------------------------|-----------------------------------|
| `to_json(data, path)`                 | Pretty-printed JSON file          |
| `to_csv(frames, path)`                | One CSV row per frame             |
| `to_coco(frames, clip_name, path)`    | COCO-compatible JSON              |
| `summary_stats(frames) -> dict`       | Aggregated stats dict             |

## visualizer.py

```python
from visualizer import draw_hand_landmarks, draw_quality_hud, open_video_writer
```

All drawing functions modify frames **in-place** and return `None` except
`draw_depth_heatmap` which returns a new BGR frame, and `open_video_writer`
which returns a `cv2.VideoWriter`.

| Function                            | Effect                                   |
|-------------------------------------|------------------------------------------|
| `draw_hand_landmarks(frame, lms)`   | Draw skeleton + keypoints on frame       |
| `draw_arm_skeleton(frame, kps)`     | Draw arm bone connections on frame       |
| `draw_depth_heatmap(depth_arr)`     | Return INFERNO-colourmap depth frame     |
| `draw_quality_hud(frame, score)`    | Overlay score badge top-right            |
| `open_video_writer(path, fps, w, h)`| Open VideoWriter; raises on failure      |

## batch_runner.py

```python
from batch_runner import run_batch, collect_clips
```

### `collect_clips(videos_dir, resume=False) -> list[Path]`
Return sorted `.mp4` clips from `videos_dir`. If `resume=True`, skip clips
that already have a `*_full.json` annotation in `ANNOTATIONS_DIR`.

### `run_batch(videos_dir, workers=2, resume=False, batch_size=4)`
Process all clips using a `ProcessPoolExecutor`. Prints per-clip status and
a summary on completion.

## Error handling

All pipeline functions raise **`ValueError`** for bad inputs (wrong file type,
missing path, out-of-range value) and **`RuntimeError`** for unrecoverable
runtime failures (model load error, VideoWriter open failure).

Use the validators module to catch input errors before calling expensive
inference functions:

```python
from validators import validate_video_path, validate_language
path = validate_video_path("my_clip.mp4")
lang = validate_language("hi")
```

## Configuration override

All `config.py` constants can be overridden at runtime by setting environment
variables before importing the module:

```bash
QUALITY_REJECT_THRESHOLD=70 python pipeline/quality_gate.py assets/videos/
```

Or by patching the module after import:

```python
import config
config.QUALITY_REJECT_THRESHOLD = 70
```

## Thread safety

`Logger` instances are **not** thread-safe. Create one instance per worker
process in parallel workloads:

```python
def process_clip(clip_path):
    log = get_logger("worker")
    log.stage_start("hand_pose")
    # ...
```

All other utility functions (`load_json`, `save_json`, `video_info`) are
stateless and safe to call from multiple threads.
