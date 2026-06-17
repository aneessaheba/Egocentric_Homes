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
