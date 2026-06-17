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
