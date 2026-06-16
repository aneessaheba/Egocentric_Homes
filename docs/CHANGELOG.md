# Changelog

All notable changes to this project will be documented here.

## [Unreleased]

### Added
- `pipeline/config.py` — centralised constants for all pipeline modules
- `pipeline/logger.py` — shared logging with colour output and stage timing
- `pipeline/validators.py` — input validation helpers

- `pipeline/exporter.py` — JSON, CSV, and COCO export for annotations
- `pipeline/visualizer.py` — OpenCV drawing helpers for annotation overlays
- `pipeline/batch_runner.py` — parallel multi-clip processing

- `pipeline/arm_pose.py` — `--batch` mode for processing entire video folders
- `pipeline/run_pipeline.py` — `--resume` flag to skip already-processed clips
- `pipeline/quality_gate.py` — `--dry-run` flag to preview rejections

- `pipeline/transcribe.py` — `--language` flag for non-English narration
- `pipeline/utils.py` — shared JSON I/O, video metadata, timer utilities

### Changed
- Removed HomeDepth section from dataset page (dataset not yet public)
