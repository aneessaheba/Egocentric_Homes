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

### Fixed
- Fixed left/right arm identity swap in YOLO arm tracking
- Filtered low-confidence YOLO arm keypoints (threshold 0.4)
- Fixed MediaPipe handedness collision bug in hand_pose.py

## [0.2.0] — 2026-06-05

### Added
- SAM 2.1 Hiera-Tiny segmentation stage
- Depth-Anything-V2 Metric Indoor depth stage
- Whisper transcription stage
- Quality check scoring with weighted metrics
- Quality gate to quarantine low-scoring clips

## [0.1.0] — 2026-06-01

### Added
- Initial EgoLoop platform scaffolding (index, campaigns, dataset, pipeline pages)
- MediaPipe hand pose annotation script
- YOLOv8n arm pose annotation script
- Basic homepage with hero, stats, and campaign card

## [0.3.0] — 2026-06-17

### Added
- `docs/API.md` — full public API reference for all pipeline modules
- `docs/MODELS.md` — detailed model cards for each inference stage
- `docs/FAQ.md` — answers to common setup and usage questions
- `docs/SETUP.md` — step-by-step environment setup guide
- `docs/CONTRIBUTING.md` — contributor guidelines and clip naming rules

### Changed
- Incremented campaign slot display as new clips were accepted
- Batch runner now shows per-clip status and final pass/fail summary

### Fixed
- Exporter `to_csv` no longer crashes on frames with missing keys
