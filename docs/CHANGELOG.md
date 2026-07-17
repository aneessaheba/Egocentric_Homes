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

## [0.3.1] — 2026-06-20

### Added
- Model size and VRAM usage reference table in `docs/MODELS.md`
- Hardware requirements FAQ entries for NVIDIA and Apple Silicon
- Troubleshooting section in `docs/SETUP.md`

## [0.3.2] — 2026-06-22

### Added
- `docs/MODELS.md` model swap guide and Whisper size selection
- Activity categories and payout structure in `docs/CONTRIBUTING.md`
- Privacy and logging FAQ entries
- Configuration override documentation in `docs/API.md`

### Changed
- Campaign slot counter updated daily as clips are accepted

## [0.3.3] — 2026-06-23

### Added
- Thread safety notes for Logger in `docs/API.md`
- Model licence reference table in `docs/MODELS.md`
- Data privacy guidelines for contributors
- Video technical requirements table in FAQ
- Linting/formatting guide in SETUP.md

### Changed
- Campaign acceptance rate display updated

## [0.3.4] — 2026-06-24

### Added
- FAQ sections for payments, leaderboard, and video specifications
- Pipeline extension guide in `docs/API.md`
- Setup testing guide in `docs/SETUP.md`
- Contributor setup test workflow in `docs/CONTRIBUTING.md`

## [0.4.0] — 2026-06-25

### Added
- `docs/MODELS.md` inference pipeline flow diagram
- Environment variable reference table in `docs/SETUP.md`
- Post-acceptance contributor workflow in `docs/CONTRIBUTING.md`
- Dataset release timeline and account management FAQ entries

### Changed
- Campaign count tracking now incremented daily

## [0.4.1] — 2026-06-26

### Added
- `docs/GLOSSARY.md` — key terminology definitions
- In-review status and FAQ for clips under manual review
- Model licence and inference flow documentation updates

## [0.4.2] — 2026-06-27

### Added
- Edge case FAQ entries (no audio, low hand visibility, multiple people)
- SAM 2 to SAM 2.1 upgrade guide
- Virtual environment setup instructions
- API changelog reference and stability note

### Fixed
- Corrected weight column alignment in quality scoring FAQ table

## [0.4.3] — 2026-06-28

### Added
- Depth output format and numpy visualisation example in MODELS.md
- Full annotation JSON schema in API.md
- GPU acceleration install commands in SETUP.md
- Recording equipment and lighting tips in CONTRIBUTING.md
- Debugging guide for stage failures in FAQ.md
- GLOSSARY: landmark confidence, segmentation mask definitions

## [0.4.4] — 2026-06-29

### Added
- Whisper transcript JSON structure example in MODELS.md
- Installation verification commands in SETUP.md
- Logging integration example in API.md
- Common rejection reasons guide in CONTRIBUTING.md
- RLE and optical flow glossary definitions
- International contributors FAQ section

## [0.5.0] — 2026-06-30

### Added
- Arm pose COCO keypoint index table in MODELS.md
- Dependency update guide in SETUP.md
- Deprecation and versioning policy in API.md
- Detailed review process steps in CONTRIBUTING.md
- Multi-camera FAQ entry
- Glossary additions: annotation schema, stage, optical flow, RLE

### Changed
- Campaign slot count now at 72 accepted clips

## [0.7.2] — 2026-07-16

### Added
- `docs/DATASET.md` — HomeHands-50 overview, distributions, quality stats
- `docs/BENCHMARKS.md` — per-stage timing, throughput, memory, and Whisper comparison

### Changed
- Campaign slot count at 74
