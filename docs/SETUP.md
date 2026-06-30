# Setting up the HomeHands Pipeline

## Requirements

- Python 3.10+
- CUDA-capable GPU (recommended)
- 8 GB+ RAM

## Installation

```bash
git clone https://github.com/aneessaheba/Egocentric_Homes.git
cd Egocentric_Homes
pip install -r requirements.txt
```

## Directory structure

```
assets/
  videos/          # raw .mp4 clips go here
  processed/
    hand_pose/     # hand landmark JSONs
    arm_pose/      # arm keypoint JSONs
    segmentation/  # SAM2 mask outputs
    depth/         # depth map arrays
    transcripts/   # Whisper transcript JSONs
    quality/       # quality score JSONs
    annotations/   # merged full annotation JSONs
```

## Running the pipeline

```bash
# Process a single video
python pipeline/run_pipeline.py

# Skip already-processed clips
python pipeline/run_pipeline.py --resume

# Run quality gate on all clips
python pipeline/quality_gate.py assets/videos/

# Preview rejections without moving files
python pipeline/quality_gate.py assets/videos/ --dry-run
```

## Batch processing

```bash
python pipeline/batch_runner.py assets/videos/ --workers 2
python pipeline/batch_runner.py assets/videos/ --resume --workers 4
```

## Exporting annotations

```bash
# Convert JSON annotations to CSV
python pipeline/exporter.py assets/processed/annotations/MyClip_full.json csv

# Convert to COCO format
python pipeline/exporter.py assets/processed/annotations/MyClip_full.json coco
```

## Visualising annotations

```bash
# Draw hand landmarks and quality HUD on every frame
python pipeline/visualizer.py assets/videos/MyClip.mp4     assets/processed/annotations/MyClip_full.json
# Output: assets/videos/MyClip_viz.mp4
```

## Troubleshooting

**`FileNotFoundError: Model checkpoint not found`**
Download model checkpoints and place them in the `models/` directory.
See `config.py` for expected paths.

**`No .mp4 files found`**
Ensure your clips are in `assets/videos/` (not a subdirectory) and have
the `.mp4` extension.

**Slow depth inference**
Reduce `DEPTH_RESIZE_WIDTH` in `config.py` from 518 to 384 for ~2× speedup
at a small accuracy cost.

## Linting and formatting

The pipeline codebase uses:
- `black` for code formatting (`black pipeline/`)
- `ruff` for linting (`ruff check pipeline/`)
- `mypy` for type checking (`mypy pipeline/ --ignore-missing-imports`)

These are not enforced by CI yet but are recommended before opening PRs.

## Running tests

```bash
# Validate a clip through all stages (no GPU required for this check)
python pipeline/validators.py  # not yet a standalone script; use in Python

# Quick quality preview
python pipeline/quality_gate.py assets/videos/ --dry-run
```

## Environment variables

| Variable                    | Default | Effect                                  |
|-----------------------------|---------|------------------------------------------|
| `PYTORCH_ENABLE_MPS_FALLBACK` | unset | Enable PyTorch MPS fallback ops (macOS)  |
| `CUDA_VISIBLE_DEVICES`       | all    | Restrict to specific GPU indices         |

Set these before running any pipeline script if needed.

## Virtual environment

We strongly recommend using a virtual environment:

```bash
python3 -m venv homehands_env
source homehands_env/bin/activate   # macOS / Linux
# homehands_env\Scripts\activate  # Windows

pip install -r requirements.txt
```

The `homehands_env/` directory is listed in `.gitignore` and should
never be committed.

## GPU acceleration

### NVIDIA (CUDA)
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Apple Silicon (MPS)
```bash
pip install torch torchvision  # MPS support built-in since PyTorch 2.0
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

### CPU only
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

## Checking your installation

After installing, verify each component:

```bash
# Check Python and torch
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

# Check MediaPipe
python -c "import mediapipe; print('MediaPipe OK')"

# Check Whisper
python -c "import whisper; print('Whisper OK')"

# Check OpenCV
python -c "import cv2; print(cv2.__version__)"
```

## Updating dependencies

To update all packages to their latest compatible versions:

```bash
pip install --upgrade -r requirements.txt
```

If a model checkpoint format changes after an update, the relevant
pipeline script will raise a clear error pointing to the affected file.
Pin versions in `requirements.txt` to avoid unexpected breakage.

## Running on a headless server

For server environments without a display:

```bash
export DISPLAY=:0          # if Xvfb is running
# or use headless OpenCV:
pip install opencv-python-headless
```

Whisper and PyTorch run fine without a display. OpenCV `imshow` calls
are not used by any pipeline script — all output is to JSON files.
