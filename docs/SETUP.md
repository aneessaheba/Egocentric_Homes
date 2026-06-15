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
