# Benchmarks

Performance measurements for the HomeHands pipeline on reference hardware.

## Reference hardware

| Config | CPU              | GPU            | RAM   |
|--------|------------------|----------------|-------|
| A      | AMD Ryzen 7 5800 | RTX 3060 12 GB | 32 GB |
| B      | Apple M2 Pro     | MPS            | 16 GB |
| C      | Intel i7-12700   | RTX 4060 8 GB  | 16 GB |

## Per-stage timing (30-second 1080p clip)

| Stage         | Config A | Config B | Config C |
|---------------|----------|----------|----------|
| Hand pose     | 12 s     | 18 s     | 14 s     |
| Arm pose      | 8 s      | 12 s     | 9 s      |
| Segmentation  | 35 s     | 55 s     | 40 s     |
| Depth         | 28 s     | 45 s     | 32 s     |
| Transcription | 6 s      | 9 s      | 7 s      |
| Quality check | 4 s      | 6 s      | 5 s      |
| **Total**     | **93 s** |**145 s** |**107 s** |

## Throughput

| Config | Clips / hour (30-sec avg) |
|--------|--------------------------|
| A      | ~38                      |
| B      | ~25                      |
| C      | ~33                      |

With `batch_runner.py --workers 2`, throughput roughly doubles for multi-clip
sessions by overlapping CPU pre/post-processing with GPU inference.

## Memory usage per stage

| Stage         | Peak GPU VRAM | Peak System RAM |
|---------------|---------------|-----------------|
| Hand pose     | 0 MB (CPU)    | ~500 MB         |
| Arm pose      | ~480 MB       | ~600 MB         |
| Segmentation  | ~980 MB       | ~1.2 GB         |
| Depth         | ~1500 MB      | ~1.1 GB         |
| Transcription | ~900 MB       | ~800 MB         |

Peak concurrent VRAM (all models loaded): ~2.8 GB.

## Whisper model comparison

| Model  | Size    | Speed (30 s clip) | WER (en) |
|--------|---------|-------------------|----------|
| tiny   | 39 MB   | 2 s               | ~15%     |
| base   | 150 MB  | 6 s               | ~10%     |
| small  | 461 MB  | 15 s              | ~7%      |
| medium | 1.5 GB  | 40 s              | ~5%      |

Pipeline default: `base`. Set `WHISPER_MODEL` in `config.py` to change.
