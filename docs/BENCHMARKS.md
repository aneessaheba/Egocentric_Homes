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
