# HomeHands-50 Dataset

HomeHands-50 is an egocentric video dataset of everyday domestic activities,
captured by contributors wearing head-mounted or wrist-mounted cameras.

## Overview

| Property     | Value                      |
|--------------|----------------------------|
| Total clips  | 50 (target)                |
| Duration     | 20–60 sec per clip         |
| Resolution   | 720p – 1080p               |
| Frame rate   | 24 – 30 fps                |
| Activities   | Kitchen, laundry, cleaning |

## Derived annotations

| Stream        | File pattern                   | Format          |
|---------------|--------------------------------|-----------------|
| Hand pose     | `hand_pose/<clip>.json`        | JSON (landmarks)|
| Arm pose      | `arm_pose/<clip>.json`         | JSON (keypoints)|
| Segmentation  | `segmentation/<clip>.json`     | JSON (RLE masks)|
| Depth         | `depth/<clip>_depth.npy`       | NumPy float32   |
| Transcription | `transcripts/<clip>.json`      | JSON (Whisper)  |

## Activity distribution

| Activity category   | Clips |
|---------------------|-------|
| Washing dishes      | 12    |
| Chopping / cutting  | 10    |
| Cooking on hob      | 9     |
| Making tea / coffee | 8     |
| Cleaning counter    | 6     |
| Folding laundry     | 5     |

## Quality statistics

| Metric               | Value    |
|----------------------|----------|
| Mean quality score   | 76.4     |
| Acceptance rate      | 88%      |
| Clips rejected       | 6        |
| Mean hand visibility | 82%      |
| Mean clip duration   | 34.2 sec |
