# Frequently Asked Questions

## General

**What is HomeHands-50?**
HomeHands-50 is an egocentric video dataset of everyday domestic tasks,
annotated with hand poses, arm poses, instance segmentation masks,
metric depth maps, and speech transcripts.

**How do I contribute clips?**
Register at the EgoLoop contributor portal, join an active campaign,
and upload your clips following the recording guidelines in CONTRIBUTING.md.

**What camera should I use?**
Any action camera (GoPro, Insta360) or smartphone mounted on your head
or wrist works well. We recommend 1080p at 30 fps minimum.

## Pipeline

**What models does the pipeline use?**

| Stage          | Model                              |
|----------------|------------------------------------|
| Hand pose      | MediaPipe Hands                    |
| Arm pose       | YOLOv8n-pose                       |
| Segmentation   | SAM 2.1 Hiera-Tiny                 |
| Depth          | Depth-Anything-V2 Metric Indoor-S  |
| Transcription  | OpenAI Whisper base                |
| Quality check  | MediaPipe + OpenCV metrics         |

**How long does the pipeline take per clip?**
On a machine with an NVIDIA RTX 3060 or better, expect roughly:
- 30-second clip: ~2-3 minutes total
- 60-second clip: ~4-6 minutes total

Depth inference is the slowest stage; reducing `DEPTH_RESIZE_WIDTH`
in `config.py` speeds it up at the cost of accuracy.

## Quality gate

**Why was my clip rejected?**
Clips scoring below 60 / 100 are automatically quarantined in
`assets/videos/rejected/`. Check the `*_quality.json` file alongside
the rejected clip for a breakdown of which metrics brought the score down.

**Can I re-submit a rejected clip?**
Yes — improve the lighting or stabilisation, re-record, and re-submit
through the contributor portal.
