"""visualizer.py — Draw annotation overlays on video frames.

Provides OpenCV drawing helpers for each pipeline stage.
"""
from pathlib import Path
from typing import Optional

# BGR colours for OpenCV
_RED    = (0,   0, 255)
_GREEN  = (0, 200,  50)
_BLUE   = (255, 60,   0)
_YELLOW = (0, 210, 255)
_WHITE  = (255, 255, 255)
_BLACK  = (0,   0,   0)

# MediaPipe hand connection pairs (0-based landmark indices)
_HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(0,17),(17,18),(18,19),(19,20),
]

def draw_hand_landmarks(frame, landmarks: list, colour=_GREEN, radius: int = 4):
    """Draw hand keypoints and skeleton on *frame* in-place."""
    import cv2
    h, w = frame.shape[:2]
    pts = [(int(lm["x"] * w), int(lm["y"] * h)) for lm in landmarks]
    for a, b in _HAND_CONNECTIONS:
        if a < len(pts) and b < len(pts):
            cv2.line(frame, pts[a], pts[b], colour, 1, cv2.LINE_AA)
    for p in pts:
        cv2.circle(frame, p, radius, colour, -1, cv2.LINE_AA)
