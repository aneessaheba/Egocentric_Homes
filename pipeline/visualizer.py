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

_ARM_CONNECTIONS = [(5,7),(7,9),(6,8),(8,10)]

def draw_arm_skeleton(frame, keypoints: list, colour=_BLUE, thickness: int = 2):
    """Draw arm skeleton from YOLO pose keypoints."""
    import cv2
    h, w = frame.shape[:2]
    pts = {}
    for kp in keypoints:
        idx = kp.get("id")
        if idx is not None and kp.get("confidence", 0) > 0.3:
            pts[idx] = (int(kp["x"] * w), int(kp["y"] * h))
    for a, b in _ARM_CONNECTIONS:
        if a in pts and b in pts:
            cv2.line(frame, pts[a], pts[b], colour, thickness, cv2.LINE_AA)
    for p in pts.values():
        cv2.circle(frame, p, 5, colour, -1)

def draw_depth_heatmap(depth_array, alpha: float = 0.5):
    """Return a BGR heatmap frame from a depth array."""
    import cv2, numpy as np
    d = depth_array.astype(np.float32)
    d_norm = cv2.normalize(d, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.applyColorMap(d_norm, cv2.COLORMAP_INFERNO)

def draw_quality_hud(frame, score: float, frame_id: Optional[int] = None):
    """Overlay a quality score badge in the top-right corner."""
    import cv2
    h, w = frame.shape[:2]
    colour = _GREEN if score >= 70 else (_YELLOW if score >= 50 else _RED)
    label  = f"Q:{score:.0f}" + (f"  f{frame_id}" if frame_id is not None else "")
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    x1, y1 = w - tw - 14, 8
    cv2.rectangle(frame, (x1-4, y1), (x1+tw+4, y1+th+6), _BLACK, -1)
    cv2.putText(frame, label, (x1, y1+th+2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)
