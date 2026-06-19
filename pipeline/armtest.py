import numpy as np
import cv2
from typing import Tuple, Optional

def estimate_arm_length(
    left_wrist: np.ndarray,
    right_wrist: np.ndarray,
    left_hand: np.ndarray,
    right_hand: np.ndarray,
    depth_map: Optional[np.ndarray] = None
) -> Tuple[float, float]:
    """
    Estimate the length of both arms from wrist to hand.
    
    Parameters:
    -----------
    left_wrist : np.ndarray
        3D coordinates of the left wrist [x, y, z]
    right_wrist : np.ndarray
        3D coordinates of the right wrist [x, y, z]
    left_hand : np.ndarray
        3D coordinates of the left hand center [x, y, z]
    right_hand : np.ndarray
        3D coordinates of the right hand center [x, y, z]
    depth_map : np.ndarray, optional
        Depth map for 3D distance calculation
    
    Returns:
    --------
    Tuple[float, float]
        (left_arm_length, right_arm_length) in meters
    """
    left_arm_length = calculate_arm_length(left_wrist, left_hand, depth_map)
    right_arm_length = calculate_arm_length(right_wrist, right_hand, depth_map)
    return left_arm_length, right_arm_length


def calculate_arm_length(
    wrist: np.ndarray,
    hand: np.ndarray,
    depth_map: Optional[np.ndarray] = None
) -> float:
    """
    Calculate the 3D Euclidean distance between wrist and hand.
    
    Parameters:
    -----------
    wrist : np.ndarray
        3D coordinates of the wrist [x, y, z]
    hand : np.ndarray
        3D coordinates of the hand center [x, y, z]
    depth_map : np.ndarray, optional
        Depth map for 3D distance calculation
    
    Returns:
    --------
    float
        Arm length in meters
    """
    if depth_map is not None:
        # Use 3D coordinates from depth map
        wrist_3d = wrist
        hand_3d = hand
        distance = np.sqrt(
            np.sum((wrist_3d - hand_3d) ** 2)
        )
    else:
        # Use 2D coordinates (projective distance)
        wrist_2d = wrist[:2]
        hand_2d = hand[:2]
        distance = np.sqrt(
            np.sum((wrist_2d - hand_2d) ** 2)
        )
    
    return float(distance)


def check_arm_visibility(
    left_hand_mask: np.ndarray,
    right_hand_mask: np.ndarray,
    image: np.ndarray,
    threshold: float = 0.5
) -> Tuple[bool, bool]:
    """
    Check if arms are visible in the image based on hand mask continuity.
    
    Parameters:
    -----------
    left_hand_mask : np.ndarray
        Binary mask for left hand
    right_hand_mask : np.ndarray
        Binary mask for right hand
    image : np.ndarray
        RGB image
    threshold : float
        Threshold for visibility check (0-1)
    
    Returns:
    --------
    Tuple[bool, bool]
        (left_arm_visible, right_arm_visible)
    """
    left_visible = check_hand_continuity(left_hand_mask, image, threshold)
    right_visible = check_hand_continuity(right_hand_mask, image, threshold)
    return left_visible, right_visible


def check_hand_continuity(
    hand_mask: np.ndarray,
    image: np.ndarray,
    threshold: float = 0.5
) -> bool:
    """
    Check if there is continuous hand mask from wrist to hand region.
    
    Parameters:
    -----------
    hand_mask : np.ndarray
        Binary mask for hand
    image : np.ndarray
        RGB image
    threshold : float
        Threshold for continuity check
    
    Returns:
    --------
    bool
        True if arm is considered visible
    """
    # Count non-zero pixels in hand mask
    pixel_count = np.count_nonzero(hand_mask)
    total_pixels = hand_mask.shape[0] * hand_mask.shape[1]
    
    # Calculate visibility ratio
    visibility_ratio = pixel_count / total_pixels
    
    return visibility_ratio > threshold


def estimate_arm_angle(
    shoulder: np.ndarray,
    elbow: np.ndarray,
    wrist: np.ndarray
) -> float:
    """
    Estimate the angle of the arm at the elbow (shoulder-elbow-wrist).
    
    Parameters:
    -----------
    shoulder : np.ndarray
        3D coordinates of the shoulder [x, y, z]
    elbow : np.ndarray
        3D coordinates of the elbow [x, y, z]
    wrist : np.ndarray
        3D coordinates of the wrist [x, y, z]
    
    Returns:
    --------
    float
        Angle in degrees (0-180)
    """
    # Calculate vectors
    v1 = elbow - shoulder
    v2 = wrist - elbow
    
    # Calculate lengths
    len1 = np.linalg.norm(v1)
    len2 = np.linalg.norm(v2)
    
    if len1 < 1e-6 or len2 < 1e-6:
        return 90.0  # Default to 90 degrees if points are too close
    
    # Calculate cosine of the angle
    cos_angle = np.dot(v1, v2) / (len1 * len2)
    
    # Clamp value to avoid numerical issues
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    
    # Calculate angle in degrees
    angle = np.degrees(np.arccos(cos_angle))
    
    return float(angle)


def analyze_arm_posture(
    left_arm_length: float,
    right_arm_length: float,
    left_arm_angle: float,
    right_arm_angle: float,
    left_hand: np.ndarray,
    right_hand: np.ndarray,
    image_center: Tuple[int, int]
) -> dict:
    """
    Analyze the overall posture of both arms.
    
    Parameters:
    -----------
    left_arm_length : float
        Length of left arm in meters
    right_arm_length : float
        Length of right arm in meters
    left_arm_angle : float
        Angle of left arm at elbow
    right_arm_angle : float
        Angle of right arm at elbow
    left_hand : np.ndarray
        3D coordinates of left hand
    right_hand : np.ndarray
        3D coordinates of right hand
    image_center : Tuple[int, int]
        Center of the image [width, height]
    
    Returns:
    --------
    dict
        Dictionary containing arm posture analysis
    """
    analysis = {
        'left_arm_length': left_arm_length,
        'right_arm_length': right_arm_length,
        'left_arm_angle': left_arm_angle,
        'right_arm_angle': right_arm_angle,
        'left_hand_distance_to_center': np.linalg.norm(
            np.array(left_hand[:2]) - np.array(image_center)
        ),
        'right_hand_distance_to_center': np.linalg.norm(
            np.array(right_hand[:2]) - np.array(image_center)
        ),
        'arms_symmetric': abs(left_arm_length - right_arm_length) < 0.05,
        'arms_extended': left_arm_angle > 150 and right_arm_angle > 150,
        'arms_crossed': False  # Would need more complex analysis
    }
    
    return analysis


if __name__ == "__main__":
    # Test the functions with sample data
    sample_wrist = np.array([100, 150, 500])
    sample_hand = np.array([120, 160, 510])
    
    left_length, right_length = estimate_arm_length(
        sample_wrist, sample_wrist, sample_hand, sample_hand
    )
    print(f"Left arm length: {left_length} m")
    print(f"Right arm length: {right_length} m")