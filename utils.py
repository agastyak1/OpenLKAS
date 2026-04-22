"""
Utility functions for OpenLCWS (Open Lane and Collision Warning System)
Helper functions for image processing, coordinate calculations, and common utilities.
"""

import cv2
import numpy as np
import math
from typing import Tuple, List, Optional


def create_roi_mask(image: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    """
    Create a region of interest mask for the image.
    
    Args:
        image: Input image
        vertices: Array of vertices defining the ROI polygon
        
    Returns:
        Mask with ROI area filled
    """
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, [vertices], 255)
    return mask


def get_default_roi_vertices(image_shape: Tuple[int, int]) -> np.ndarray:
    """
    Get default region of interest vertices for lane detection.
    Focuses on the lower half of the image where the road is typically visible.
    
    Args:
        image_shape: Tuple of (height, width) of the image
        
    Returns:
        Array of vertices defining the ROI polygon
    """
    height, width = image_shape
    
    # Define ROI vertices - trapezoid shape focusing on road area
    bottom_left = (width * 0.1, height * 0.85)
    top_left = (width * 0.4, height * 0.6)
    top_right = (width * 0.6, height * 0.6)
    bottom_right = (width * 0.9, height * 0.85)
    
    vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    return vertices


def calculate_center_offset(lane_center: float, image_center: float) -> float:
    """
    Calculate the offset between lane center and image center.
    
    Args:
        lane_center: X-coordinate of the detected lane center
        image_center: X-coordinate of the image center
        
    Returns:
        Offset in pixels (positive = right drift, negative = left drift)
    """
    return lane_center - image_center


def is_lane_departure(offset: float, threshold: float) -> bool:
    """
    Determine if the vehicle is departing from the lane based on offset.
    
    Args:
        offset: Center offset in pixels
        threshold: Departure threshold in pixels
        
    Returns:
        True if lane departure detected, False otherwise
    """
    return abs(offset) > threshold


def draw_lane_lines(image: np.ndarray, lines: List[np.ndarray], 
                   color: Tuple[int, int, int] = (0, 255, 0), 
                   thickness: int = 3) -> np.ndarray:
    """
    Draw detected lane lines on the image.
    
    Args:
        image: Input image to draw on
        lines: List of detected line segments
        color: BGR color tuple for lines
        thickness: Line thickness
        
    Returns:
        Image with lane lines drawn
    """
    if lines is None:
        return image
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(image, (x1, y1), (x2, y2), color, thickness)
    
    return image


def draw_drift_indicator(image: np.ndarray, offset: float, 
                        threshold: float, image_center: float) -> np.ndarray:
    """
    Draw visual indicators for lane drift on the image.
    
    Args:
        image: Input image to draw on
        offset: Center offset in pixels
        threshold: Departure threshold in pixels
        image_center: X-coordinate of image center
        
    Returns:
        Image with drift indicators drawn
    """
    height, width = image.shape[:2]
    
    # Draw center line
    cv2.line(image, (int(image_center), height), (int(image_center), height - 100), 
             (255, 255, 255), 2)
    
    # Draw lane center indicator
    lane_center = image_center + offset
    cv2.line(image, (int(lane_center), height), (int(lane_center), height - 100), 
             (0, 255, 255), 2)
    
    # Draw threshold boundaries
    left_threshold = image_center - threshold
    right_threshold = image_center + threshold
    cv2.line(image, (int(left_threshold), height), (int(left_threshold), height - 50), 
             (0, 0, 255), 1)
    cv2.line(image, (int(right_threshold), height), (int(right_threshold), height - 50), 
             (0, 0, 255), 1)
    
    # Draw drift direction arrow
    if abs(offset) > threshold:
        arrow_color = (0, 0, 255) if offset > 0 else (255, 0, 0)  # Red for right, Blue for left
        arrow_text = "RIGHT DRIFT" if offset > 0 else "LEFT DRIFT"
        cv2.putText(image, arrow_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, arrow_color, 2)
    
    return image


def add_text_overlay(image: np.ndarray, text: str, 
                    position: Tuple[int, int] = (10, 60),
                    color: Tuple[int, int, int] = (255, 255, 255),
                    scale: float = 0.7) -> np.ndarray:
    """
    Add text overlay to the image.
    
    Args:
        image: Input image
        text: Text to display
        position: (x, y) position for text
        color: BGR color tuple
        scale: Text scale
        
    Returns:
        Image with text overlay
    """
    cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)
    return image


def resize_image(image: np.ndarray, width: int = None, height: int = None) -> np.ndarray:
    """
    Resize image while maintaining aspect ratio.
    
    Args:
        image: Input image
        width: Target width (None to maintain aspect ratio)
        height: Target height (None to maintain aspect ratio)
        
    Returns:
        Resized image
    """
    if width is None and height is None:
        return image
    
    h, w = image.shape[:2]
    
    if width is None:
        aspect_ratio = w / h
        width = int(height * aspect_ratio)
    elif height is None:
        aspect_ratio = h / w
        height = int(width * aspect_ratio)
    
    return cv2.resize(image, (width, height))


def calculate_lane_center(lines: List[np.ndarray], image_width: int, image_height: int = 720) -> Optional[float]:
    """
    Calculate the center of the detected lane from line segments.
    
    Args:
        lines: List of detected line segments
        image_width: Width of the image
        image_height: Height of the image to find bottom intercept
        
    Returns:
        X-coordinate of lane center, or None if no lines detected
    """
    if lines is None or len(lines) == 0:
        return None
    
    # Separate left and right lines
    left_lines = []
    right_lines = []
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        slope = (y2 - y1) / (x2 - x1) if x2 != x1 else float('inf')
        
        # Filter out horizontal lines and very steep lines
        if abs(slope) < 0.5 or abs(slope) > 2.0:
            continue
            
        if slope < 0:  # Left lane (negative slope)
            left_lines.append(line)
        else:  # Right lane (positive slope)
            right_lines.append(line)
    
    # Calculate x intercepts at the bottom of the image for the left and right lanes
    left_x_intercepts = []
    right_x_intercepts = []
    
    bottom_y = image_height
    
    for line in left_lines:
        x1, y1, x2, y2 = line[0]
        slope = (y2 - y1) / (x2 - x1) if x2 != x1 else float('inf')
        b = y1 - slope * x1
        x_intercept = (bottom_y - b) / slope if slope != 0 and slope != float('inf') else x1
        left_x_intercepts.append(x_intercept)

    for line in right_lines:
        x1, y1, x2, y2 = line[0]
        slope = (y2 - y1) / (x2 - x1) if x2 != x1 else float('inf')
        b = y1 - slope * x1
        x_intercept = (bottom_y - b) / slope if slope != 0 and slope != float('inf') else x1
        right_x_intercepts.append(x_intercept)

    left_intercept = np.median(left_x_intercepts) if left_x_intercepts else None
    right_intercept = np.median(right_x_intercepts) if right_x_intercepts else None
    
    # Calculate lane center at bottom of image
    if left_intercept is not None and right_intercept is not None:
        return (left_intercept + right_intercept) / 2, left_intercept, right_intercept
    elif left_intercept is not None:
        return left_intercept + 200, left_intercept, None  # Estimate lane center from left lane
    elif right_intercept is not None:
        return right_intercept - 200, None, right_intercept  # Estimate lane center from right lane

    return None, None, None 


def draw_detection_boxes(image: np.ndarray, tracked_objects, 
                         ttc_caution: float = 3.0, ttc_warning: float = 2.0,
                         ttc_danger: float = 1.0) -> np.ndarray:
    """
    Draw bounding boxes around detected vehicles with TTC-based color coding.

    Args:
        image: Input image to draw on
        tracked_objects: List of TrackedObject instances from collision_detector
        ttc_caution: TTC threshold for caution tier (seconds)
        ttc_warning: TTC threshold for warning tier (seconds)
        ttc_danger: TTC threshold for danger tier (seconds)

    Returns:
        Image with detection boxes drawn
    """
    if not tracked_objects:
        return image

    for obj in tracked_objects:
        x1, y1, x2, y2 = obj.bbox
        ttc = obj.ttc

        # Color code: green=safe, yellow=caution, orange=warning, red=danger
        if ttc <= ttc_danger:
            color = (0, 0, 255)       # Red
            tier = "DANGER"
        elif ttc <= ttc_warning:
            color = (0, 128, 255)     # Orange
            tier = "WARNING"
        elif ttc <= ttc_caution:
            color = (0, 255, 255)     # Yellow
            tier = "CAUTION"
        else:
            color = (0, 255, 0)       # Green
            tier = ""

        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # Label with class name + TTC
        if ttc != float('inf') and ttc > 0:
            label = f"{obj.label} {ttc:.1f}s"
        else:
            label = f"{obj.label}"

        # Draw label background
        font_scale = 0.5
        thickness = 1
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 
                                                font_scale, thickness)
        cv2.rectangle(image, (x1, y1 - text_h - 8), (x1 + text_w + 4, y1), color, -1)
        cv2.putText(image, label, (x1 + 2, y1 - 4), 
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)

        # Draw tier badge if threatening
        if tier:
            cv2.putText(image, tier, (x1, y2 + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return image


def draw_collision_warning(image: np.ndarray, ttc: float, 
                           ttc_danger: float = 1.0) -> np.ndarray:
    """
    Draw a large collision warning banner when TTC is critical.

    Args:
        image: Input image to draw on
        ttc: Current time-to-collision (seconds)
        ttc_danger: TTC threshold for danger banner

    Returns:
        Image with collision warning overlay
    """
    if ttc is None or ttc == float('inf') or ttc > ttc_danger or ttc <= 0:
        return image

    height, width = image.shape[:2]

    # Semi-transparent red banner across the top
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (width, 80), (0, 0, 200), -1)

    # Pulsing opacity based on time for visual urgency
    import time
    pulse = 0.4 + 0.3 * abs(np.sin(time.time() * 6))  # Oscillates 0.4-0.7
    cv2.addWeighted(overlay, pulse, image, 1 - pulse, 0, image)

    # Warning text
    warning = f"!! COLLISION WARNING  TTC: {ttc:.1f}s !!"
    text_size = cv2.getTextSize(warning, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)[0]
    text_x = (width - text_size[0]) // 2
    cv2.putText(image, warning, (text_x, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)

    return image