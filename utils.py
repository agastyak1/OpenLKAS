"""
Utility functions for OpenLKAS (Open Lane Keeping Assist System)
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
    bottom_left = (width * 0.1, height)
    top_left = (width * 0.4, height * 0.6)
    top_right = (width * 0.6, height * 0.6)
    bottom_right = (width * 0.9, height)
    
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


def calculate_lane_center(lines: List[np.ndarray], image_width: int) -> Optional[float]:
    """
    Calculate the center of the detected lane from line segments.
    
    Args:
        lines: List of detected line segments
        image_width: Width of the image
        
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
    
    # Calculate average positions for left and right lanes
    left_center = None
    right_center = None
    
    if left_lines:
        left_x_coords = []
        for line in left_lines:
            x1, y1, x2, y2 = line[0]
            left_x_coords.extend([x1, x2])
        left_center = np.mean(left_x_coords)
    
    if right_lines:
        right_x_coords = []
        for line in right_lines:
            x1, y1, x2, y2 = line[0]
            right_x_coords.extend([x1, x2])
        right_center = np.mean(right_x_coords)
    
    # Calculate lane center
    if left_center is not None and right_center is not None:
        return (left_center + right_center) / 2
    elif left_center is not None:
        return left_center + 100  # Estimate right lane
    elif right_center is not None:
        return right_center - 100  # Estimate left lane
    
    return None 