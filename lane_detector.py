"""
Lane detection module for OpenLKAS (Open Lane Keeping Assist System)
Processes frames to detect lane lines and calculate vehicle drift from lane center.
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional, Dict
import logging
from utils import (create_roi_mask, get_default_roi_vertices, 
                   calculate_center_offset, is_lane_departure,
                   draw_lane_lines, draw_drift_indicator, 
                   calculate_lane_center)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LaneDetector:
    """
    Lane detection system using computer vision techniques.
    """
    
    def __init__(self, 
                 canny_low: int = 50,
                 canny_high: int = 150,
                 gaussian_kernel: int = 5,
                 hough_threshold: int = 50,
                 min_line_length: int = 100,
                 max_line_gap: int = 50,
                 departure_threshold: float = 50.0,
                 roi_vertices: Optional[np.ndarray] = None):
        """
        Initialize lane detector with configurable parameters.
        
        Args:
            canny_low: Lower threshold for Canny edge detection
            canny_high: Upper threshold for Canny edge detection
            gaussian_kernel: Kernel size for Gaussian blur
            hough_threshold: Threshold for Hough line detection
            min_line_length: Minimum line length for Hough transform
            max_line_gap: Maximum gap between line segments
            departure_threshold: Threshold for lane departure detection (pixels)
            roi_vertices: Custom ROI vertices (None for default)
        """
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.gaussian_kernel = gaussian_kernel
        self.hough_threshold = hough_threshold
        self.min_line_length = min_line_length
        self.max_line_gap = max_line_gap
        self.departure_threshold = departure_threshold
        self.roi_vertices = roi_vertices
        
        # State tracking
        self.last_lane_center = None
        self.smoothing_factor = 0.7  # For temporal smoothing
        
        logger.info(f"Lane detector initialized with departure threshold: {departure_threshold}px")
    
    def detect_lanes(self, frame: np.ndarray) -> Dict:
        """
        Detect lane lines in the given frame.
        
        Args:
            frame: Input BGR image frame
            
        Returns:
            Dictionary containing detection results:
            - 'lines': Detected line segments
            - 'lane_center': Calculated lane center
            - 'image_center': Image center
            - 'offset': Offset from lane center
            - 'off_lane': Boolean indicating lane departure
            - 'processed_frame': Frame with visual overlays
        """
        if frame is None:
            logger.error("Input frame is None")
            return self._empty_result()
        
        try:
            # Step 1: Preprocess the frame
            processed = self._preprocess_frame(frame)
            
            # Step 2: Apply region of interest mask
            masked = self._apply_roi_mask(processed, frame.shape)
            
            # Step 3: Detect edges
            edges = self._detect_edges(masked)
            
            # Step 4: Detect lines using Hough transform
            lines = self._detect_lines(edges)
            
            # Step 5: Calculate lane center and drift
            result = self._calculate_drift(frame, lines)
            
            # Step 6: Add visual overlays
            result['processed_frame'] = self._add_visual_overlays(
                frame.copy(), lines, result['offset'], result['lane_center']
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error in lane detection: {e}")
            return self._empty_result()
    
    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess frame for lane detection.
        
        Args:
            frame: Input BGR frame
            
        Returns:
            Preprocessed grayscale frame
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (self.gaussian_kernel, self.gaussian_kernel), 0)
        
        return blurred
    
    def _apply_roi_mask(self, processed: np.ndarray, frame_shape: Tuple[int, ...]) -> np.ndarray:
        """
        Apply region of interest mask to focus on road area.
        
        Args:
            processed: Preprocessed frame
            frame_shape: Shape of original frame
            
        Returns:
            Masked frame
        """
        if self.roi_vertices is None:
            self.roi_vertices = get_default_roi_vertices(frame_shape[:2])
        
        mask = create_roi_mask(processed, self.roi_vertices)
        masked = cv2.bitwise_and(processed, mask)
        
        return masked
    
    def _detect_edges(self, masked: np.ndarray) -> np.ndarray:
        """
        Detect edges using Canny edge detection.
        
        Args:
            masked: Masked preprocessed frame
            
        Returns:
            Edge map
        """
        edges = cv2.Canny(masked, self.canny_low, self.canny_high)
        return edges
    
    def _detect_lines(self, edges: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect lines using Hough line transform.
        
        Args:
            edges: Edge map from Canny detection
            
        Returns:
            Array of detected line segments
        """
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=self.hough_threshold,
            minLineLength=self.min_line_length,
            maxLineGap=self.max_line_gap
        )
        
        return lines
    
    def _calculate_drift(self, frame: np.ndarray, lines: Optional[np.ndarray]) -> Dict:
        """
        Calculate lane center and drift from detected lines.
        
        Args:
            frame: Original frame
            lines: Detected line segments
            
        Returns:
            Dictionary with drift calculation results
        """
        height, width = frame.shape[:2]
        image_center = width / 2
        
        # Calculate lane center from detected lines
        lane_center = calculate_lane_center(lines, width) if lines is not None else None
        
        # Apply temporal smoothing if we have a previous lane center
        if lane_center is not None and self.last_lane_center is not None:
            lane_center = (self.smoothing_factor * self.last_lane_center + 
                          (1 - self.smoothing_factor) * lane_center)
        
        # Update last lane center
        if lane_center is not None:
            self.last_lane_center = lane_center
        else:
            # If no lines detected, use previous center or image center
            lane_center = self.last_lane_center if self.last_lane_center else image_center
        
        # Calculate offset
        offset = calculate_center_offset(lane_center, image_center)
        
        # Check for lane departure
        off_lane = is_lane_departure(offset, self.departure_threshold)
        
        return {
            'lines': lines,
            'lane_center': lane_center,
            'image_center': image_center,
            'offset': offset,
            'off_lane': off_lane
        }
    
    def _add_visual_overlays(self, frame: np.ndarray, lines: Optional[np.ndarray], 
                           offset: float, lane_center: float) -> np.ndarray:
        """
        Add visual overlays to the frame for debugging and visualization.
        
        Args:
            frame: Frame to add overlays to
            lines: Detected line segments
            offset: Calculated offset
            lane_center: Calculated lane center
            
        Returns:
            Frame with visual overlays
        """
        # Draw detected lane lines
        if lines is not None:
            frame = draw_lane_lines(frame, lines)
        
        # Draw drift indicators
        height, width = frame.shape[:2]
        image_center = width / 2
        frame = draw_drift_indicator(frame, offset, self.departure_threshold, image_center)
        
        # Add text information
        info_text = f"Offset: {offset:.1f}px | Threshold: {self.departure_threshold}px"
        cv2.putText(frame, info_text, (10, height - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add lane departure warning
        if abs(offset) > self.departure_threshold:
            warning_text = "LANE DEPARTURE WARNING!"
            cv2.putText(frame, warning_text, (width // 2 - 150, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        
        return frame
    
    def _empty_result(self) -> Dict:
        """Return empty result structure."""
        return {
            'lines': None,
            'lane_center': None,
            'image_center': None,
            'offset': 0.0,
            'off_lane': False,
            'processed_frame': None
        }
    
    def update_threshold(self, new_threshold: float):
        """
        Update the lane departure threshold.
        
        Args:
            new_threshold: New threshold value in pixels
        """
        self.departure_threshold = new_threshold
        logger.info(f"Updated departure threshold to: {new_threshold}px")
    
    def get_detection_stats(self) -> Dict:
        """
        Get current detection statistics.
        
        Returns:
            Dictionary with detection parameters and statistics
        """
        return {
            'canny_low': self.canny_low,
            'canny_high': self.canny_high,
            'gaussian_kernel': self.gaussian_kernel,
            'hough_threshold': self.hough_threshold,
            'min_line_length': self.min_line_length,
            'max_line_gap': self.max_line_gap,
            'departure_threshold': self.departure_threshold,
            'smoothing_factor': self.smoothing_factor,
            'last_lane_center': self.last_lane_center
        }


def create_lane_detector(departure_threshold: float = 50.0, 
                        canny_low: int = 50,
                        canny_high: int = 150,
                        hough_threshold: int = 50) -> LaneDetector:
    """
    Factory function to create lane detector with common configurations.
    
    Args:
        departure_threshold: Threshold for lane departure detection
        canny_low: Lower Canny threshold
        canny_high: Upper Canny threshold
        hough_threshold: Hough line detection threshold
        
    Returns:
        Configured LaneDetector instance
    """
    return LaneDetector(
        departure_threshold=departure_threshold,
        canny_low=canny_low,
        canny_high=canny_high,
        hough_threshold=hough_threshold
    ) 