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
                 roi_vertices: Optional[np.ndarray] = None,
                 car_width: float = 70.0,
                 lane_width: float = 144.0,
                 camera_offset: float = 0.0):
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
        
        self.car_width = car_width
        self.lane_width = lane_width
        self.camera_offset = camera_offset
        self.last_lane_width_pixels = 400.0  # Safe initial pixel estimate
        
        # Cache for ROI mask to optimize processing
        self.cached_roi_mask = None
        self.cached_frame_shape = None

        # State tracking
        self.last_lane_center = None
        self.smoothing_factor = 0.3  # For temporal smoothing (lower = more responsive to current frame)
        
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
            
            # Step 2: Detect edges
            edges = self._detect_edges(processed)
            
            # Step 3: Apply region of interest mask
            masked_edges = self._apply_roi_mask(edges, frame.shape)
            
            # Step 4: Detect lines using Hough transform
            lines = self._detect_lines(masked_edges)
            
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
        # Create and cache the mask if not already cached, or if frame shape changed
        if self.cached_roi_mask is None or self.cached_frame_shape != frame_shape[:2]:
            if self.roi_vertices is None:
                self.roi_vertices = get_default_roi_vertices(frame_shape[:2])

            # extract the actual 2D array if we are dealing with the nested array format
            vertices_to_use = self.roi_vertices[0] if len(self.roi_vertices.shape) == 3 and self.roi_vertices.shape[0] == 1 else self.roi_vertices

            # Create a blank mask matching the shape of the processed frame (e.g. grayscale/edges)
            # which is just 2D, rather than the 3D frame shape.
            mask_shape = processed.shape[:2]
            mask = np.zeros(mask_shape, dtype=np.uint8)
            cv2.fillPoly(mask, [vertices_to_use], 255)

            self.cached_roi_mask = mask
            self.cached_frame_shape = frame_shape[:2]

        masked = cv2.bitwise_and(processed, self.cached_roi_mask)
        
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
        lane_center, left_intercept, right_intercept = calculate_lane_center(lines, width, height) if lines is not None else (None, None, None)
        
        # Update dynamic lane width tracking
        if left_intercept is not None and right_intercept is not None:
             current_lane_pixel_width = right_intercept - left_intercept
             self.last_lane_width_pixels = (0.9 * self.last_lane_width_pixels) + (0.1 * current_lane_pixel_width)
             
        # Calculate dynamic physical threshold
        pixels_per_inch = max(1.0, self.last_lane_width_pixels / self.lane_width)
        camera_offset_pixels = self.camera_offset * pixels_per_inch
        true_car_center_in_image = image_center + camera_offset_pixels
        
        wiggle_room_inches = (self.lane_width - self.car_width) / 2.0
        self.departure_threshold = max(5.0, wiggle_room_inches * pixels_per_inch)

        
        # Apply temporal smoothing if we have a previous lane center
        if lane_center is not None and self.last_lane_center is not None:
            # New value gets 1-smoothing_factor weight, old value gets smoothing_factor weight
            lane_center = (self.smoothing_factor * self.last_lane_center + 
                          (1 - self.smoothing_factor) * lane_center)
        
        # Update last lane center
        if lane_center is not None:
            self.last_lane_center = lane_center
        else:
            # If no lines detected, use previous center or image center
            lane_center = self.last_lane_center if self.last_lane_center else image_center
        
        # Calculate offset using camera-aligned structural center
        offset = calculate_center_offset(lane_center, true_car_center_in_image)
        
        # Check for lane departure
        off_lane = is_lane_departure(offset, self.departure_threshold)
        
        return {
            'lines': lines,
            'lane_center': lane_center,
            'image_center': image_center,
            'offset': offset,
            'off_lane': off_lane,
            'left_intercept': left_intercept,
            'right_intercept': right_intercept
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

        # Visualize ROI
        if self.roi_vertices is not None:
            pts = self.roi_vertices.reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], isClosed=True, color=(255, 0, 255), thickness=2)
            cv2.putText(frame, "ROI Focus Area", (int(pts[0][0][0]), int(pts[0][0][1]) - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
        
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
        
    def _normalize_vertices(self):
        """Ensure vertices are stored as a (4, 2) array rather than (1, 4, 2)"""
        if self.roi_vertices is not None and len(self.roi_vertices.shape) == 3 and self.roi_vertices.shape[0] == 1:
            self.roi_vertices = self.roi_vertices[0].copy()

    def offset_roi(self, dx: int, dy: int):
        """
        Shift the region of interest by dx and dy pixels.
        """
        self._normalize_vertices()
        if self.roi_vertices is not None:
            self.roi_vertices = self.roi_vertices + np.array([dx, dy], dtype=np.int32)
            self.cached_roi_mask = None
            logger.info(f"Shifted ROI by ({dx}, {dy})")

    def adjust_top_edge(self, dy: int):
        """Move only the Top Edge (the horizon) up and down."""
        self._normalize_vertices()
        if self.roi_vertices is not None:
            self.roi_vertices[1][1] += dy
            self.roi_vertices[2][1] += dy
            self.cached_roi_mask = None
            logger.info(f"Adjusted Top Edge Y by {dy}")

    def adjust_bottom_edge(self, dy: int):
        """Move only the Bottom Edge up and down."""
        self._normalize_vertices()
        if self.roi_vertices is not None:
            self.roi_vertices[0][1] += dy
            self.roi_vertices[3][1] += dy
            self.cached_roi_mask = None
            logger.info(f"Adjusted Bottom Edge Y by {dy}")

    def adjust_top_width(self, dx: int):
        """Expand or shrink the Top Width. dx > 0 expands it."""
        self._normalize_vertices()
        if self.roi_vertices is not None:
            self.roi_vertices[1][0] -= dx
            self.roi_vertices[2][0] += dx
            self.cached_roi_mask = None
            logger.info(f"Adjusted Top Width by {dx}")

    def adjust_bottom_width(self, dx: int):
        """Expand or shrink the Bottom Width. dx > 0 expands it."""
        self._normalize_vertices()
        if self.roi_vertices is not None:
            self.roi_vertices[0][0] -= dx
            self.roi_vertices[3][0] += dx
            self.cached_roi_mask = None
            logger.info(f"Adjusted Bottom Width by {dx}")

    def scale_roi(self, width_factor: float, height_factor: float):
        """
        Scale the region of interest's width and height relative to its centroid.
        """
        self._normalize_vertices()
        if self.roi_vertices is not None:
            pts = self.roi_vertices
            centroid_x = np.mean(pts[:, 0])
            centroid_y = np.mean(pts[:, 1])
            
            for i in range(len(pts)):
                pts[i][0] = int(centroid_x + (pts[i][0] - centroid_x) * width_factor)
                pts[i][1] = int(centroid_y + (pts[i][1] - centroid_y) * height_factor)
                
            self.roi_vertices = pts
            self.cached_roi_mask = None
            logger.info(f"Scaled ROI by W:{width_factor} H:{height_factor}")

    def auto_calibrate_roi(self, frame: np.ndarray):
        """
        Perform a full open-scan to auto-calibrate the lane and frame the horizon.
        """
        h, w = frame.shape[:2]
        temp_vertices = np.array([[[w*0.05, h], [w*0.2, h*0.4], [w*0.8, h*0.4], [w*0.95, h]]], dtype=np.int32)
        
        processed = self._preprocess_frame(frame)
        edges = self._detect_edges(processed)
        
        mask = np.zeros(edges.shape, dtype=np.uint8)
        cv2.fillPoly(mask, [temp_vertices[0]], 255)
        masked_edges = cv2.bitwise_and(edges, mask)
        
        lines = self._detect_lines(masked_edges)
        if lines is not None:
            # We detected a horizon, auto generate bounds avoiding dash
            self.roi_vertices = np.array([[
                [w * 0.1, h * 0.85],
                [w * 0.4, h * 0.6],
                [w * 0.6, h * 0.6],
                [w * 0.9, h * 0.85]
            ]], dtype=np.int32)[0]
            self.cached_roi_mask = None
            logger.info("Auto calibrated ROI based on lane perspective.")

    
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
                        hough_threshold: int = 50,
                        car_width: float = 70.0,
                        lane_width: float = 144.0,
                        camera_offset: float = 0.0) -> LaneDetector:
    """
    Factory function to create lane detector with common configurations.
    
    Args:
        departure_threshold: Baseline departure
        canny_low: Lower Canny threshold
        canny_high: Upper Canny threshold
        hough_threshold: Hough line detection threshold
        car_width: Physical width in inches
        lane_width: Physical width in inches
        camera_offset: Camera center offset
        
    Returns:
        Configured LaneDetector instance
    """
    return LaneDetector(
        departure_threshold=departure_threshold,
        canny_low=canny_low,
        canny_high=canny_high,
        hough_threshold=hough_threshold,
        car_width=car_width,
        lane_width=lane_width,
        camera_offset=camera_offset
    ) 