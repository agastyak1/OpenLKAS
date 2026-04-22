"""
Collision detection module for OpenLCWS (Open Lane and Collision Warning System)
Detects vehicles ahead using MobileNet-SSD and estimates Time-to-Collision (TTC).
Runs inference on a background thread to avoid blocking the main lane detection loop.
"""

import cv2
import numpy as np
import threading
import time
import os
from typing import List, Optional, Dict, Tuple, NamedTuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# VOC class labels for MobileNet-SSD
VOC_CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair",
    "cow", "diningtable", "dog", "horse", "motorbike",
    "person", "pottedplant", "sheep", "sofa", "train",
    "tvmonitor"
]

# Vehicle class IDs (VOC indices): car=7, bus=6, motorbike=14
VEHICLE_CLASS_IDS = {6, 7, 14}


class Detection(NamedTuple):
    """A single object detection result."""
    class_id: int
    label: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)


class TrackedObject:
    """An object tracked across frames for TTC calculation."""

    def __init__(self, detection: Detection, timestamp: float):
        self.detection = detection
        self.bbox = detection.bbox
        self.label = detection.label
        self.confidence = detection.confidence
        self.height = detection.bbox[3] - detection.bbox[1]
        self.timestamp = timestamp
        self.ttc = float('inf')
        self.h_dot_smoothed = 0.0  # Smoothed height derivative


class CollisionDetector:
    """
    Vehicle detection and TTC estimation using MobileNet-SSD.
    """

    def __init__(self,
                 model_dir: str = "models",
                 confidence_threshold: float = 0.5,
                 input_size: int = 300,
                 target_classes: set = None,
                 ema_alpha: float = 0.3):
        """
        Initialize collision detector.

        Args:
            model_dir: Directory containing model files
            confidence_threshold: Minimum confidence to accept a detection
            input_size: DNN input resolution (300 for MobileNet-SSD)
            target_classes: Set of VOC class IDs to detect (default: vehicles)
            ema_alpha: EMA smoothing factor for height derivative (lower = smoother)
        """
        self.confidence_threshold = confidence_threshold
        self.input_size = input_size
        self.target_classes = target_classes or VEHICLE_CLASS_IDS
        self.ema_alpha = ema_alpha
        self.net = None
        self.is_initialized = False

        # Tracking state
        self.prev_detections: List[Detection] = []
        self.tracked_objects: List[TrackedObject] = []
        self.prev_timestamp: float = 0.0

        self._load_model(model_dir)

    def _load_model(self, model_dir: str):
        """Load the MobileNet-SSD Caffe model."""
        prototxt = os.path.join(model_dir, "MobileNetSSD_deploy.prototxt")
        caffemodel = os.path.join(model_dir, "MobileNetSSD_deploy.caffemodel")

        if not os.path.exists(prototxt):
            logger.error(f"Model prototxt not found: {prototxt}")
            logger.error("Run 'python download_models.py' to download model files.")
            return

        if not os.path.exists(caffemodel):
            logger.error(f"Model weights not found: {caffemodel}")
            logger.error("Run 'python download_models.py' to download model files.")
            return

        try:
            self.net = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)
            # Use OpenCV's CPU backend — leverages NEON SIMD on ARM (RPi)
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            self.is_initialized = True
            logger.info("Collision detector model loaded successfully (MobileNet-SSD)")
        except Exception as e:
            logger.error(f"Failed to load collision detection model: {e}")

    def detect_objects(self, frame: np.ndarray) -> List[Detection]:
        """
        Run MobileNet-SSD inference on a single frame.

        Args:
            frame: Input BGR image

        Returns:
            List of Detection namedtuples for vehicles only
        """
        if not self.is_initialized or self.net is None:
            return []

        h, w = frame.shape[:2]

        # Create blob — 300x300 with mean subtraction
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (self.input_size, self.input_size)),
            scalefactor=0.007843,
            size=(self.input_size, self.input_size),
            mean=127.5
        )

        self.net.setInput(blob)
        raw_detections = self.net.forward()

        results = []
        for i in range(raw_detections.shape[2]):
            confidence = raw_detections[0, 0, i, 2]

            if confidence < self.confidence_threshold:
                continue

            class_id = int(raw_detections[0, 0, i, 1])

            if class_id not in self.target_classes:
                continue

            # Scale bounding box back to frame coordinates
            box = raw_detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)

            # Clamp to frame bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            label = VOC_CLASSES[class_id] if class_id < len(VOC_CLASSES) else f"class_{class_id}"

            results.append(Detection(
                class_id=class_id,
                label=label,
                confidence=float(confidence),
                bbox=(x1, y1, x2, y2)
            ))

        return results

    def _iou(self, box_a: Tuple, box_b: Tuple) -> float:
        """Calculate Intersection over Union between two bounding boxes."""
        x1 = max(box_a[0], box_b[0])
        y1 = max(box_a[1], box_b[1])
        x2 = min(box_a[2], box_b[2])
        y2 = min(box_a[3], box_b[3])

        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
        union = area_a + area_b - inter

        return inter / union if union > 0 else 0.0

    def calculate_ttc(self, current_detections: List[Detection],
                      current_time: float) -> List[TrackedObject]:
        """
        Calculate Time-to-Collision for each detection by tracking
        bounding box height expansion across frames.

        Args:
            current_detections: Detections from the current frame
            current_time: Timestamp of the current frame

        Returns:
            List of TrackedObject with TTC estimates
        """
        dt = current_time - self.prev_timestamp if self.prev_timestamp > 0 else 0.0
        tracked = []

        for det in current_detections:
            obj = TrackedObject(det, current_time)

            if dt > 0 and self.tracked_objects:
                # Match to previous frame using IoU
                best_iou = 0.0
                best_prev = None

                for prev_obj in self.tracked_objects:
                    iou = self._iou(det.bbox, prev_obj.bbox)
                    if iou > best_iou:
                        best_iou = iou
                        best_prev = prev_obj

                if best_prev is not None and best_iou > 0.2:
                    # Calculate height derivative
                    h_curr = det.bbox[3] - det.bbox[1]
                    h_prev = best_prev.height
                    h_dot_raw = (h_curr - h_prev) / dt

                    # EMA smoothing
                    obj.h_dot_smoothed = (
                        self.ema_alpha * h_dot_raw +
                        (1 - self.ema_alpha) * best_prev.h_dot_smoothed
                    )

                    # TTC = h / h_dot (only when object is approaching)
                    if obj.h_dot_smoothed > 1.0:  # Minimum 1px/s to avoid division noise
                        obj.ttc = h_curr / obj.h_dot_smoothed
                    else:
                        obj.ttc = float('inf')

            tracked.append(obj)

        # Update state for next frame
        self.tracked_objects = tracked
        self.prev_detections = current_detections
        self.prev_timestamp = current_time

        return tracked

    def get_closest_threat(self, tracked: List[TrackedObject]) -> Optional[TrackedObject]:
        """
        Return the tracked object with the lowest positive TTC.

        Args:
            tracked: List of tracked objects from calculate_ttc

        Returns:
            The most imminent threat, or None
        """
        threats = [obj for obj in tracked if obj.ttc > 0 and obj.ttc != float('inf')]
        if not threats:
            return None
        return min(threats, key=lambda o: o.ttc)


class AsyncDetector:
    """
    Threaded wrapper that runs CollisionDetector on a background daemon thread.
    The main loop hands frames in and reads results out without blocking.
    """

    def __init__(self, detector: CollisionDetector):
        """
        Args:
            detector: Initialized CollisionDetector instance
        """
        self.detector = detector
        self._lock = threading.Lock()
        self._frame = None
        self._lane_bounds: Tuple[Optional[float], Optional[float]] = (None, None)
        self._results: List[TrackedObject] = []
        self._closest_threat: Optional[TrackedObject] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self):
        """Start the background detection thread."""
        if not self.detector.is_initialized:
            logger.warning("Collision detector not initialized — skipping async start.")
            return

        self._running = True
        self._thread = threading.Thread(target=self._detection_loop, daemon=True)
        self._thread.start()
        logger.info("Async collision detector thread started.")

    def stop(self):
        """Stop the background detection thread."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        logger.info("Async collision detector thread stopped.")

    def update_frame(self, frame: np.ndarray,
                     left_intercept: Optional[float] = None,
                     right_intercept: Optional[float] = None):
        """
        Hand the latest frame to the detection thread.
        If the thread is still processing, this simply replaces the pending frame.
        No queue buildup — always processes the most recent frame.

        Args:
            frame: Current video frame
            left_intercept: X-coordinate where left lane line hits frame bottom
            right_intercept: X-coordinate where right lane line hits frame bottom
        """
        with self._lock:
            self._frame = frame
            self._lane_bounds = (left_intercept, right_intercept)

    def get_latest_results(self) -> Tuple[List[TrackedObject], Optional[TrackedObject]]:
        """
        Read the latest detection results from the background thread.

        Returns:
            (tracked_objects, closest_threat)
        """
        with self._lock:
            return list(self._results), self._closest_threat

    @staticmethod
    def _is_in_lane(det: Detection, left_x: Optional[float],
                    right_x: Optional[float], frame_width: int) -> bool:
        """
        Check whether a detection's center falls within the ego-lane corridor.

        Edge cases:
          - Both intercepts None → no lane data, fall back to center 40% of frame
          - Only left known  → corridor is [left, frame_width]
          - Only right known → corridor is [0, right]
          - Both known       → corridor is [left, right]
        """
        box_center_x = (det.bbox[0] + det.bbox[2]) / 2.0

        if left_x is None and right_x is None:
            # No lane data — use the central 40% of frame as a conservative guess
            margin = frame_width * 0.30
            return margin <= box_center_x <= (frame_width - margin)

        lo = left_x if left_x is not None else 0
        hi = right_x if right_x is not None else frame_width
        return lo <= box_center_x <= hi

    def _detection_loop(self):
        """Background loop: grab latest frame, run inference, filter by lane, update results."""
        while self._running:
            # Grab latest frame and lane bounds
            with self._lock:
                frame = self._frame
                self._frame = None  # Mark as consumed
                lane_left, lane_right = self._lane_bounds

            if frame is None:
                time.sleep(0.01)  # Avoid busy-wait
                continue

            try:
                current_time = time.time()
                h, w = frame.shape[:2]
                all_detections = self.detector.detect_objects(frame)

                # Filter: only keep vehicles inside the ego-lane corridor
                lane_detections = [
                    d for d in all_detections
                    if self._is_in_lane(d, lane_left, lane_right, w)
                ]

                tracked = self.detector.calculate_ttc(lane_detections, current_time)
                closest = self.detector.get_closest_threat(tracked)

                with self._lock:
                    self._results = tracked
                    self._closest_threat = closest

            except Exception as e:
                logger.error(f"Error in collision detection thread: {e}")
                time.sleep(0.1)


def create_collision_detector(model_dir: str = "models",
                              confidence_threshold: float = 0.5) -> Tuple[CollisionDetector, AsyncDetector]:
    """
    Factory function to create collision detector with async wrapper.

    Args:
        model_dir: Path to directory containing model files
        confidence_threshold: Minimum detection confidence

    Returns:
        Tuple of (CollisionDetector, AsyncDetector)
    """
    detector = CollisionDetector(
        model_dir=model_dir,
        confidence_threshold=confidence_threshold
    )
    async_detector = AsyncDetector(detector)
    return detector, async_detector
