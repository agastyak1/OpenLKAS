"""
Camera module for OpenLKAS (Open Lane Keeping Assist System)
Handles frame capture from live webcam or demo video files.
"""

import cv2
import os
from typing import Optional, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CameraModule:
    """
    Camera module for capturing frames from webcam or video files.
    """
    
    def __init__(self, source: str = 'live', video_path: str = None, 
                 resolution: Tuple[int, int] = (1280, 720), fps: int = 30):
        """
        Initialize camera module.
        
        Args:
            source: 'live' for webcam, 'demo' for video file
            video_path: Path to video file (required for demo mode)
            resolution: Camera resolution (width, height)
            fps: Target frame rate
        """
        self.source = source
        self.video_path = video_path
        self.resolution = resolution
        self.fps = fps
        self.cap = None
        self.is_initialized = False
        
        self._initialize_camera()
    
    def _initialize_camera(self):
        """Initialize camera or video capture based on source."""
        try:
            if self.source == 'live':
                self._initialize_webcam()
            elif self.source == 'demo':
                self._initialize_video()
            else:
                raise ValueError(f"Invalid source: {self.source}. Use 'live' or 'demo'")
                
        except Exception as e:
            logger.error(f"Failed to initialize camera: {e}")
            self.is_initialized = False
    
    def _initialize_webcam(self):
        """Initialize webcam capture."""
        logger.info("Initializing webcam...")
        
        # Try different camera indices
        for camera_index in [0, 1, 2]:
            self.cap = cv2.VideoCapture(camera_index)
            if self.cap.isOpened():
                logger.info(f"Webcam initialized successfully on index {camera_index}")
                break
        
        if not self.cap.isOpened():
            raise RuntimeError("No webcam found or accessible")
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        
        # Verify settings
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        
        logger.info(f"Camera settings - Width: {actual_width}, Height: {actual_height}, FPS: {actual_fps}")
        
        self.is_initialized = True
    
    def _initialize_video(self):
        """Initialize video file capture."""
        if not self.video_path:
            raise ValueError("Video path is required for demo mode")
        
        if not os.path.exists(self.video_path):
            raise FileNotFoundError(f"Video file not found: {self.video_path}")
        
        logger.info(f"Initializing video capture from: {self.video_path}")
        
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video file: {self.video_path}")
        
        # Get video properties
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = self.cap.get(cv2.CAP_PROP_FPS)
        video_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        logger.info(f"Video properties - Frames: {total_frames}, FPS: {video_fps:.2f}, "
                   f"Resolution: {video_width}x{video_height}")
        
        self.is_initialized = True
    
    def get_frame(self) -> Optional[Tuple[bool, object]]:
        """
        Capture a frame from camera or video.
        
        Returns:
            Tuple of (success, frame) where success is boolean and frame is numpy array
        """
        if not self.is_initialized or self.cap is None:
            logger.error("Camera not initialized")
            return False, None
        
        ret, frame = self.cap.read()
        
        if not ret:
            if self.source == 'demo':
                logger.info("End of video reached")
                # Reset video to beginning
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()
                if not ret:
                    logger.error("Failed to reset video")
                    return False, None
            else:
                logger.error("Failed to capture frame from webcam")
                return False, None
        
        return True, frame
    
    def get_frame_info(self) -> dict:
        """
        Get current frame information.
        
        Returns:
            Dictionary with frame information
        """
        if not self.is_initialized or self.cap is None:
            return {}
        
        info = {
            'width': int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': self.cap.get(cv2.CAP_PROP_FPS),
            'source': self.source
        }
        
        if self.source == 'demo':
            info['current_frame'] = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            info['total_frames'] = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        return info
    
    def release(self):
        """Release camera resources."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            self.is_initialized = False
            logger.info("Camera resources released")
    
    def __del__(self):
        """Destructor to ensure camera is released."""
        self.release()


def get_available_demo_videos(demo_dir: str = "demo_videos") -> list:
    """
    Get list of available demo video files.
    
    Args:
        demo_dir: Directory containing demo videos
        
    Returns:
        List of video file paths
    """
    if not os.path.exists(demo_dir):
        logger.warning(f"Demo directory not found: {demo_dir}")
        return []
    
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
    video_files = []
    
    for file in os.listdir(demo_dir):
        if any(file.lower().endswith(ext) for ext in video_extensions):
            video_files.append(os.path.join(demo_dir, file))
    
    logger.info(f"Found {len(video_files)} demo video files")
    return video_files


def create_camera_module(source: str = 'live', video_path: str = None, 
                        resolution: Tuple[int, int] = (1280, 720), 
                        fps: int = 30) -> CameraModule:
    """
    Factory function to create camera module with proper configuration.
    
    Args:
        source: 'live' for webcam, 'demo' for video file
        video_path: Path to video file (required for demo mode)
        resolution: Camera resolution
        fps: Target frame rate
        
    Returns:
        Initialized CameraModule instance
    """
    if source == 'demo' and not video_path:
        # Try to find a demo video automatically
        demo_videos = get_available_demo_videos()
        if demo_videos:
            video_path = demo_videos[0]
            logger.info(f"Using demo video: {video_path}")
        else:
            raise ValueError("No demo videos found in demo_videos directory")
    
    return CameraModule(source=source, video_path=video_path, 
                       resolution=resolution, fps=fps) 