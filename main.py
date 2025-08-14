#!/usr/bin/env python3
"""
Main control loop for OpenLKAS (Open Lane Keeping Assist System)
Integrates camera, lane detection, and audio alert modules.
"""

import cv2
import argparse
import time
import signal
import sys
import logging
from typing import Optional

# Import our modules
from camera_module import create_camera_module, get_available_demo_videos
from lane_detector import create_lane_detector
from audio_alert import create_audio_alert, LaneDepartureAlert
from utils import resize_image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class OpenLKAS:
    """
    Main OpenLKAS system that integrates all modules.
    """
    
    def __init__(self, mode: str = 'live', video_path: str = None,
                 threshold: float = 50.0, show_display: bool = True,
                 resolution: tuple = (1280, 720), fps: int = 30):
        """
        Initialize OpenLKAS system.
        
        Args:
            mode: 'live' for webcam, 'demo' for video file
            video_path: Path to demo video (required for demo mode)
            threshold: Lane departure threshold in pixels
            show_display: Whether to show video display
            resolution: Camera resolution
            fps: Target frame rate
        """
        self.mode = mode
        self.video_path = video_path
        self.threshold = threshold
        self.show_display = show_display
        self.resolution = resolution
        self.fps = fps
        
        # System components
        self.camera = None
        self.lane_detector = None
        self.audio_alert = None
        self.departure_alert = None
        
        # Control flags
        self.running = False
        self.frame_count = 0
        self.start_time = None
        
        # Initialize system
        self._initialize_system()
    
    def _initialize_system(self):
        """Initialize all system components."""
        try:
            logger.info("Initializing OpenLKAS system...")
            
            # Initialize camera module
            logger.info(f"Initializing camera in {self.mode} mode...")
            self.camera = create_camera_module(
                source=self.mode,
                video_path=self.video_path,
                resolution=self.resolution,
                fps=self.fps
            )
            
            # Initialize lane detector
            logger.info("Initializing lane detector...")
            self.lane_detector = create_lane_detector(
                departure_threshold=self.threshold
            )
            
            # Initialize audio alert system
            logger.info("Initializing audio alert system...")
            self.audio_alert = create_audio_alert(
                frequency=800,
                duration=0.3,
                volume=0.5,
                alert_cooldown=1.0
            )
            
            # Initialize lane departure alert system
            self.departure_alert = LaneDepartureAlert(self.audio_alert)
            
            logger.info("OpenLKAS system initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize system: {e}")
            self.cleanup()
            raise
    
    def run(self):
        """Main system loop."""
        if not self.camera or not self.camera.is_initialized:
            logger.error("Camera not initialized")
            return
        
        self.running = True
        self.start_time = time.time()
        
        logger.info("Starting OpenLKAS main loop...")
        logger.info(f"Mode: {self.mode}, Threshold: {self.threshold}px, Display: {self.show_display}")
        
        try:
            while self.running:
                # Capture frame
                ret, frame = self.camera.get_frame()
                if not ret:
                    logger.warning("Failed to capture frame")
                    continue
                
                # Process frame for lane detection
                detection_result = self.lane_detector.detect_lanes(frame)
                
                # Process lane departure alert
                self.departure_alert.process_departure(
                    detection_result['off_lane'],
                    detection_result['offset']
                )
                
                # Update frame count and calculate FPS
                self.frame_count += 1
                elapsed_time = time.time() - self.start_time
                current_fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
                
                # Display results
                if self.show_display and detection_result['processed_frame'] is not None:
                    display_frame = detection_result['processed_frame'].copy()
                    
                    # Add system info overlay
                    info_text = f"FPS: {current_fps:.1f} | Frame: {self.frame_count}"
                    cv2.putText(display_frame, info_text, (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    # Add mode info
                    mode_text = f"Mode: {self.mode.upper()}"
                    cv2.putText(display_frame, mode_text, (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    # Resize for display if needed
                    display_frame = resize_image(display_frame, width=1280)
                    
                    # Show frame
                    cv2.imshow('OpenLKAS - Lane Keeping Assist System', display_frame)
                    
                    # Handle key presses
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        logger.info("Quit requested by user")
                        break
                    elif key == ord('t'):
                        # Toggle threshold
                        new_threshold = self.threshold + 10 if self.threshold < 100 else 20
                        self.lane_detector.update_threshold(new_threshold)
                        self.threshold = new_threshold
                        logger.info(f"Threshold updated to: {new_threshold}px")
                    elif key == ord('v'):
                        # Toggle volume
                        current_volume = self.audio_alert.volume
                        new_volume = 0.0 if current_volume > 0.5 else 0.5
                        self.audio_alert.update_volume(new_volume)
                        logger.info(f"Volume updated to: {new_volume}")
                
                # Add small delay to control frame rate
                time.sleep(1.0 / self.fps)
                
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up system resources."""
        logger.info("Cleaning up system resources...")
        
        self.running = False
        
        # Clean up camera
        if self.camera:
            self.camera.release()
        
        # Clean up audio
        if self.departure_alert:
            self.departure_alert.cleanup()
        
        # Close display windows
        cv2.destroyAllWindows()
        
        logger.info("System cleanup completed")


def signal_handler(signum, frame):
    """Handle system signals for graceful shutdown."""
    logger.info(f"Received signal {signum}, shutting down gracefully...")
    sys.exit(0)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='OpenLKAS - Open Lane Keeping Assist System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --mode live                    # Live webcam mode
  python main.py --mode demo                    # Demo mode with auto-detected video
  python main.py --mode demo --video demo.mp4   # Demo mode with specific video
  python main.py --mode live --threshold 30     # Live mode with custom threshold
  python main.py --mode live --no-display       # Live mode without display
        """
    )
    
    parser.add_argument('--mode', choices=['live', 'demo'], default='live',
                       help='Operation mode: live (webcam) or demo (video file)')
    parser.add_argument('--video', type=str, default=None,
                       help='Path to demo video file (required for demo mode if no videos in demo_videos/)')
    parser.add_argument('--threshold', type=float, default=50.0,
                       help='Lane departure threshold in pixels (default: 50.0)')
    parser.add_argument('--no-display', action='store_true',
                       help='Disable video display (useful for headless operation)')
    parser.add_argument('--resolution', type=str, default='1280x720',
                       help='Camera resolution in format WxH (default: 1280x720)')
    parser.add_argument('--fps', type=int, default=30,
                       help='Target frame rate (default: 30)')
    parser.add_argument('--list-videos', action='store_true',
                       help='List available demo videos and exit')
    
    args = parser.parse_args()
    
    # Handle list videos option
    if args.list_videos:
        demo_videos = get_available_demo_videos()
        if demo_videos:
            print("Available demo videos:")
            for i, video in enumerate(demo_videos, 1):
                print(f"  {i}. {video}")
        else:
            print("No demo videos found in demo_videos/ directory")
        return
    
    # Parse resolution
    try:
        width, height = map(int, args.resolution.split('x'))
        resolution = (width, height)
    except ValueError:
        logger.error("Invalid resolution format. Use WxH (e.g., 1280x720)")
        return
    
    # Validate demo mode
    if args.mode == 'demo' and not args.video:
        demo_videos = get_available_demo_videos()
        if not demo_videos:
            logger.error("No demo videos found. Please provide --video path or add videos to demo_videos/")
            return
        args.video = demo_videos[0]
        logger.info(f"Using demo video: {args.video}")
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Create and run system
    try:
        system = OpenLKAS(
            mode=args.mode,
            video_path=args.video,
            threshold=args.threshold,
            show_display=not args.no_display,
            resolution=resolution,
            fps=args.fps
        )
        system.run()
    except Exception as e:
        logger.error(f"Failed to start system: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 