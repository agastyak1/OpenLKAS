#!/usr/bin/env python3
"""
Main control loop for OpenLCWS (Open Lane and Collision Warning System)
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
from audio_alert import create_audio_alert, LaneDepartureAlert, CollisionAlert
from collision_detector import create_collision_detector
from utils import resize_image, draw_detection_boxes, draw_collision_warning

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class OpenLCWS:
    """
    Main OpenLCWS system that integrates all modules.
    """
    
    def __init__(self, mode: str = 'live', video_path: str = None,
                 threshold: float = 50.0, show_display: bool = True,
                 resolution: tuple = (1280, 720), fps: int = 30,
                 car_width: float = 70.0, lane_width: float = 144.0, camera_offset: float = 0.0,
                 enable_fcw: bool = False, fcw_confidence: float = 0.5):
        """
        Initialize OpenLCWS system.
        
        Args:
            mode: 'live' for webcam, 'demo' for video file
            video_path: Path to demo video (required for demo mode)
            threshold: Lane departure threshold in pixels (fallback)
            show_display: Whether to show video display
            resolution: Camera resolution
            fps: Target frame rate
            car_width: Real-world width of car (inches)
            lane_width: Real-world lane width (inches)
            camera_offset: Offset of camera from true center of car (inches)
            enable_fcw: Enable Forward Collision Warning
            fcw_confidence: Minimum confidence for FCW detections
        """
        self.mode = mode
        self.video_path = video_path
        self.threshold = threshold
        self.show_display = show_display
        self.resolution = resolution
        self.fps = fps
        self.car_width = car_width
        self.lane_width = lane_width
        self.camera_offset = camera_offset
        self.enable_fcw = enable_fcw
        self.fcw_confidence = fcw_confidence
        self.fcw_active = enable_fcw  # Runtime toggle state
        
        # System components
        self.camera = None
        self.lane_detector = None
        self.audio_alert = None
        self.departure_alert = None
        self.collision_detector = None
        self.async_detector = None
        self.collision_alert = None
        
        # Control flags
        self.running = False
        self.frame_count = 0
        self.start_time = None
        
        # Performance tracking
        self.frame_times = []

        # Initialize system
        self._initialize_system()
    
    def _initialize_system(self):
        """Initialize all system components."""
        try:
            logger.info("Initializing OpenLCWS system...")
            
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
                departure_threshold=self.threshold,
                car_width=self.car_width,
                lane_width=self.lane_width,
                camera_offset=self.camera_offset
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
            
            # Initialize collision detection (FCW) if enabled
            if self.enable_fcw:
                logger.info("Initializing Forward Collision Warning...")
                try:
                    _, self.async_detector = create_collision_detector(
                        confidence_threshold=self.fcw_confidence
                    )
                    self.collision_alert = CollisionAlert(self.audio_alert)
                    self.async_detector.start()
                    logger.info("FCW system initialized successfully")
                except Exception as e:
                    logger.warning(f"FCW initialization failed (continuing without it): {e}")
                    self.enable_fcw = False
                    self.fcw_active = False
            
            logger.info("OpenLCWS system initialized successfully")
            
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
        
        logger.info("Starting OpenLCWS main loop...")
        logger.info(f"Mode: {self.mode}, Threshold: {self.threshold}px, Display: {self.show_display}")
        
        try:
            while self.running:
                frame_start_time = time.time()
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
                
                # Process Forward Collision Warning (async — non-blocking)
                fcw_tracked = []
                fcw_threat = None
                if self.fcw_active and self.async_detector:
                    # Pass lane intercepts so FCW only alerts on vehicles in our lane
                    self.async_detector.update_frame(
                        frame,
                        left_intercept=detection_result.get('left_intercept'),
                        right_intercept=detection_result.get('right_intercept')
                    )
                    fcw_tracked, fcw_threat = self.async_detector.get_latest_results()
                    if self.collision_alert:
                        ttc = fcw_threat.ttc if fcw_threat else None
                        self.collision_alert.process_collision(ttc)
                
                # Track frame time for moving average FPS
                frame_end_time = time.time()
                self.frame_times.append(frame_end_time)
                if len(self.frame_times) > 30:
                    self.frame_times.pop(0)

                # Update frame count and calculate moving average FPS
                self.frame_count += 1
                if len(self.frame_times) > 1:
                    time_diff = self.frame_times[-1] - self.frame_times[0]
                    current_fps = (len(self.frame_times) - 1) / time_diff if time_diff > 0 else 0.0
                else:
                    current_fps = 0.0
                
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
                    
                    # Draw FCW overlays
                    if self.fcw_active and fcw_tracked:
                        display_frame = draw_detection_boxes(display_frame, fcw_tracked)
                        if fcw_threat and fcw_threat.ttc != float('inf'):
                            display_frame = draw_collision_warning(display_frame, fcw_threat.ttc)
                    
                    # Draw FCW status indicator
                    if self.enable_fcw:
                        fcw_status = "FCW: ON" if self.fcw_active else "FCW: OFF"
                        fcw_color = (0, 255, 0) if self.fcw_active else (0, 0, 255)
                        cv2.putText(display_frame, fcw_status, (display_frame.shape[1] - 120, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, fcw_color, 2)
                    
                    # Resize for display if needed
                    display_frame = resize_image(display_frame, width=1280)
                    
                    # Show frame
                    cv2.imshow('OpenLCWS - Lane and Collision Warning System', display_frame)
                    
                    # Handle key presses
                    key = cv2.waitKeyEx(1)
                    char_key = key & 0xFF
                    
                    if char_key == ord('q'):
                        logger.info("Quit requested by user")
                        break
                    elif key in [63232, 2490368, 0] or char_key == ord('w'): # UP
                        self.lane_detector.offset_roi(0, -20)
                    elif key in [63233, 2621440, 1] or char_key == ord('s'): # DOWN
                        self.lane_detector.offset_roi(0, 20)
                    elif key in [63234, 2424832, 2] or char_key == ord('a'): # LEFT
                        self.lane_detector.offset_roi(-20, 0)
                    elif key in [63235, 2555904, 3] or char_key == ord('d'): # RIGHT
                        self.lane_detector.offset_roi(20, 0)
                    elif char_key == ord('i'):             # Top Edge UP
                        self.lane_detector.adjust_top_edge(-10)
                    elif char_key == ord('k'):             # Top Edge DOWN
                        self.lane_detector.adjust_top_edge(10)
                    elif char_key == ord('j'):             # Top Width SHRINK
                        self.lane_detector.adjust_top_width(-10)
                    elif char_key == ord('l'):             # Top Width EXPAND
                        self.lane_detector.adjust_top_width(10)
                    elif char_key == ord('t'):             # Bottom Edge UP
                        self.lane_detector.adjust_bottom_edge(-10)
                    elif char_key == ord('g'):             # Bottom Edge DOWN
                        self.lane_detector.adjust_bottom_edge(10)
                    elif char_key == ord('f'):             # Bottom Width SHRINK
                        self.lane_detector.adjust_bottom_width(-10)
                    elif char_key == ord('h'):             # Bottom Width EXPAND
                        self.lane_detector.adjust_bottom_width(10)
                    elif char_key == ord('c'):             # Auto Calibrate
                        self.lane_detector.auto_calibrate_roi(frame)
                    elif char_key == ord('n'):             # Toggle FCW
                        if self.enable_fcw:
                            self.fcw_active = not self.fcw_active
                            logger.info(f"FCW toggled: {'ON' if self.fcw_active else 'OFF'}")
                        else:
                            logger.info("FCW not available (start with --enable-fcw)")
                    elif char_key == ord('v'):
                        # Toggle volume
                        current_volume = self.audio_alert.volume
                        new_volume = 0.0 if current_volume > 0.5 else 0.5
                        self.audio_alert.update_volume(new_volume)
                        logger.info(f"Volume updated to: {new_volume}")
                
                # Add delay to control frame rate (only in demo mode to prevent live webcam buffer latency)
                if self.mode == 'demo':
                    processing_time = time.time() - frame_start_time
                    sleep_time = max(0.0, (1.0 / self.fps) - processing_time)
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                
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
        
        # Clean up collision detector
        if self.async_detector:
            self.async_detector.stop()
        if self.collision_alert:
            self.collision_alert.cleanup()
        
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
        description='OpenLCWS - Open Lane and Collision Warning System',
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
    parser.add_argument('--car-width', type=float, default=70.0,
                       help='Width of the car in inches (default: 70.0)')
    parser.add_argument('--lane-width', type=float, default=144.0,
                       help='Standard lane width in inches (default: 144.0 for US Highway)')
    parser.add_argument('--camera-offset', type=float, default=0.0,
                       help='Camera offset from center in inches (positive=right, default: 0.0)')
    parser.add_argument('--enable-fcw', action='store_true',
                       help='Enable Forward Collision Warning system')
    parser.add_argument('--fcw-confidence', type=float, default=0.5,
                       help='FCW detection confidence threshold (default: 0.5)')
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
        system = OpenLCWS(
            mode=args.mode,
            video_path=args.video,
            threshold=args.threshold,
            show_display=not args.no_display,
            resolution=resolution,
            fps=args.fps,
            car_width=args.car_width,
            lane_width=args.lane_width,
            camera_offset=args.camera_offset,
            enable_fcw=args.enable_fcw,
            fcw_confidence=args.fcw_confidence
        )
        system.run()
    except Exception as e:
        logger.error(f"Failed to start system: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 