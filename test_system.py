#!/usr/bin/env python3
"""
Test script for OpenLKAS system
Verifies that all modules can be imported and basic functionality works.
"""

import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_imports():
    """Test that all modules can be imported."""
    logger.info("Testing module imports...")
    
    try:
        import cv2
        logger.info("✓ OpenCV imported successfully")
    except ImportError as e:
        logger.error(f"✗ Failed to import OpenCV: {e}")
        return False
    
    try:
        import numpy as np
        logger.info("✓ NumPy imported successfully")
    except ImportError as e:
        logger.error(f"✗ Failed to import NumPy: {e}")
        return False
    
    try:
        import pygame
        logger.info("✓ Pygame imported successfully")
    except ImportError as e:
        logger.error(f"✗ Failed to import Pygame: {e}")
        return False
    
    try:
        import imutils
        logger.info("✓ Imutils imported successfully")
    except ImportError as e:
        logger.error(f"✗ Failed to import Imutils: {e}")
        return False
    
    # Test our custom modules
    try:
        from utils import calculate_center_offset, is_lane_departure
        logger.info("✓ Utils module imported successfully")
    except ImportError as e:
        logger.error(f"✗ Failed to import Utils module: {e}")
        return False
    
    try:
        from camera_module import CameraModule, get_available_demo_videos
        logger.info("✓ Camera module imported successfully")
    except ImportError as e:
        logger.error(f"✗ Failed to import Camera module: {e}")
        return False
    
    try:
        from lane_detector import LaneDetector, create_lane_detector
        logger.info("✓ Lane detector module imported successfully")
    except ImportError as e:
        logger.error(f"✗ Failed to import Lane detector module: {e}")
        return False
    
    try:
        from audio_alert import AudioAlert, create_audio_alert
        logger.info("✓ Audio alert module imported successfully")
    except ImportError as e:
        logger.error(f"✗ Failed to import Audio alert module: {e}")
        return False
    
    return True


def test_utils_functions():
    """Test utility functions."""
    logger.info("Testing utility functions...")
    
    try:
        from utils import calculate_center_offset, is_lane_departure
        
        # Test center offset calculation
        offset = calculate_center_offset(100, 50)
        assert offset == 50, f"Expected offset 50, got {offset}"
        logger.info("✓ Center offset calculation works")
        
        # Test lane departure detection
        is_departing = is_lane_departure(60, 50)
        assert is_departing == True, f"Expected True for departure, got {is_departing}"
        
        is_not_departing = is_lane_departure(30, 50)
        assert is_not_departing == False, f"Expected False for no departure, got {is_not_departing}"
        logger.info("✓ Lane departure detection works")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Utility function test failed: {e}")
        return False


def test_camera_module():
    """Test camera module creation (without actual camera)."""
    logger.info("Testing camera module...")
    
    try:
        from camera_module import get_available_demo_videos
        
        # Test demo video detection
        demo_videos = get_available_demo_videos()
        logger.info(f"✓ Found {len(demo_videos)} demo videos")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Camera module test failed: {e}")
        return False


def test_lane_detector():
    """Test lane detector creation."""
    logger.info("Testing lane detector...")
    
    try:
        from lane_detector import create_lane_detector
        
        # Create lane detector
        detector = create_lane_detector(departure_threshold=50.0)
        assert detector is not None, "Lane detector creation failed"
        logger.info("✓ Lane detector created successfully")
        
        # Test stats
        stats = detector.get_detection_stats()
        assert 'departure_threshold' in stats, "Stats missing departure_threshold"
        logger.info("✓ Lane detector stats work")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Lane detector test failed: {e}")
        return False


def test_audio_alert():
    """Test audio alert creation."""
    logger.info("Testing audio alert...")
    
    try:
        from audio_alert import create_audio_alert
        
        # Create audio alert
        alert = create_audio_alert(frequency=800, volume=0.5)
        assert alert is not None, "Audio alert creation failed"
        logger.info("✓ Audio alert created successfully")
        
        # Test stats
        stats = alert.get_audio_stats()
        assert 'frequency' in stats, "Stats missing frequency"
        logger.info("✓ Audio alert stats work")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Audio alert test failed: {e}")
        return False


def main():
    """Run all tests."""
    logger.info("Starting OpenLKAS system tests...")
    
    tests = [
        ("Module Imports", test_imports),
        ("Utility Functions", test_utils_functions),
        ("Camera Module", test_camera_module),
        ("Lane Detector", test_lane_detector),
        ("Audio Alert", test_audio_alert)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n--- Testing {test_name} ---")
        if test_func():
            passed += 1
            logger.info(f"✓ {test_name} PASSED")
        else:
            logger.error(f"✗ {test_name} FAILED")
    
    logger.info(f"\n--- Test Results ---")
    logger.info(f"Passed: {passed}/{total}")
    
    if passed == total:
        logger.info("🎉 All tests passed! System is ready to use.")
        return 0
    else:
        logger.error("❌ Some tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 