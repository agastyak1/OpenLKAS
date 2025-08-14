"""
Audio alert module for OpenLKAS (Open Lane Keeping Assist System)
Handles audio alerts when lane departure is detected.
"""

import pygame
import numpy as np
import time
import threading
from typing import Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AudioAlert:
    """
    Audio alert system using pygame for lane departure warnings.
    """
    
    def __init__(self, 
                 frequency: int = 800,
                 duration: float = 0.3,
                 volume: float = 0.5,
                 sample_rate: int = 44100,
                 alert_cooldown: float = 1.0):
        """
        Initialize audio alert system.
        
        Args:
            frequency: Frequency of the beep sound in Hz
            duration: Duration of each beep in seconds
            volume: Volume level (0.0 to 1.0)
            sample_rate: Audio sample rate
            alert_cooldown: Minimum time between alerts in seconds
        """
        self.frequency = frequency
        self.duration = duration
        self.volume = max(0.0, min(1.0, volume))  # Clamp between 0 and 1
        self.sample_rate = sample_rate
        self.alert_cooldown = alert_cooldown
        
        # State tracking
        self.last_alert_time = 0
        self.is_initialized = False
        self.alert_thread = None
        self.stop_alert = False
        
        # Initialize pygame mixer
        self._initialize_audio()
    
    def _initialize_audio(self):
        """Initialize pygame audio system."""
        try:
            pygame.mixer.init(frequency=self.sample_rate, size=-16, channels=1, buffer=512)
            self.is_initialized = True
            logger.info(f"Audio system initialized - Frequency: {self.frequency}Hz, "
                       f"Duration: {self.duration}s, Volume: {self.volume}")
        except Exception as e:
            logger.error(f"Failed to initialize audio system: {e}")
            self.is_initialized = False
    
    def _generate_beep_sound(self) -> np.ndarray:
        """
        Generate a beep sound waveform.
        
        Returns:
            Audio samples as numpy array
        """
        # Generate time array
        t = np.linspace(0, self.duration, int(self.sample_rate * self.duration), False)
        
        # Generate sine wave
        tone = np.sin(2 * np.pi * self.frequency * t)
        
        # Apply volume
        tone = tone * self.volume
        
        # Convert to 16-bit integer format
        tone = (tone * 32767).astype(np.int16)
        
        return tone
    
    def _play_beep_threaded(self):
        """Play beep sound in a separate thread to avoid blocking."""
        try:
            # Generate beep sound
            beep_sound = self._generate_beep_sound()
            
            # Create pygame Sound object
            sound = pygame.sndarray.make_sound(beep_sound)
            
            # Play the sound
            sound.play()
            
            # Wait for sound to finish
            pygame.time.wait(int(self.duration * 1000))
            
        except Exception as e:
            logger.error(f"Error playing beep sound: {e}")
    
    def play_beep(self, force: bool = False) -> bool:
        """
        Play a beep sound if cooldown period has passed.
        
        Args:
            force: Force play regardless of cooldown
            
        Returns:
            True if beep was played, False otherwise
        """
        if not self.is_initialized:
            logger.warning("Audio system not initialized")
            return False
        
        current_time = time.time()
        
        # Check cooldown period
        if not force and (current_time - self.last_alert_time) < self.alert_cooldown:
            return False
        
        # Update last alert time
        self.last_alert_time = current_time
        
        # Play beep in separate thread
        self.alert_thread = threading.Thread(target=self._play_beep_threaded)
        self.alert_thread.daemon = True
        self.alert_thread.start()
        
        logger.debug("Beep alert triggered")
        return True
    
    def play_continuous_alert(self, duration: float = 2.0):
        """
        Play continuous beep alerts for a specified duration.
        
        Args:
            duration: Total duration of continuous alerts in seconds
        """
        if not self.is_initialized:
            logger.warning("Audio system not initialized")
            return
        
        start_time = time.time()
        self.stop_alert = False
        
        def continuous_beep():
            while time.time() - start_time < duration and not self.stop_alert:
                self.play_beep(force=True)
                time.sleep(self.alert_cooldown)
        
        # Start continuous alert in separate thread
        self.alert_thread = threading.Thread(target=continuous_beep)
        self.alert_thread.daemon = True
        self.alert_thread.start()
        
        logger.info(f"Continuous alert started for {duration} seconds")
    
    def stop_continuous_alert(self):
        """Stop any ongoing continuous alert."""
        self.stop_alert = True
        if self.alert_thread and self.alert_thread.is_alive():
            self.alert_thread.join(timeout=1.0)
        logger.info("Continuous alert stopped")
    
    def update_volume(self, new_volume: float):
        """
        Update the volume level.
        
        Args:
            new_volume: New volume level (0.0 to 1.0)
        """
        self.volume = max(0.0, min(1.0, new_volume))
        logger.info(f"Updated volume to: {self.volume}")
    
    def update_frequency(self, new_frequency: int):
        """
        Update the beep frequency.
        
        Args:
            new_frequency: New frequency in Hz
        """
        self.frequency = max(100, min(2000, new_frequency))  # Clamp between 100 and 2000 Hz
        logger.info(f"Updated frequency to: {self.frequency}Hz")
    
    def update_duration(self, new_duration: float):
        """
        Update the beep duration.
        
        Args:
            new_duration: New duration in seconds
        """
        self.duration = max(0.1, min(2.0, new_duration))  # Clamp between 0.1 and 2.0 seconds
        logger.info(f"Updated duration to: {self.duration}s")
    
    def get_audio_stats(self) -> dict:
        """
        Get current audio configuration.
        
        Returns:
            Dictionary with audio parameters
        """
        return {
            'frequency': self.frequency,
            'duration': self.duration,
            'volume': self.volume,
            'sample_rate': self.sample_rate,
            'alert_cooldown': self.alert_cooldown,
            'is_initialized': self.is_initialized,
            'last_alert_time': self.last_alert_time
        }
    
    def cleanup(self):
        """Clean up audio resources."""
        self.stop_continuous_alert()
        if self.is_initialized:
            pygame.mixer.quit()
            self.is_initialized = False
            logger.info("Audio system cleaned up")
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        self.cleanup()


def create_audio_alert(frequency: int = 800,
                      duration: float = 0.3,
                      volume: float = 0.5,
                      alert_cooldown: float = 1.0) -> AudioAlert:
    """
    Factory function to create audio alert with common configurations.
    
    Args:
        frequency: Beep frequency in Hz
        duration: Beep duration in seconds
        volume: Volume level (0.0 to 1.0)
        alert_cooldown: Cooldown between alerts in seconds
        
    Returns:
        Configured AudioAlert instance
    """
    return AudioAlert(
        frequency=frequency,
        duration=duration,
        volume=volume,
        alert_cooldown=alert_cooldown
    )


class LaneDepartureAlert:
    """
    High-level lane departure alert system that combines detection with audio alerts.
    """
    
    def __init__(self, audio_alert: AudioAlert = None):
        """
        Initialize lane departure alert system.
        
        Args:
            audio_alert: AudioAlert instance (None to create default)
        """
        self.audio_alert = audio_alert if audio_alert else create_audio_alert()
        self.last_departure_state = False
        self.departure_start_time = None
        self.continuous_alert_threshold = 2.0  # Seconds before continuous alert
        
    def process_departure(self, is_departing: bool, offset: float = 0.0):
        """
        Process lane departure detection and trigger appropriate alerts.
        
        Args:
            is_departing: Whether lane departure is detected
            offset: Offset from lane center (for logging)
        """
        current_time = time.time()
        
        if is_departing:
            # Lane departure detected
            if not self.last_departure_state:
                # Just started departing
                self.departure_start_time = current_time
                self.audio_alert.play_beep()
                logger.info(f"Lane departure detected - Offset: {offset:.1f}px")
            else:
                # Still departing - check if we should start continuous alert
                if (self.departure_start_time and 
                    current_time - self.departure_start_time > self.continuous_alert_threshold):
                    # Start continuous alert if not already playing
                    if not self.audio_alert.alert_thread or not self.audio_alert.alert_thread.is_alive():
                        self.audio_alert.play_continuous_alert(duration=5.0)
        else:
            # No departure detected
            if self.last_departure_state:
                # Just returned to lane
                self.audio_alert.stop_continuous_alert()
                self.departure_start_time = None
                logger.info("Vehicle returned to lane")
        
        self.last_departure_state = is_departing
    
    def cleanup(self):
        """Clean up resources."""
        self.audio_alert.cleanup() 