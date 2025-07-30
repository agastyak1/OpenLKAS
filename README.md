# OpenLKAS
OpenLKAS is a lightweight, open-source Lane Keeping Assist System (LKAS) designed for DIY and educational use, intended to add a safety cushion to some of my older cars before the era of advanced safety systems. Built for the Raspberry Pi 4 and developed in Python, OpenLKAS uses a standard USB webcam or Pi Camera Module to detect road lanes and provide real-time audio warnings when the vehicle deviates from the lane. It offers both a live driving mode and a demo mode for testing on pre-recorded driving footage.
The project utilizes OpenCV for all image processing tasks, including edge detection (Canny), region masking, and Hough Line Transform to detect lanes. The system calculates the vehicle's position relative to the lane center and triggers beeping alerts using pygame when drifting is detected.

The system runs entirely on a Raspberry Pi 4 powered via USB-C:
- optionally cooled with a fan or heatsinks; recommended
- supports additional GPIO-based sensors for collision warning and is designed to be expanded with minimal hardware.

OpenLKAS is currently in development and is scheduled to complete its first software launch (open source) by September 2025. 
- I intend for this project to be used within DIY scopes and educational learning. Please don't rely solely on OpenLKAS as your primary safety system; always be attentive and safe while driving.
