# OpenLKAS (Open Lane Keeping Assist System)

A Python-based lane departure warning system designed for real-time use on Raspberry Pi 4 with Logitech 720p webcam.

## 🎯 Objective

Build a computer vision-based lane departure warning system that detects lane markings in real-time and triggers audio alerts when the vehicle drifts from the lane center.

## 🧱 Hardware Requirements

- Raspberry Pi 4 (4GB or 8GB)
- Logitech C270 720p webcam (USB connected)
- Passive or active buzzer/speaker (3.5mm jack or pygame audio output)
- microSD card with Raspberry Pi OS
- (Optional) PiTFT or HDMI monitor for live display

## 🧰 Software Dependencies

```bash
pip install opencv-python
pip install numpy
pip install pygame
pip install imutils
```

## 📁 Project Structure

```
OpenLKAS/
├── main.py                 # Main control loop
├── camera_module.py        # Camera/video frame capture
├── lane_detector.py        # Lane detection and drift calculation
├── audio_alert.py          # Audio alert system
├── utils.py               # Helper functions
├── demo_videos/           # Sample video files for testing
└── README.md
```

## 🚀 Usage

### Demo Mode (Testing with pre-recorded videos)
```bash
python main.py --mode demo
```

### Live Mode (Real-time webcam)
```bash
python main.py --mode live
```

### Configuration Options
```bash
python main.py --mode live --threshold 50 --show-display
```

## 🔧 Configuration

Key parameters can be adjusted in the respective modules:

- **Lane departure threshold**: Adjust sensitivity in `lane_detector.py`
- **Camera resolution**: Modify in `camera_module.py`
- **Audio alert volume**: Configure in `audio_alert.py`

## 🧪 Testing

1. Start with demo mode using sample videos in `demo_videos/`
2. Validate lane detection accuracy
3. Switch to live mode for real-time testing
4. Verify audio alerts trigger correctly

## 📋 Features

- ✅ Real-time lane detection using OpenCV
- ✅ Audio alerts for lane departure
- ✅ Live and demo modes
- ✅ Configurable sensitivity
- ✅ Visual overlays and debugging
- ✅ Modular architecture

## ❌ Excluded Features

- Forward collision detection
- Object detection (YOLO models)
- Ultrasonic/GPIO sensor integration

## 🔍 Technical Details

The system uses:
- **Canny Edge Detection** for lane line identification
- **Hough Line Transform** for line detection
- **Region of Interest** masking for road area focus
- **Center calculation** for drift detection
- **Pygame** for audio alerts

## 📝 License

Open source project for educational and research purposes. 🔧 Project Name: OpenLKAS (Open Lane Keeping Assist System)
🧠 Objective:

Build a Python-based lane departure warning system for real-time use on a Raspberry Pi 4, using a Logitech 720p webcam. The system should use computer vision (OpenCV) to detect lane markings in front of the vehicle, determine if the car is veering off the lane, and trigger an audio beep if lane departure is detected. The system will support both Live Mode (real-time from webcam) and Demo Mode (processing pre-recorded video).
🧱 Hardware:

    Raspberry Pi 4 (4GB or 8GB)

    Logitech C270 720p webcam (connected via USB)

    Passive or active buzzer/speaker (via 3.5mm jack or pygame audio output)

    microSD card with Raspberry Pi OS

    (Optional: PiTFT or HDMI monitor for live display)

🧰 Required Software:

    Python 3 (pre-installed)

    OpenCV (cv2)

    NumPy

    pygame for audio alerts

    (Optional: imutils, argparse, etc.)

📦 Modules & Architecture:

The software will be modular and structured like this:
📁 File Structure

OpenLKAS/
├── main.py
├── camera_module.py         # Captures frames from webcam or video
├── lane_detector.py         # Detects lane lines and calculates drift
├── audio_alert.py           # Plays a beep if drifting off
├── utils.py                 # Helper functions
├── demo_videos/             # Contains sample video files
└── README.md

🧩 Module Descriptions
1. camera_module.py

    Function: get_frame(source='live' or 'demo')

    If in live mode, captures frames from Logitech webcam at 720p resolution.

    If in demo mode, reads frames from a sample dashcam video.

    Returns each frame (BGR) to be processed.

2. lane_detector.py

    Processes each frame using OpenCV:

        Converts to grayscale

        Applies Gaussian blur

        Uses Canny edge detection

        Applies region-of-interest mask to focus on road area

        Uses Hough Line Transform to find lane lines

    Calculates center of lane vs center of car.

    Outputs:

        Lane lines overlay

        Boolean flag off_lane (True if car is drifting left/right outside lane)

3. audio_alert.py

    Uses pygame.mixer to load a beep sound.

    Function: play_beep() when off_lane == True

4. main.py

    Main control loop.

    Accepts CLI argument: --mode=live or --mode=demo

    Continuously:

        Captures a frame

        Detects lane drift

        If lane departure detected → play audio alert

        Displays output frame with overlays

🖼️ Visualization Features:

    Draw lane lines on frame

    Highlight left/right lane drift

    Display message on screen if alert is triggered

🧪 Testing Instructions:

    Run main.py in demo mode with --mode=demo and test using videos in demo_videos/

    Once lane detection is validated, switch to --mode=live for real-time Raspberry Pi camera input

    Make sure audio plays when the car drifts off center

❌ Explicitly Exclude:

    Do not include any forward-collision detection

    No object detection or YOLO models🔧 Project Name: OpenLKAS (Open Lane Keeping Assist System)
🧠 Objective:

Build a Python-based lane departure warning system for real-time use on a Raspberry Pi 4, using a Logitech 720p webcam. The system should use computer vision (OpenCV) to detect lane markings in front of the vehicle, determine if the car is veering off the lane, and trigger an audio beep if lane departure is detected. The system will support both Live Mode (real-time from webcam) and Demo Mode (processing pre-recorded video).
🧱 Hardware:

    Raspberry Pi 4 (4GB or 8GB)

    Logitech C270 720p webcam (connected via USB)

    Passive or active buzzer/speaker (via 3.5mm jack or pygame audio output)

    microSD card with Raspberry Pi OS

    (Optional: PiTFT or HDMI monitor for live display)

🧰 Required Software:

    Python 3 (pre-installed)

    OpenCV (cv2)

    NumPy

    pygame for audio alerts

    (Optional: imutils, argparse, etc.)

📦 Modules & Architecture:

The software will be modular and structured like this:
📁 File Structure

OpenLKAS/
├── main.py
├── camera_module.py         # Captures frames from webcam or video
├── lane_detector.py         # Detects lane lines and calculates drift
├── audio_alert.py           # Plays a beep if drifting off
├── utils.py                 # Helper functions
├── demo_videos/             # Contains sample video files
└── README.md

🧩 Module Descriptions
1. camera_module.py

    Function: get_frame(source='live' or 'demo')

    If in live mode, captures frames from Logitech webcam at 720p resolution.

    If in demo mode, reads frames from a sample dashcam video.

    Returns each frame (BGR) to be processed.

2. lane_detector.py

    Processes each frame using OpenCV:

        Converts to grayscale

        Applies Gaussian blur

        Uses Canny edge detection

        Applies region-of-interest mask to focus on road area

        Uses Hough Line Transform to find lane lines

    Calculates center of lane vs center of car.

    Outputs:

        Lane lines overlay

        Boolean flag off_lane (True if car is drifting left/right outside lane)

3. audio_alert.py

    Uses pygame.mixer to load a beep sound.

    Function: play_beep() when off_lane == True

4. main.py

    Main control loop.

    Accepts CLI argument: --mode=live or --mode=demo

    Continuously:

        Captures a frame

        Detects lane drift

        If lane departure detected → play audio alert

        Displays output frame with overlays

🖼️ Visualization Features:

    Draw lane lines on frame

    Highlight left/right lane drift

    Display message on screen if alert is triggered

🧪 Testing Instructions:

    Run main.py in demo mode with --mode=demo and test using videos in demo_videos/

    Once lane detection is validated, switch to --mode=live for real-time Raspberry Pi camera input

    Make sure audio plays when the car drifts off center
🔧 Project Name: OpenLKAS (Open Lane Keeping Assist System)
🧠 Objective:

Build a Python-based lane departure warning system for real-time use on a Raspberry Pi 4, using a Logitech 720p webcam. The system should use computer vision (OpenCV) to detect lane markings in front of the vehicle, determine if the car is veering off the lane, and trigger an audio beep if lane departure is detected. The system will support both Live Mode (real-time from webcam) and Demo Mode (processing pre-recorded video).
🧱 Hardware:

    Raspberry Pi 4 (4GB or 8GB)

    Logitech C270 720p webcam (connected via USB)

    Passive or active buzzer/speaker (via 3.5mm jack or pygame audio output)

    microSD card with Raspberry Pi OS

    (Optional: PiTFT or HDMI monitor for live display)

🧰 Required Software:

    Python 3 (pre-installed)

    OpenCV (cv2)

    NumPy

    pygame for audio alerts

    (Optional: imutils, argparse, etc.)

📦 Modules & Architecture:

The software will be modular and structured like this:
📁 File Structure

OpenLKAS/
├── main.py
├── camera_module.py         # Captures frames from webcam or video
├── lane_detector.py         # Detects lane lines and calculates drift
├── audio_alert.py           # Plays a beep if drifting off
├── utils.py                 # Helper functions
├── demo_videos/             # Contains sample video files
└── README.md

🧩 Module Descriptions
1. camera_module.py

    Function: get_frame(source='live' or 'demo')

    If in live mode, captures frames from Logitech webcam at 720p resolution.

    If in demo mode, reads frames from a sample dashcam video.

    Returns each frame (BGR) to be processed.

2. lane_detector.py

    Processes each frame using OpenCV:

        Converts to grayscale

        Applies Gaussian blur

        Uses Canny edge detection

        Applies region-of-interest mask to focus on road area

        Uses Hough Line Transform to find lane lines

    Calculates center of lane vs center of car.

    Outputs:

        Lane lines overlay

        Boolean flag off_lane (True if car is drifting left/right outside lane)

3. audio_alert.py

    Uses pygame.mixer to load a beep sound.

    Function: play_beep() when off_lane == True

4. main.py

    Main control loop.

    Accepts CLI argument: --mode=live or --mode=demo

    Continuously:

        Captures a frame

        Detects lane drift

        If lane departure detected → play audio alert

        Displays output frame with overlays

🖼️ Visualization Features:

    Draw lane lines on frame

    Highlight left/right lane drift

    Display message on screen if alert is triggered

🧪 Testing Instructions:

    Run main.py in demo mode with --mode=demo and test using videos in demo_videos/

    Once lane detection is validated, switch to --mode=live for real-time Raspberry Pi camera input

    Make sure audio plays when the car drifts off center

❌ Explicitly Exclude:

    Do not include any forward-collision detection

    No object detection or YOLO models

    No ultrasonic/GPIO sensor logic for now

✅ Deliverables:

    Fully modular Python code

    Easily configurable threshold for lane departure (in pixels from center)

    Toggleable demo vs live mode

    Readable overlays and structured logs (if needed)

❌ Explicitly Exclude:

    Do not include any forward-collision detection

    No object detection or YOLO models

    No ultrasonic/GPIO sensor logic for now

✅ Deliverables:

    Fully modular Python code

    Easily configurable threshold for lane departure (in pixels from center)

    Toggleable demo vs live mode

    Readable overlays and structured logs (if needed)


    No ultrasonic/GPIO sensor logic for now

✅ Deliverables:

    Fully modular Python code

    Easily configurable threshold for lane departure (in pixels from center)

    Toggleable demo vs live mode

    Readable overlays and structured logs (if needed)
