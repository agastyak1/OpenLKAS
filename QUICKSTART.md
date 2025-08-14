# OpenLKAS Quick Start Guide

Get your OpenLKAS (Open Lane Keeping Assist System) running in minutes!

## 🚀 Quick Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Test the System
```bash
python test_system.py
```
You should see: `🎉 All tests passed! System is ready to use.`

### 3. Run in Demo Mode (Recommended for first test)
```bash
python main.py --mode demo
```
*Note: You'll need to add a video file to the `demo_videos/` directory first*

### 4. Run in Live Mode (Webcam)
```bash
python main.py --mode live
```

## 🎮 Controls

When running with display enabled:
- **Q**: Quit the application
- **T**: Toggle threshold sensitivity
- **V**: Toggle audio volume

## 📁 Adding Demo Videos

1. Place your test video files in the `demo_videos/` directory
2. Supported formats: MP4, AVI, MOV, MKV, WMV
3. Check available videos: `python main.py --list-videos`

## ⚙️ Configuration Options

### Basic Usage
```bash
# Live mode with custom threshold
python main.py --mode live --threshold 30

# Demo mode with specific video
python main.py --mode demo --video demo_videos/my_video.mp4

# Headless operation (no display)
python main.py --mode live --no-display

# Custom resolution
python main.py --mode live --resolution 640x480
```

### Advanced Options
```bash
# Custom frame rate
python main.py --mode live --fps 15

# List all available demo videos
python main.py --list-videos

# Get help
python main.py --help
```

## 🔧 Troubleshooting

### No Webcam Found
- Ensure your webcam is connected and accessible
- Try different camera indices (system will auto-detect)
- Check permissions for camera access

### Audio Issues
- Ensure your system has audio output
- Check pygame installation: `python -c "import pygame; pygame.mixer.init()"`

### Performance Issues
- Lower resolution: `--resolution 640x480`
- Lower frame rate: `--fps 15`
- Disable display: `--no-display`

### Lane Detection Issues
- Adjust threshold: `--threshold 30` (lower = more sensitive)
- Ensure good lighting conditions
- Use videos with clear lane markings

## 📊 System Requirements

- **Python**: 3.7+
- **RAM**: 2GB+ (4GB recommended)
- **Storage**: 100MB free space
- **Camera**: USB webcam (720p recommended)
- **Audio**: Speakers or headphones

## 🎯 Expected Behavior

1. **Normal Driving**: No alerts, green lane lines visible
2. **Lane Departure**: Red warning text + audio beep
3. **Continuous Departure**: Continuous audio alerts
4. **Return to Lane**: Alerts stop, normal operation resumes

## 📝 Next Steps

1. Test with demo videos to understand detection
2. Adjust threshold for your driving conditions
3. Mount camera securely in your vehicle
4. Test in real driving conditions

## 🆘 Need Help?

- Check the main README.md for detailed documentation
- Review the code comments for technical details
- Test individual modules using `test_system.py` 