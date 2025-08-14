
## 🧰 Software Dependencies

```bash
pip install opencv-python
pip install numpy
pip install pygame
pip install imutils
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


## 🔍 Technical Details

The system uses:
- **Canny Edge Detection** for lane line identification
- **Hough Line Transform** for line detection
- **Region of Interest** masking for road area focus
- **Center calculation** for drift detection
- **Pygame** for audio alerts

## 📝 License

Open source project for educational and research purposes. 
