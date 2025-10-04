## Changelogs - IN PROGRESS 
- Ver 0.0: Initial upload of OpenLKAS
- 

##  Software Dependencies

```bash
pip install opencv-python
pip install numpy
pip install pygame
pip install imutils
```


## Usage

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

##  Config

Key parameters can be adjusted in the respective modules:

- **Lane departure threshold**: Adjust sensitivity in `lane_detector.py`
- **Camera resolution**: Modify in `camera_module.py`
- **Audio alert volume**: Configure in `audio_alert.py`

## Hardware Requirements

I have tested with a Raspberry Pi 4 with 4GB of RAM, and OpenLKAS runs smoothly. However, OpenLKAS can run on any device with the software dependency prerequisites - as long as said device has above 2GB of RAM, and a processor around the power of an ARM Cortex A72. However, results may vary. Also, a camera must be connected in order for live mode to run, preferably of a resolution of 720p and above.
Furthermore, I have also tested with a MacBook (3.4 gHz i7) running macOS Ventura utilizing the "iPhone as Camera" feature - where I keep the laptop running OpenLKAS, and the phone is mounted to the dash pointing the rear camera towards the road. The higher the resolution camera and higher CPU processing power you have, the better.
I may test with an Android device as well.


## My RPI testing setup:
- Raspberry Pi 4B with 4GB RAM, ARM Cortex A72 - along with an active cooler
- Logitech 720p Webcam
- Dashboard mounted setup with the webcam directly in the middle of the car's dashboard, pointing towards the road
## My MacBook testing setup:
- 2015 Macbook Pro with an i7-5557u and 16GB RAM
- Dashboard mounted iPhone via macOS Ventura and later's iPhone as Camera feature pointing towards the road


## Testing

1. Start with demo mode using sample videos in `demo_videos/`
2. Validate lane detection accuracy
3. Switch to live mode for real-time testing
4. Verify audio alerts trigger correctly
5. Please act accordingly while testing - do NOT rely on OpenLKAS as a complete safety system. Only test on empty roads and be cautious while driving.


## Technical Details

The system uses:
- **Canny Edge Detection** for lane line identification
- **Hough Line Transform** for line detection
- **Region of Interest** masking for road area focus
- **Center calculation** for drift detection
- **Pygame** for audio alerts

## License - MIT License

Open source project for educational and research purposes. 
