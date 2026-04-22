> ⚠️ **WARNING: This software is experimental and still buggy. Do NOT rely on OpenLKAS as a safety system. It is for educational and research purposes only. Always keep your eyes on the road and drive responsibly.**

## OpenLKAS is currently IN PROGRESS and in bugfixing.
- Ver 0.0: Initial upload of OpenLKAS
- Ver 0.1: Forward Collision Warning (FCW) system added — MobileNet-SSD vehicle detection with Time-to-Collision estimation
- Future Ver 0.2: Revamp of code with optimizations towards RPI4/5, & Nvidia Jetson Nano

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

## 🎮 Controls & Customization
While the display is active, you can use these keys to dynamically warp the lane detection bounds on the fly to perfectly map your dashboard and camera positioning:
- **`W`/`A`/`S`/`D`** or **Arrows**: Shift entire Focus Area Up/Left/Down/Right
- **`I`/`K`**: Move Top Edge (Horizon) Up/Down
- **`J`/`L`**: Expand/Shrink Top Width
- **`T`/`G`**: Move Bottom Edge Up/Down
- **`F`/`H`**: Expand/Shrink Bottom Width
- **`C`**: Auto-Calibrate Focus Area mathematically
- **`N`**: Toggle Forward Collision Warning (FCW) on/off
- **`V`**: Toggle audio volume

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

## 🚨 Forward Collision Warning (FCW)

OpenLKAS includes an optional Forward Collision Warning system that detects vehicles ahead and estimates Time-to-Collision (TTC).

### Setup
```bash
# Download the MobileNet-SSD model files (~23MB)
python download_models.py
```

### Usage
```bash
# Enable FCW alongside lane detection
python main.py --mode demo --enable-fcw

# With custom confidence threshold
python main.py --mode live --enable-fcw --fcw-confidence 0.6
```

### How It Works
- Uses **MobileNet-SSD v2** via OpenCV DNN for vehicle detection
- Runs on a **background thread** — does not slow down lane detection
- Calculates **Time-to-Collision (TTC)** using bounding box expansion rate
- Three alert tiers with a distinct **1200Hz tone** (vs 800Hz for lane departure):
  - 🟡 **CAUTION** (TTC ≤ 3.0s): Single beep every 1.0s
  - 🟠 **WARNING** (TTC ≤ 2.0s): Rapid beeping every 0.3s
  - 🔴 **DANGER** (TTC ≤ 1.0s): Continuous alarm + visual banner
- Press **`N`** at runtime to toggle FCW on/off

## License - MIT License

Open source project for educational and research purposes. 
