# Demo Videos Directory

This directory contains sample video files for testing the OpenLKAS system in demo mode.

## Supported Video Formats

The system supports the following video formats:
- MP4 (.mp4)
- AVI (.avi)
- MOV (.mov)
- MKV (.mkv)
- WMV (.wmv)

## Adding Demo Videos

1. Place your test video files in this directory
2. The system will automatically detect and use the first available video
3. You can specify a specific video using the `--video` argument

## Recommended Video Characteristics

For best lane detection results, use videos with:
- Clear lane markings
- Good lighting conditions
- Stable camera mounting
- Resolution of 720p or higher
- Duration of 30 seconds to 5 minutes

## Example Usage

```bash
# Use auto-detected demo video
python main.py --mode demo

# Use specific demo video
python main.py --mode demo --video demo_videos/my_test_video.mp4

# List available demo videos
python main.py --list-videos
```

## Sample Videos

You can download sample dashcam videos from:
- [Dashcam Owners Australia](https://dashcamownersaus.com.au/)
- [Open Source Dashcam Videos](https://github.com/udacity/self-driving-car)
- [YouTube dashcam channels](https://www.youtube.com/results?search_query=dashcam+driving)

**Note**: Make sure you have permission to use any videos you download. 