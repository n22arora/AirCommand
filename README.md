# AirCommand

AirCommand shows a live webcam preview with 21 landmarks per detected hand, gesture labels, and a rolling FPS counter. Camera frames are processed in memory on your Mac. The app does not record or upload them.

## Setup on macOS

From the repository root, use Python 3.11. The dependencies below were installed with Python 3.11.4 on Apple Silicon running macOS 14.5.

```sh
python3.11 -m venv .venv
.venv/bin/python -m pip install --only-binary=:all: -r requirements.txt
mkdir -p models
curl --fail --location --output models/hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

MediaPipe Tasks provides the on-device Hand Landmarker. It depends on `opencv-contrib-python`, the desktop OpenCV package used here for webcam capture, drawing, and the preview window. The model is a separate download from Google. Once setup is complete, the preview can run offline. [MediaPipe Hand Landmarker guide](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker?tab=Python)

## Run and check the preview

```sh
.venv/bin/python -m aircommand
```

If macOS prompts for Camera access, allow it for the app launching Python, such as Terminal. You can also check **System Settings → Privacy & Security → Camera**. If you have multiple cameras, use `--camera-index` to select an alternative to index `0`.

Hold one or both hands in view. Each detected hand should show 21 points connected at the fingers and palm. Move your hands out of view and check that the overlay clears while the FPS counter keeps updating. With the preview focused, press **Q** to close it and release the webcam.

Try these poses with your hand fully in view. Hold each pose briefly for its label to appear near your wrist.

- **Pinch** by touching your thumb tip to your index fingertip.
- **Open palm** by straightening all four fingers and extending your thumb away from your palm.
- **Fist** by curling all four fingers and tucking your thumb across them.

## How the frame loop works

- `camera.py` opens the webcam through AVFoundation and returns OpenCV BGR frames. Its context manager releases the camera when the loop exits or a frame read fails.
- `hand_tracker.py` converts BGR to RGB and passes each frame to MediaPipe. Video mode reuses tracking information across frames. It returns normalized landmark coordinates for up to two hands.
- `gestures.py` compares landmark distances as ratios, so the rules adapt as the hand changes size in the frame. Pinch compares the gap between the fingertips to palm width. For an open palm or a fist, it compares each finger's end-to-end reach with its length along the joints. Wrist position matches hands across frames, and a label changes after three consecutive frames agree.
- `overlay.py` maps normalized coordinates to pixels and draws the skeleton, gesture labels, and FPS in OpenCV.
- `__main__.py` runs the preview loop and handles keyboard input. Its FPS calculation uses the last 30 frame timestamps to measure the rate of the preview loop, including capture and inference.
