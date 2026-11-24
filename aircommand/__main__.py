"""Run the real-time hand tracking preview."""

import argparse
from collections import deque
from pathlib import Path
import sys
from time import monotonic_ns

import cv2

from .camera import CameraError, open_camera, read_frame
from .hand_tracker import create_tracker, detect_landmarks
from .overlay import draw_fps, draw_landmarks


MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "hand_landmarker.task"
WINDOW_NAME = "AirCommand"


def run_preview(camera_index: int) -> None:
    recent_frames = deque(maxlen=30)
    timestamp_ms = -1

    try:
        with create_tracker(MODEL_PATH) as tracker, open_camera(camera_index) as camera:
            while True:
                frame = read_frame(camera)
                # MediaPipe needs strictly increasing timestamps in milliseconds.
                timestamp_ms = max(monotonic_ns() // 1_000_000, timestamp_ms + 1)
                hands = detect_landmarks(tracker, frame, timestamp_ms)
                draw_landmarks(frame, hands)

                recent_frames.append(monotonic_ns())
                elapsed_ns = recent_frames[-1] - recent_frames[0]
                fps = (len(recent_frames) - 1) * 1e9 / elapsed_ns if elapsed_ns else 0.0
                draw_fps(frame, fps)

                cv2.imshow(WINDOW_NAME, frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q")):
                    break
    finally:
        cv2.destroyAllWindows()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-index", type=int, default=0, help="Webcam index (default: 0)")
    args = parser.parse_args()

    if not MODEL_PATH.is_file():
        print(f"Model missing at {MODEL_PATH}. See README.md for setup.", file=sys.stderr)
        return 1

    try:
        run_preview(args.camera_index)
    except KeyboardInterrupt:
        return 0
    except CameraError as exc:
        print(f"AirCommand: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
