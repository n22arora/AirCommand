"""Run MediaPipe inference on frames independently of the camera and UI."""

from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks.python.components.containers import NormalizedLandmark


def create_tracker(model_path: Path) -> mp.tasks.vision.HandLandmarker:
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(model_path), delegate=mp.tasks.BaseOptions.Delegate.CPU
        ),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=2,
    )
    return mp.tasks.vision.HandLandmarker.create_from_options(options)


def detect_landmarks(
    tracker: mp.tasks.vision.HandLandmarker,
    frame: cv2.typing.MatLike,
    timestamp_ms: int,
) -> list[list[NormalizedLandmark]]:
    """Return the normalized landmarks for each hand in a BGR frame."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    return tracker.detect_for_video(image, timestamp_ms).hand_landmarks
