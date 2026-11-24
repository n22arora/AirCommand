"""OpenCV webcam capture on macOS."""

from collections.abc import Iterator
from contextlib import contextmanager

import cv2


class CameraError(RuntimeError):
    """The camera could not be opened or could not supply a frame."""


@contextmanager
def open_camera(index: int = 0) -> Iterator[cv2.VideoCapture]:
    """Open the selected camera and release it when the preview ends."""
    camera = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
    try:
        if not camera.isOpened():
            raise CameraError(
                f"Could not open camera {index}. Check macOS Camera permissions "
                "for the app launching Python and check the camera index."
            )
        yield camera
    finally:
        camera.release()


def read_frame(camera: cv2.VideoCapture) -> cv2.typing.MatLike:
    success, frame = camera.read()
    if not success or frame is None:
        raise CameraError("The camera stopped supplying frames.")
    return frame
