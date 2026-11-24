"""Draw hand landmarks and preview performance with OpenCV."""

import cv2
from mediapipe import tasks
from mediapipe.tasks.python.components.containers import NormalizedLandmark


def draw_landmarks(
    frame: cv2.typing.MatLike, hands: list[list[NormalizedLandmark]]
) -> None:
    height, width = frame.shape[:2]
    for hand in hands:
        points = [
            (round(point.x * (width - 1)), round(point.y * (height - 1)))
            for point in hand
        ]
        for connection in tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS:
            start, end = points[connection.start], points[connection.end]
            cv2.line(frame, start, end, (0, 200, 110), 2, cv2.LINE_AA)
        for point in points:
            cv2.circle(frame, point, 4, (0, 230, 255), -1, cv2.LINE_AA)


def draw_fps(frame: cv2.typing.MatLike, fps: float) -> None:
    label = f"FPS: {fps:.1f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, label, (16, 32), font, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, label, (16, 32), font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
