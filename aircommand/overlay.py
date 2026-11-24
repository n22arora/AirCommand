"""Draw hand landmarks, gesture labels, and FPS with OpenCV."""

import cv2
from mediapipe import tasks
from mediapipe.tasks.python.components.containers import NormalizedLandmark


FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.8


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
    _draw_text(frame, f"FPS: {fps:.1f}", (16, 32))


def draw_gesture(frame: cv2.typing.MatLike, wrist: NormalizedLandmark, label: str) -> None:
    height, width = frame.shape[:2]
    (text_width, _), baseline = cv2.getTextSize(label, FONT, FONT_SCALE, 4)
    x = max(6, min(round(wrist.x * (width - 1)), width - text_width - 6))
    y = min(max(56, round(wrist.y * (height - 1)) + 24), height - baseline - 6)
    _draw_text(frame, label, (x, y))


def _draw_text(frame: cv2.typing.MatLike, label: str, origin: tuple[int, int]) -> None:
    cv2.putText(frame, label, origin, FONT, FONT_SCALE, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, label, origin, FONT, FONT_SCALE, (255, 255, 255), 1, cv2.LINE_AA)
