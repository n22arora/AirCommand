"""Classify hand poses and stabilize their labels across preview frames."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from math import dist, hypot
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mediapipe.tasks.python.components.containers import NormalizedLandmark


class Gesture(StrEnum):
    PINCH = "Pinch"
    OPEN_PALM = "Open palm"
    FIST = "Fist"


Joint = tuple[float, float, float]
THUMB_TIP = 4
INDEX_BASE = 5
INDEX_TIP = 8
PINKY_BASE = 17
FINGER_BASES = (INDEX_BASE, 9, 13, PINKY_BASE)
PINCH_MAX_GAP = 0.40
PINCH_DEPTH_WEIGHT = 0.5
FIST_MAX_THUMB_REACH = 0.65
FIST_MAX_FINGER_REACH = 0.65
OPEN_MIN_THUMB_REACH = 0.50
OPEN_MIN_FINGER_REACH = 0.88
STABLE_FRAMES = 3


def classify_gesture(joints: Sequence[Joint]) -> Gesture | None:
    """Classify 21 landmarks in a coordinate system with one scale for all axes."""
    if len(joints) != 21:
        return None

    palm_width = dist(joints[INDEX_BASE], joints[PINKY_BASE])
    if palm_width == 0:
        return None

    thumb_reach = dist(joints[THUMB_TIP], joints[INDEX_BASE]) / palm_width
    finger_reaches = []
    for base in FINGER_BASES:
        finger_length = sum(dist(joints[i], joints[i + 1]) for i in range(base, base + 3))
        if finger_length == 0:
            return None
        finger_reaches.append(dist(joints[base], joints[base + 3]) / finger_length)

    # A tucked thumb can touch the index fingertip in a fist, so check this first.
    if thumb_reach <= FIST_MAX_THUMB_REACH and max(finger_reaches) <= FIST_MAX_FINGER_REACH:
        return Gesture.FIST

    thumb_x, thumb_y, thumb_z = joints[THUMB_TIP]
    index_x, index_y, index_z = joints[INDEX_TIP]
    # The model can estimate different depths for fingertips that visibly touch.
    fingertip_gap = hypot(
        thumb_x - index_x, thumb_y - index_y, (thumb_z - index_z) * PINCH_DEPTH_WEIGHT
    )
    if fingertip_gap <= PINCH_MAX_GAP * palm_width:
        return Gesture.PINCH

    if thumb_reach >= OPEN_MIN_THUMB_REACH and min(finger_reaches) >= OPEN_MIN_FINGER_REACH:
        return Gesture.OPEN_PALM

    return None


@dataclass
class _HandState:
    wrist: tuple[float, float]
    candidate: Gesture | None = None
    consecutive_frames: int = 0
    displayed: Gesture | None = None

    def observe(self, gesture: Gesture | None) -> None:
        if gesture == self.candidate:
            self.consecutive_frames += 1
        else:
            self.candidate = gesture
            self.consecutive_frames = 1

        if self.consecutive_frames == STABLE_FRAMES:
            self.displayed = self.candidate


class GestureStabilizer:
    """Return one stable label per hand, in the current frame's detection order."""

    def __init__(self) -> None:
        self._hands: list[_HandState] = []

    def update(
        self, hands: Sequence[Sequence[NormalizedLandmark]], width: int, height: int
    ) -> list[Gesture | None]:
        # MediaPipe scales x and z by image width, and y by image height.
        poses = [
            [(point.x * width, point.y * height, point.z * width) for point in hand]
            for hand in hands
        ]
        wrists = [(pose[0][0], pose[0][1]) for pose in poses]

        # Match nearby wrists so a change in MediaPipe's output order preserves label history.
        max_movement = 0.15 * min(width, height)
        pairs = sorted(
            (dist(wrist, old_hand.wrist), index, old_index)
            for index, wrist in enumerate(wrists)
            for old_index, old_hand in enumerate(self._hands)
        )
        matches: dict[int, _HandState] = {}
        used_hands: set[int] = set()
        for distance, index, old_index in pairs:
            if distance > max_movement:
                break
            if index not in matches and old_index not in used_hands:
                matches[index] = self._hands[old_index]
                used_hands.add(old_index)

        current_hands = []
        for index, (pose, wrist) in enumerate(zip(poses, wrists)):
            state = matches.get(index)
            if state is None:
                state = _HandState(wrist)
            state.wrist = wrist
            state.observe(classify_gesture(pose))
            current_hands.append(state)

        self._hands = current_hands
        return [hand.displayed for hand in current_hands]
