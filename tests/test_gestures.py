import unittest
from types import SimpleNamespace

from aircommand.gestures import Gesture, GestureStabilizer, classify_gesture


def open_palm():
    return [(0, 0, 0), (-7, -1, 0), (-9, -2, 0), (-12, -3, 0), (-14, -4, 0)] + [
        (x, y, 0) for x in (-6, -2, 2, 6) for y in (-3, -6, -8, -10)
    ]


def fist():
    joints = open_palm()
    joints[5:] = [
        (x, y, z) for x in (-6, -2, 2, 6) for y, z in ((-3, 0), (-5, 0), (-5, 2), (-3, 2))
    ]
    joints[4] = joints[8]
    return joints


def pinch():
    joints = open_palm()
    joints[4] = joints[8]
    return joints


def camera_landmarks(joints, wrist_x):
    return [
        SimpleNamespace(x=(x + wrist_x) / 640, y=(y + 240) / 480, z=z / 640)
        for x, y, z in joints
    ]


class GestureTests(unittest.TestCase):
    def test_open_palm_fist_and_pinch(self):
        self.assertEqual(classify_gesture(open_palm()), Gesture.OPEN_PALM)
        self.assertEqual(classify_gesture(fist()), Gesture.FIST)
        self.assertEqual(classify_gesture(pinch()), Gesture.PINCH)

    def test_pinch_adapts_to_hand_size_and_uses_depth(self):
        joints = pinch()
        larger_hand = [(3 * x, 3 * y, 3 * z) for x, y, z in joints]
        self.assertEqual(classify_gesture(larger_hand), Gesture.PINCH)
        x, y, z = joints[4]
        joints[4] = (x - 3.6, y, z + 5)
        self.assertEqual(classify_gesture(joints), Gesture.PINCH)
        joints[4] = (x, y, z + 12)
        self.assertNotEqual(classify_gesture(joints), Gesture.PINCH)

    def test_extended_thumb_with_curled_fingers_is_unclassified(self):
        joints = fist()
        joints[4] = open_palm()[4]
        self.assertIsNone(classify_gesture(joints))

    def test_labels_require_three_consecutive_frames_and_reset_on_hand_loss(self):
        stabilizer = GestureStabilizer()
        palm = camera_landmarks(open_palm(), 120)
        pinched = camera_landmarks(pinch(), 120)
        transition = (
            (palm, None),
            (palm, None),
            (palm, Gesture.OPEN_PALM),
            (pinched, Gesture.OPEN_PALM),
            (palm, Gesture.OPEN_PALM),
            (pinched, Gesture.OPEN_PALM),
            (pinched, Gesture.OPEN_PALM),
            (pinched, Gesture.PINCH),
        )
        for hand, expected in transition:
            self.assertEqual(stabilizer.update([hand], 640, 480), [expected])

        self.assertEqual(stabilizer.update([], 640, 480), [])
        self.assertEqual(stabilizer.update([pinched], 640, 480), [None])

    def test_two_labels_stay_with_their_hands_after_detection_order_changes(self):
        stabilizer = GestureStabilizer()
        palm = camera_landmarks(open_palm(), 120)
        closed = camera_landmarks(fist(), 480)
        for _ in range(3):
            stabilizer.update([palm, closed], 640, 480)

        self.assertEqual(stabilizer.update([closed, palm], 640, 480), [Gesture.FIST, Gesture.OPEN_PALM])
