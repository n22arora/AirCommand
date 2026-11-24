import subprocess
import unittest
from unittest.mock import patch

from aircommand.controls import MuteError, PinchTrigger, toggle_output_mute
from aircommand.gestures import Gesture


class PinchTriggerTests(unittest.TestCase):
    def test_pinch_requires_an_open_palm_and_fires_once_per_cycle(self):
        trigger = PinchTrigger()
        frames = (
            (Gesture.PINCH, False),
            (Gesture.OPEN_PALM, False),
            (None, False),
            (Gesture.PINCH, True),
            (Gesture.PINCH, False),
            (None, False),
            (Gesture.PINCH, False),
            (Gesture.OPEN_PALM, False),
            (Gesture.PINCH, True),
        )
        self.assertEqual([trigger.update([label]) for label, _ in frames],
                         [expected for _, expected in frames])

    def test_tracking_loss_and_multiple_hands_disarm_the_trigger(self):
        for labels in ([], [Gesture.PINCH, Gesture.OPEN_PALM]):
            with self.subTest(labels=labels):
                trigger = PinchTrigger()
                self.assertFalse(trigger.update([Gesture.OPEN_PALM]))
                self.assertFalse(trigger.update(labels))
                self.assertFalse(trigger.update([Gesture.PINCH]))


class MuteActionTests(unittest.TestCase):
    @patch("aircommand.controls.subprocess.run")
    def test_returns_actual_mute_state(self, run):
        for reply, expected in (("true\n", True), ("false\n", False)):
            with self.subTest(reply=reply):
                run.return_value.stdout = reply
                self.assertIs(toggle_output_mute(), expected)
        self.assertEqual(run.call_args.args[0][:2], ["/usr/bin/osascript", "-e"])
        self.assertTrue(run.call_args.kwargs["check"])

    @patch("aircommand.controls.subprocess.run")
    def test_surfaces_os_error_and_timeout(self, run):
        run.side_effect = subprocess.CalledProcessError(1, "osascript", stderr="Device error\n")
        with self.assertRaisesRegex(MuteError, "Device error"):
            toggle_output_mute()

        run.side_effect = subprocess.TimeoutExpired("osascript", 1)
        with self.assertRaisesRegex(MuteError, "did not respond"):
            toggle_output_mute()
