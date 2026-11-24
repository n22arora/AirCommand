"""Translate a deliberate gesture into a macOS audio action."""

from collections.abc import Sequence
import subprocess

from .gestures import Gesture


_TOGGLE_MUTE_SCRIPT = """
set shouldMute to not (output muted of (get volume settings))
set volume output muted shouldMute
set isMuted to output muted of (get volume settings)
if isMuted is not shouldMute then error "macOS did not change the output mute state."
return isMuted
"""


class PinchTrigger:
    """Arm on an open palm and fire once on the next pinch."""

    def __init__(self) -> None:
        self._armed = False

    def update(self, labels: Sequence[Gesture | None]) -> bool:
        if len(labels) != 1:
            self._armed = False
            return False

        gesture = labels[0]
        if gesture == Gesture.OPEN_PALM:
            self._armed = True
        elif gesture == Gesture.PINCH and self._armed:
            self._armed = False
            return True
        return False


class MuteError(RuntimeError):
    """The audio action failed or its result could not be confirmed."""


def toggle_output_mute() -> bool:
    """Toggle macOS output mute and return the resulting mute state."""
    try:
        result = subprocess.run(
            ["/usr/bin/osascript", "-e", _TOGGLE_MUTE_SCRIPT],
            check=True,
            capture_output=True,
            text=True,
            timeout=1,
        )
    except subprocess.CalledProcessError as exc:
        raise MuteError(exc.stderr.strip() or "AppleScript failed.") from exc
    except subprocess.TimeoutExpired as exc:
        raise MuteError("AppleScript did not respond within one second.") from exc
    except OSError as exc:
        raise MuteError(str(exc)) from exc

    mute_state = result.stdout.strip()
    if mute_state not in ("true", "false"):
        raise MuteError("AppleScript returned an unexpected mute state.")
    return mute_state == "true"
