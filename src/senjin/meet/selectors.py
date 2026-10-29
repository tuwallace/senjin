"""Meet UI labels, kept in one place because Google changes them without notice.

These come from the Meet web UI as of this writing and have NOT been checked against a live
meeting from this codebase; adjust here first when joining breaks.
"""

import re

I = re.IGNORECASE  # noqa: E741

JOIN_BUTTON = re.compile(r"^(join now|ask to join|switch here|join anyway)", I)
LEAVE_BUTTON = re.compile(r"^leave call", I)
CAMERA_ON_BUTTON = re.compile(r"^turn off camera", I)  # present while the camera is on
MIC_OFF_BUTTON = re.compile(r"^turn on microphone", I)  # present while the mic is muted
DISMISS_BUTTONS = re.compile(r"^(got it|dismiss|continue without|no thanks)", I)
NAME_FIELD = re.compile(r"your name", I)
PEOPLE_BUTTON = re.compile(r"^(show everyone|people)", I)

DENIED_TEXT = re.compile(
    r"(denied your request|you can.t join this (video )?call|no one responded)", I
)
ENDED_TEXT = re.compile(r"(you left the meeting|the meeting has ended|removed from the meeting)", I)
