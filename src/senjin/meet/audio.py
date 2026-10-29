"""Finding the virtual audio devices that bridge Meet and the voice loop (BlackHole)."""

from dataclasses import dataclass

SPEAK_DEVICE = "BlackHole 2ch"  # we play TTS here; the browser reads it as its microphone
HEAR_DEVICE = "BlackHole 16ch"  # the browser plays the call here; we record from it

INSTALL_HINT = (
    "Install the virtual audio drivers (needs your password, then log out/in or reboot "
    "if macOS does not list them):\n"
    "  brew install --cask blackhole-2ch blackhole-16ch"
)


class DeviceError(RuntimeError):
    pass


@dataclass
class Bridge:
    speak_index: int
    hear_index: int
    speak_name: str
    hear_name: str


def find_device(devices: list[dict], name: str, *, need_input: bool, need_output: bool) -> dict:
    for index, dev in enumerate(devices):
        if name.lower() not in dev["name"].lower():
            continue
        if need_input and dev["max_input_channels"] < 1:
            continue
        if need_output and dev["max_output_channels"] < 1:
            continue
        return {**dev, "index": index}
    raise DeviceError(f"audio device '{name}' not found.\n{INSTALL_HINT}")


def find_bridge(
    devices: list[dict] | None = None, speak: str = SPEAK_DEVICE, hear: str = HEAR_DEVICE
) -> Bridge:
    if devices is None:
        import sounddevice as sd

        devices = [dict(d) for d in sd.query_devices()]
    out = find_device(devices, speak, need_input=False, need_output=True)
    inp = find_device(devices, hear, need_input=True, need_output=False)
    return Bridge(out["index"], inp["index"], out["name"], inp["name"])
