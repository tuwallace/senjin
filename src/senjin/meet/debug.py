"""Join a call and report whether call audio reaches the hear device.

Every few seconds prints the level (RMS) recorded from the hear device and what the page's
media elements are doing, so you can tell "nobody spoke" from "audio is not routed".
Speak or play audio in the meeting while it runs.
"""

import asyncio

import numpy as np

from senjin.audio.devices import mic_frames
from senjin.meet.audio import find_bridge
from senjin.meet.joiner import MeetBot

INSPECT_JS = "() => window.__senjinInspect ? window.__senjinInspect() : null"


def rms(frames: list[np.ndarray]) -> float:
    if not frames:
        return 0.0
    return float(np.sqrt(np.mean(np.concatenate(frames) ** 2)))


async def run_debug(url: str, speak: str, hear: str, seconds: float, log=print) -> None:
    bridge = find_bridge(speak=speak, hear=hear)
    bot = MeetBot(mic_name=bridge.speak_name, speaker_name=bridge.hear_name)
    frames: list[np.ndarray] = []

    async def record() -> None:
        async for frame in mic_frames(bridge.hear_index, channels=2):
            frames.append(frame)

    try:
        await bot.start()
        await bot.join(url, 300, log)
        log(f"In the call. Speak in the meeting now; sampling for {seconds:.0f}s...")
        recorder = asyncio.create_task(record())
        for _ in range(int(seconds // 5)):
            await asyncio.sleep(5)
            level, frames[:] = rms(frames), []
            info = await bot.page.evaluate(INSPECT_JS)
            log(f"hear-device level {level:.4f}   people={await bot.participant_count()}")
            log(f"  chosen devices: {info['ids']}")
            if info is None:
                log("  routing script not loaded")
                continue
            for el in info["elements"]:
                log(f"  element {el}")
            for ctx in info["contexts"]:
                log(f"  audio context {ctx}")
            if not info["elements"] and not info["contexts"]:
                log("  no media elements or audio contexts seen")
        recorder.cancel()
        await asyncio.gather(recorder, return_exceptions=True)
    finally:
        await bot.leave()
        await bot.close()
