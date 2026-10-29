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

INSPECT_JS = """() => {
  const els = [];
  const walk = root => {
    root.querySelectorAll('audio, video').forEach(e => els.push({
      tag: e.tagName, sinkId: e.sinkId, paused: e.paused, muted: e.muted,
      volume: e.volume, audioTracks: e.srcObject ? e.srcObject.getAudioTracks().length : 0,
      inDom: e.isConnected,
    }));
    root.querySelectorAll('*').forEach(n => { if (n.shadowRoot) walk(n.shadowRoot); });
  };
  walk(document);
  return {ids: window.__senjin && window.__senjin.ids, elements: els};
}"""


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
            for el in info["elements"]:
                log(f"  {el}")
            if not info["elements"]:
                log("  no <audio>/<video> elements found")
        recorder.cancel()
        await asyncio.gather(recorder, return_exceptions=True)
    finally:
        await bot.leave()
        await bot.close()
