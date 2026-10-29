"""Audio bridge self-test: proves speech crosses BlackHole in both directions through Chrome.

No Google account or meeting needed. Direction A: we speak through the speak device and a web
page records what Chrome receives as its microphone. Direction B: a web page plays speech and
we record it from the hear device. Both transcribe the audio with whisper and compare words.
"""

import asyncio
import http.server
import re
import tempfile
import threading
import wave
from pathlib import Path

import numpy as np

from senjin.audio.devices import Player, mic_frames
from senjin.audio.stt import WhisperServerSTT
from senjin.audio.tts import PiperTTS
from senjin.meet.audio import find_bridge
from senjin.meet.joiner import MeetBot

SENTENCE = "Tell me about the payments ledger you built and the trade-offs you made."

PAGE = """<!doctype html><meta charset=utf-8><title>selftest</title>
<audio id=a src="/sample.wav"></audio>
<script>
window.startRecording = async () => {
  const s = await navigator.mediaDevices.getUserMedia({audio: {echoCancellation: false,
    noiseSuppression: false, autoGainControl: false}});
  const ctx = new AudioContext({sampleRate: 16000});
  const src = ctx.createMediaStreamSource(s);
  const proc = ctx.createScriptProcessor(4096, 1, 1);
  window.__rec = [];
  proc.onaudioprocess = e => window.__rec.push(new Float32Array(e.inputBuffer.getChannelData(0)));
  src.connect(proc); proc.connect(ctx.destination);
};
window.getRecording = () => Array.from(window.__rec.flatMap(c => Array.from(c)));
window.playSample = () => document.getElementById('a').play();
</script>"""


def words(text: str) -> set[str]:
    return set(re.findall(r"[a-z']+", text.lower()))


def overlap(heard: str, expected: str = SENTENCE) -> float:
    want = words(expected)
    return len(words(heard) & want) / len(want)


def write_wav(path: Path, pcm: np.ndarray, rate: int) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.astype("<i2").tobytes())


def serve(directory: Path) -> tuple[http.server.ThreadingHTTPServer, str]:
    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(directory), **kw)

        def log_message(self, *a):
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_port}"


async def run_selftest(
    speak: str, hear: str, stt_model: str, log=print, threshold: float = 0.6
) -> bool:
    bridge = find_bridge(speak=speak, hear=hear)
    log(f"speak device: {bridge.speak_name}   hear device: {bridge.hear_name}")
    tts = PiperTTS()
    pcm = await tts.synth(SENTENCE)
    seconds = len(pcm) / tts.sample_rate
    stt = WhisperServerSTT(stt_model)
    await stt.start()
    bot = MeetBot(Path(tempfile.mkdtemp()), speak, hear, headless=False)
    tmp = Path(tempfile.mkdtemp())
    (tmp / "selftest.html").write_text(PAGE)
    write_wav(tmp / "sample.wav", pcm, tts.sample_rate)
    srv, base = serve(tmp)
    results: dict[str, float] = {}
    try:
        page = await bot.start()
        await page.goto(f"{base}/selftest.html")

        # A: bot speaks -> Chrome's microphone
        await page.evaluate("window.startRecording()")
        player = Player(tts.sample_rate, device=bridge.speak_index, channels=2)
        await asyncio.sleep(0.5)
        player.enqueue(pcm, "selftest")
        await player.wait_idle()
        await asyncio.sleep(0.5)
        player.close()
        got = np.array(await page.evaluate("window.getRecording()"), dtype=np.float32)
        heard = await stt.transcribe(got) if got.size else ""
        results["bot speaks"] = overlap(heard)
        log(f"A. Chrome's microphone heard: {heard!r}")

        # B: Chrome plays -> bot hears
        frames: list[np.ndarray] = []

        async def record() -> None:
            async for frame in mic_frames(bridge.hear_index, channels=2):
                frames.append(frame)

        recorder = asyncio.create_task(record())
        await asyncio.sleep(0.5)
        await page.evaluate("window.playSample()")
        await asyncio.sleep(seconds + 1.0)
        recorder.cancel()
        await asyncio.gather(recorder, return_exceptions=True)
        got = np.concatenate(frames) if frames else np.zeros(0, dtype=np.float32)
        heard = await stt.transcribe(got) if got.size else ""
        results["bot hears"] = overlap(heard)
        log(f"B. Recording from the hear device: {heard!r}")
    finally:
        srv.shutdown()
        stt.stop()
        await bot.close()
    for name, score in results.items():
        log(f"{'PASS' if score >= threshold else 'FAIL'}  {name}  (word overlap {score:.0%})")
    return all(score >= threshold for score in results.values())
