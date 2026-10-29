"""Microphone input and interruptible speaker output (sounddevice / Core Audio)."""

import asyncio
import threading
from collections import deque
from collections.abc import AsyncIterator

import numpy as np

from senjin.audio.vad import FRAME, RATE


async def mic_frames(
    device: int | str | None = None, channels: int = 1
) -> AsyncIterator[np.ndarray]:
    """Yield 32 ms mono float32 frames at 16 kHz; multi-channel devices are averaged."""
    import sounddevice as sd

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[np.ndarray] = asyncio.Queue(maxsize=200)

    def callback(indata, frames, time_info, status) -> None:
        frame = indata[:, 0].copy() if channels == 1 else indata.mean(axis=1)
        loop.call_soon_threadsafe(_put, frame)

    def _put(frame: np.ndarray) -> None:
        if queue.full():
            queue.get_nowait()  # drop the oldest rather than fall behind real time
        queue.put_nowait(frame)

    with sd.InputStream(
        samplerate=RATE, channels=channels, dtype="float32", blocksize=FRAME, device=device,
        callback=callback,
    ):  # fmt: skip
        while True:
            yield await queue.get()


class Player:
    """Plays queued clips; stop() cuts audio immediately. Tracks which clips began playing."""

    def __init__(
        self, sample_rate: int, device: int | str | None = None, channels: int = 1
    ) -> None:
        import sounddevice as sd

        self.sample_rate = sample_rate
        self._channels = channels
        self._clips: deque[tuple[str, np.ndarray]] = deque()
        self._current: tuple[str, np.ndarray, int] | None = None
        self._lock = threading.Lock()
        self.started: list[str] = []
        self.first_sample_at: float | None = None
        self._stream = sd.OutputStream(
            samplerate=sample_rate, channels=channels, dtype="int16", device=device,
            callback=self._callback,
        )  # fmt: skip
        self._stream.start()

    def enqueue(self, audio: np.ndarray, tag: str) -> None:
        with self._lock:
            self._clips.append((tag, audio))

    @property
    def busy(self) -> bool:
        with self._lock:
            return bool(self._clips) or self._current is not None

    def stop(self) -> list[str]:
        """Drop everything; return tags of clips that had started playing."""
        with self._lock:
            self._clips.clear()
            self._current = None
            return list(self.started)

    def reset(self) -> None:
        with self._lock:
            self.started = []
            self.first_sample_at = None

    async def wait_idle(self) -> None:
        while self.busy:
            await asyncio.sleep(0.02)

    def close(self) -> None:
        self._stream.stop()
        self._stream.close()

    def _callback(self, outdata, frames, time_info, status) -> None:
        import time

        out = np.zeros(frames, dtype=np.int16)
        filled = 0
        with self._lock:
            while filled < frames:
                if self._current is None:
                    if not self._clips:
                        break
                    tag, audio = self._clips.popleft()
                    self._current = (tag, audio, 0)
                    self.started.append(tag)
                    if self.first_sample_at is None:
                        self.first_sample_at = time.monotonic()
                tag, audio, pos = self._current
                n = min(frames - filled, len(audio) - pos)
                out[filled : filled + n] = audio[pos : pos + n]
                filled += n
                pos += n
                self._current = None if pos >= len(audio) else (tag, audio, pos)
        outdata[:] = out[:, None]  # same signal on every channel


class SilentPlayer:
    """Same interface as Player but plays nothing; clips 'play' for their real duration.

    Lets the whole loop run (and be timed) without a speaker, e.g. in `senjin-voice simulate`.
    """

    def __init__(self, sample_rate: int) -> None:
        import time

        self._time = time.monotonic
        self.sample_rate = sample_rate
        self.clips: list[tuple[str, float, float]] = []  # tag, start, end
        self.first_sample_at: float | None = None

    def enqueue(self, audio: np.ndarray, tag: str) -> None:
        start = max(self._time(), self.clips[-1][2] if self.clips else 0.0)
        self.clips.append((tag, start, start + len(audio) / self.sample_rate))
        if self.first_sample_at is None:
            self.first_sample_at = start

    @property
    def busy(self) -> bool:
        return bool(self.clips) and self._time() < self.clips[-1][2]

    def stop(self) -> list[str]:
        started = [tag for tag, start, _ in self.clips if start <= self._time()]
        self.clips = []
        return started

    def reset(self) -> None:
        self.clips = []
        self.first_sample_at = None

    async def wait_idle(self) -> None:
        while self.busy:
            await asyncio.sleep(0.02)

    def close(self) -> None:
        pass
