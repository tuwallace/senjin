"""Text to speech (Piper, with macOS `say` as a fallback) and sentence splitting for streaming."""

import asyncio
import re
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import Protocol

import numpy as np

from senjin.audio.models import PIPER_PATH
from senjin.config import DEFAULT_VOICE_PACE


class TTS(Protocol):
    sample_rate: int

    async def synth(self, text: str) -> np.ndarray:
        """Mono int16 audio at `sample_rate`."""
        ...


class PiperTTS:
    def __init__(self, model_path: Path = PIPER_PATH, pace: float = DEFAULT_VOICE_PACE) -> None:
        from piper import PiperVoice, SynthesisConfig

        self._config = SynthesisConfig(length_scale=pace)  # above 1.0 is slower

        self._voice = PiperVoice.load(str(model_path))
        self.sample_rate = self._voice.config.sample_rate

    def _synth(self, text: str) -> np.ndarray:
        chunks = [
            c.audio_int16_array for c in self._voice.synthesize(text, syn_config=self._config)
        ]
        return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.int16)

    async def synth(self, text: str) -> np.ndarray:
        return await asyncio.to_thread(self._synth, text)


class SayTTS:
    """Zero-install fallback using the macOS voice."""

    sample_rate = 22050

    def _synth(self, text: str) -> np.ndarray:
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "out.wav"
            subprocess.run(["say", "-o", str(path), "--data-format=LEI16@22050", text], check=True)
            with wave.open(str(path)) as w:
                return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")

    async def synth(self, text: str) -> np.ndarray:
        return await asyncio.to_thread(self._synth, text)


_ABBREV = ("e.g.", "i.e.", "vs.", "etc.", "mr.", "mrs.", "dr.", "st.")
_END = re.compile(r"([.!?]+[\"')\]]*)\s+")


class SentenceSplitter:
    """Incrementally cut streamed text into sentences so TTS can start before the LLM finishes."""

    def __init__(self, min_chars: int = 12) -> None:
        self._buf = ""
        self._min = min_chars

    def feed(self, chunk: str) -> list[str]:
        self._buf += chunk
        out: list[str] = []
        start = 0
        for m in _END.finditer(self._buf):
            sentence = self._buf[start : m.end(1)].strip()
            if sentence.lower().endswith(_ABBREV) or len(sentence) < self._min:
                continue
            out.append(sentence)
            start = m.end()
        self._buf = self._buf[start:]
        return out

    def flush(self) -> str | None:
        rest, self._buf = self._buf.strip(), ""
        return rest or None
