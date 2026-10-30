"""Speech to text through a local whisper.cpp server (Metal-accelerated, model stays loaded)."""

import asyncio
import io
import re
import subprocess
import time
import wave
from pathlib import Path
from typing import Protocol

import httpx
import numpy as np

from senjin.audio.models import WHISPER_DIR
from senjin.audio.vad import RATE

DEFAULT_WHISPER_MODEL = "ggml-small.bin"
_NOISE = re.compile(r"\[[^\]]*\]|\([^)]*\)|\*[^*]*\*")  # [BLANK_AUDIO], (music), *cough*
# Whisper tends to invent these on near-silence.
_HALLUCINATIONS = {"you", "thank you.", "thanks for watching!", "bye.", "."}


class STT(Protocol):
    async def transcribe(self, audio: np.ndarray) -> str: ...


def clean_transcript(text: str) -> str:
    text = _NOISE.sub(" ", text)
    text = re.sub(r"(^|\s)(>>+|-{1,2})(?=\s|$)", " ", text)  # speaker-change marks
    text = re.sub(r"\s+", " ", text).strip()
    return "" if text.lower() in _HALLUCINATIONS else text


def to_wav_bytes(audio: np.ndarray, rate: int = RATE) -> bytes:
    pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


class WhisperServerSTT:
    def __init__(self, model: str = DEFAULT_WHISPER_MODEL, port: int = 8178) -> None:
        self.model_path = Path(model) if "/" in model else WHISPER_DIR / model
        self.url = f"http://127.0.0.1:{port}"
        self.port = port
        self._proc: subprocess.Popen | None = None
        self.prompt = ""  # names and terms to expect (from the resume), so they are spelled right

    async def start(self, timeout: float = 60.0) -> None:
        if self._proc or await self._alive():
            return
        if not self.model_path.exists():
            raise FileNotFoundError(f"whisper model not found: {self.model_path}")
        self._proc = subprocess.Popen(
            ["whisper-server", "-m", str(self.model_path), "--port", str(self.port), "-nt"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if await self._alive():
                return
            if self._proc.poll() is not None:
                raise RuntimeError("whisper-server exited during startup")
            await asyncio.sleep(0.3)
        raise TimeoutError("whisper-server did not become ready")

    async def _alive(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=1) as c:
                return (await c.get(self.url + "/")).status_code < 500
        except httpx.HTTPError:
            return False

    def stop(self) -> None:
        if self._proc:
            self._proc.terminate()
            self._proc = None

    async def transcribe(self, audio: np.ndarray) -> str:
        # Whisper tends to drop the first word of audio that starts abruptly: lead in with silence.
        padded = np.concatenate([np.zeros(RATE * 3 // 10, dtype=audio.dtype), audio])
        files = {"file": ("audio.wav", to_wav_bytes(padded), "audio/wav")}
        data = {"temperature": "0.0", "response_format": "json", "language": "en"}
        if self.prompt:
            data["prompt"] = self.prompt
        async with httpx.AsyncClient(timeout=60) as c:
            r = await c.post(self.url + "/inference", files=files, data=data)
        r.raise_for_status()
        text = clean_transcript(r.json().get("text", ""))
        if self.prompt and text.lower().startswith(self.prompt.lower()[:24]):
            return ""  # whisper echoed its hint on near-silence
        return text
