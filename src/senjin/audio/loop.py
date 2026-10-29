"""The voice loop: mic -> VAD -> STT -> interviewer -> sentence-streamed TTS -> speaker.

Candidate speech during the interviewer's reply (barge-in) stops playback at once; the
interviewer's history keeps only the words that were actually spoken.
"""

import asyncio
import json
import time
from collections.abc import AsyncIterator, Callable
from contextlib import aclosing
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Protocol

import numpy as np

from senjin.audio.stt import STT
from senjin.audio.tts import TTS, SentenceSplitter
from senjin.audio.vad import TurnConfig, TurnDetector
from senjin.interview.session import InterviewSession


class AudioOut(Protocol):
    busy: bool

    def enqueue(self, audio: np.ndarray, tag: str) -> None: ...
    def stop(self) -> list[str]: ...
    def reset(self) -> None: ...
    async def wait_idle(self) -> None: ...


@dataclass
class VoiceConfig:
    barge_in: bool = True
    barge_in_ms: int = 300  # candidate speech needed to cut the interviewer off


class VoiceLoop:
    def __init__(
        self,
        session: InterviewSession,
        vad: Callable[[np.ndarray], float],
        stt: STT,
        tts: TTS,
        player: AudioOut,
        config: VoiceConfig | None = None,
        turn_config: TurnConfig | None = None,
        clock: Callable[[], float] = time.monotonic,
        latency_path: Path | None = None,
        on_text: Callable[[str, str], None] | None = None,
    ) -> None:
        self.session, self.vad, self.stt, self.tts, self.player = session, vad, stt, tts, player
        self.config = config or VoiceConfig()
        self.turn_config = turn_config or TurnConfig()
        self.detector = TurnDetector(self.turn_config)
        self.clock = clock
        self.latency_path = latency_path
        self.on_text = on_text or (lambda role, text: None)
        self.records: list[dict] = []
        self.errors: list[BaseException] = []
        self.interruptions = 0
        self._turn: asyncio.Task | None = None
        self._stt_audio: np.ndarray | None = None
        self._carry: np.ndarray | None = None

    @property
    def busy(self) -> bool:
        return self._turn is not None and not self._turn.done()

    async def run(self, frames: AsyncIterator[np.ndarray]) -> None:
        self._start_turn(self._speak(self.session.open(), {"turn": "open"}, None))
        try:
            async for frame in frames:
                if self.session.finished and not self.busy:
                    break
                prob = self.vad(frame)
                if self.busy and not self.config.barge_in:
                    self.detector.reset()
                    continue
                events = self.detector.feed(frame, prob)
                if (
                    self.busy
                    and self.detector.in_speech
                    and self.detector.speech_ms >= self.config.barge_in_ms
                ):
                    await self._interrupt()
                for event in events:
                    if event.kind == "end":
                        await self._on_utterance(event.audio)
        finally:
            await self._cancel_turn()
            self.player.stop()

    def _start_turn(self, coro) -> None:
        self._turn = asyncio.create_task(coro)
        self._turn.add_done_callback(self._collect_error)

    def _collect_error(self, task: asyncio.Task) -> None:
        if not task.cancelled() and task.exception():
            self.errors.append(task.exception())

    async def _cancel_turn(self) -> None:
        if self._turn and not self._turn.done():
            self._turn.cancel()
            await asyncio.gather(self._turn, return_exceptions=True)

    async def _interrupt(self) -> None:
        self.interruptions += 1
        started = self.player.stop()
        if self._stt_audio is not None:  # cut in while the last utterance was being transcribed
            self._carry = self._stt_audio
        await self._cancel_turn()
        self.session.interrupt(" ".join(started))

    async def _on_utterance(self, audio: np.ndarray) -> None:
        if self.busy:  # speech ended before barge-in fired; treat it as a barge-in now
            await self._interrupt()
        if self._carry is not None:
            audio, self._carry = np.concatenate([self._carry, audio]), None
        self._start_turn(self._respond(audio, self.clock()))

    async def _respond(self, audio: np.ndarray, t_end: float) -> None:
        rec: dict = {"turn": len(self.records)}
        self._stt_audio = audio
        t0 = self.clock()
        text = await self.stt.transcribe(audio)
        rec["stt_s"] = round(self.clock() - t0, 3)
        self._stt_audio = None
        if not text:
            return
        rec["text"] = text
        self.on_text("candidate", text)
        await self._speak(self.session.respond(text), rec, t_end)

    async def _speak(self, stream, rec: dict, t_end: float | None) -> None:
        self.player.reset()
        splitter = SentenceSplitter()
        sentences: asyncio.Queue[str | None] = asyncio.Queue()
        said: list[str] = []
        t_llm = self.clock()

        async def produce() -> None:
            async with aclosing(stream) as agen:
                async for chunk in agen:
                    rec.setdefault("llm_first_s", round(self.clock() - t_llm, 3))
                    for sentence in splitter.feed(chunk):
                        await sentences.put(sentence)
                if tail := splitter.flush():
                    await sentences.put(tail)
            await sentences.put(None)

        async def consume() -> None:
            while (sentence := await sentences.get()) is not None:
                t0 = self.clock()
                audio = await self.tts.synth(sentence)
                if "tts_first_s" not in rec:
                    rec["tts_first_s"] = round(self.clock() - t0, 3)
                    rec["_enqueued_at"] = self.clock()
                said.append(sentence)
                self.player.enqueue(audio, sentence)

        producer, consumer = asyncio.create_task(produce()), asyncio.create_task(consume())
        try:
            await asyncio.gather(producer, consumer)
            await self.player.wait_idle()
        finally:
            for task in (producer, consumer):
                task.cancel()
            await asyncio.gather(producer, consumer, return_exceptions=True)
            if said:
                self.on_text("interviewer", " ".join(said))
            self._finish_record(rec, t_end)

    def _finish_record(self, rec: dict, t_end: float | None) -> None:
        enqueued = rec.pop("_enqueued_at", None)
        if t_end is not None and enqueued is not None:
            rec["to_first_audio_s"] = round(enqueued - t_end, 3)
        first = getattr(self.player, "first_sample_at", None)
        if t_end is not None and first is not None:
            rec["to_first_sound_s"] = round(first - t_end, 3)
        rec["endpoint_ms"] = self.turn_config.end_silence_ms
        self.records.append(rec)
        if self.latency_path:
            self.latency_path.parent.mkdir(parents=True, exist_ok=True)
            with self.latency_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")


def summarize(records: list[dict]) -> str:
    """Per-stage means over turns that have a measured response."""
    rows = [r for r in records if "to_first_audio_s" in r]
    if not rows:
        return "no completed responses to summarize"

    def avg(key: str) -> float:
        return mean(r[key] for r in rows if key in r)

    total = avg("to_first_sound_s") if all("to_first_sound_s" in r for r in rows) else None
    worst = max(r["to_first_audio_s"] for r in rows)
    lines = [
        f"turns measured: {len(rows)}",
        f"STT            {avg('stt_s'):.2f}s",
        f"LLM first tok  {avg('llm_first_s'):.2f}s",
        f"TTS first clip {avg('tts_first_s'):.2f}s",
        f"end-of-speech -> audio queued  mean {avg('to_first_audio_s'):.2f}s  worst {worst:.2f}s",
    ]
    if total is not None:
        lines.append(f"end-of-speech -> first sound   mean {total:.2f}s")
    lines.append(f"(endpoint wait before this clock starts: {rows[0]['endpoint_ms']} ms)")
    return "\n".join(lines)
