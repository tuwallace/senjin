"""The voice loop: mic -> VAD -> STT -> interviewer -> sentence-streamed TTS -> speaker.

Candidate speech during the interviewer's reply (barge-in) stops playback at once; the
interviewer's history keeps only the words that were actually spoken.
"""

import asyncio
import json
import random
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
from senjin.interview.state import Phase

IDLE_PROMPTS = ["Are you still there?", "I still can't hear you. Are you able to continue?"]
GOODBYE_IDLE = (
    "I haven't heard from you, so I'll end the call here. "
    "The team will follow up about next steps. Goodbye."
)
APOLOGY = "Sorry, I had a technical problem. Could you say that again?"
FILLER_TAG = "\x00filler:"  # marks filler clips so they never count as spoken interviewer words
# Plain words only: Piper spells out interjections such as "Mm-hm" letter by letter.
FILLERS = ["Okay, right.", "I see."]
GOODBYE_ERRORS = (
    "I'm having technical difficulties, so I'll end the call here. "
    "Someone from the team will follow up with you. Goodbye."
)


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
    idle_prompt_s: float | None = 25.0  # silence before "Are you still there?" (None disables)
    max_idle_prompts: int = 2  # prompts before the call is ended politely
    max_consecutive_errors: int = 3  # failed replies in a row before the call is ended
    filler_delay_ms: int | None = 1100  # play a short filler if no reply by then (None = off)


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
        self._idle_since = clock()
        self._idle_prompts = 0
        self._errors_in_row = 0
        self._recover = False
        self._fillers: list[tuple[str, np.ndarray]] = []
        self._rng = random.Random()
        self._filled_last_turn = False  # never two fillers in a row: it would sound canned

    @property
    def busy(self) -> bool:
        return self._turn is not None and not self._turn.done()

    async def run(self, frames: AsyncIterator[np.ndarray]) -> None:
        await self._prepare_fillers()
        self._start_turn(self._speak(self.session.open(), {"turn": "open"}, None))
        try:
            async for frame in frames:
                if self.session.finished and not self.busy:
                    break
                prob = self.vad(frame)
                await self._watchdog()
                if self.busy and not self.config.barge_in:
                    self.detector.reset()
                    continue
                events = self.detector.feed(frame, prob)
                if (
                    self.busy
                    and not self._saying_goodbye
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
            self._errors_in_row += 1
            self._recover = True

    async def _watchdog(self) -> None:
        """Between turns: recover from a failed reply, or nudge a silent candidate."""
        if self.busy or self.session.finished:
            return
        if self._recover:
            self._recover = False
            if self._errors_in_row >= self.config.max_consecutive_errors:
                self._start_turn(self._speak_canned(GOODBYE_ERRORS, final=True))
            else:
                self._start_turn(self._speak_canned(APOLOGY))
            return
        limit = self.config.idle_prompt_s
        if limit is None or self.detector.in_speech or self.clock() - self._idle_since < limit:
            return
        if self._idle_prompts >= self.config.max_idle_prompts:
            self._start_turn(self._speak_canned(GOODBYE_IDLE, final=True))
        else:
            prompt = IDLE_PROMPTS[min(self._idle_prompts, len(IDLE_PROMPTS) - 1)]
            self._idle_prompts += 1
            self._start_turn(self._speak_canned(prompt))

    async def _speak_canned(self, text: str, final: bool = False) -> None:
        self.player.reset()
        self.player.enqueue(await self.tts.synth(text), text)
        self.session.say(text, final=final)
        self.on_text("interviewer", text)
        await self.player.wait_idle()
        self._idle_since = self.clock()

    async def _cancel_turn(self) -> None:
        if self._turn and not self._turn.done():
            self._turn.cancel()
            await asyncio.gather(self._turn, return_exceptions=True)

    async def _interrupt(self) -> None:
        self.interruptions += 1
        started = [t for t in self.player.stop() if not t.startswith(FILLER_TAG)]
        if self._stt_audio is not None:  # cut in while the last utterance was being transcribed
            self._carry = self._stt_audio
        await self._cancel_turn()
        self.session.interrupt(" ".join(started))

    @property
    def _saying_goodbye(self) -> bool:
        """Talking over the sign-off must not cut it short; the goodbye is brief."""
        return self.session.state.phase == Phase.SIGNOFF

    async def _on_utterance(self, audio: np.ndarray) -> None:
        if self.busy and self._saying_goodbye:
            return
        if self.busy:  # speech ended before barge-in fired; treat it as a barge-in now
            await self._interrupt()
        self._idle_prompts = 0
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
        self._errors_in_row = 0

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
        filler = None
        if (
            t_end is not None
            and self._fillers
            and self.config.filler_delay_ms
            and not self._filled_last_turn
        ):
            filler = asyncio.create_task(self._filler_after(rec, t_end))
        try:
            await asyncio.gather(producer, consumer)
            await self.player.wait_idle()
        finally:
            tasks = [t for t in (producer, consumer, filler) if t]
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            if said:
                self.on_text("interviewer", " ".join(said))
            self._finish_record(rec, t_end)
            self._idle_since = self.clock()

    async def _prepare_fillers(self) -> None:
        if self.config.filler_delay_ms and not self._fillers:
            self._fillers = [(phrase, await self.tts.synth(phrase)) for phrase in FILLERS]

    async def _filler_after(self, rec: dict, t_end: float) -> None:
        """If no reply audio exists by the delay, play a short neutral acknowledgement."""
        wait = (self.config.filler_delay_ms or 0) / 1000 - (self.clock() - t_end)
        if wait > 0:
            await asyncio.sleep(wait)
        if "_enqueued_at" in rec:  # the real reply got there first
            return
        phrase, audio = self._rng.choice(self._fillers)
        rec["filler"] = phrase
        self.player.enqueue(audio, FILLER_TAG + phrase)
        if self.session.log:
            self.session.log.append("filler", phrase, self.session.state.phase.value)

    def _finish_record(self, rec: dict, t_end: float | None) -> None:
        enqueued = rec.pop("_enqueued_at", None)
        if t_end is not None and enqueued is not None:
            rec["to_first_audio_s"] = round(enqueued - t_end, 3)
        first = getattr(self.player, "first_sample_at", None)
        if t_end is not None and first is not None:
            rec["to_first_sound_s"] = round(first - t_end, 3)
        rec["endpoint_ms"] = self.turn_config.end_silence_ms
        self._filled_last_turn = "filler" in rec
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
    fillers = sum(1 for r in rows if r.get("filler"))
    if fillers:
        lines.append(f"filler clip played on {fillers} of {len(rows)} turns")
    lines.append(f"(endpoint wait before this clock starts: {rows[0]['endpoint_ms']} ms)")
    return "\n".join(lines)
