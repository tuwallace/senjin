"""Run a full spoken interview inside a Google Meet."""

import asyncio
import os
import time
from contextlib import AsyncExitStack
from pathlib import Path

from senjin.audio.devices import Player, mic_frames
from senjin.audio.loop import VoiceConfig, VoiceLoop, summarize
from senjin.audio.models import SILERO_PATH, ensure_models
from senjin.audio.stt import WhisperServerSTT
from senjin.audio.tts import PiperTTS
from senjin.audio.vad import SileroVAD, TurnConfig
from senjin.config import InterviewConfig
from senjin.interview.exemplars import ExemplarIndex
from senjin.interview.session import InterviewSession
from senjin.interview.transcript import TranscriptLog
from senjin.llm.client import OllamaClient
from senjin.llm.tools import FilteredTools, open_mcp_tools
from senjin.meet.audio import find_bridge
from senjin.meet.joiner import MeetBot
from senjin.meet.supervisor import watch_call

LABEL = {"interviewer": "Sen", "candidate": "Candidate"}


async def run_meet_interview(
    url: str,
    *,
    model: str,
    stt_model: str,
    speak: str,
    hear: str,
    out: Path,
    admit_timeout: float = 300.0,
    wait_candidate: float = 600.0,
    config: InterviewConfig | None = None,
    guest_name: str = "Sen (AI interviewer)",
    exemplars: Path | None = None,
    candidate_dir: Path | None = None,
    log=print,
) -> Path:
    cfg = config or InterviewConfig.from_env()
    log(cfg.describe())
    bridge = find_bridge(speak=speak, hear=hear)  # fail fast, before opening a browser
    ensure_models()
    tts = PiperTTS()
    stt = WhisperServerSTT(stt_model)
    log("Starting speech recognition...")
    await stt.start()
    bot = MeetBot(mic_name=bridge.speak_name, speaker_name=bridge.hear_name, guest_name=guest_name)
    player = Player(tts.sample_rate, device=bridge.speak_index, channels=2)
    async with AsyncExitStack() as stack:
        try:
            await bot.start()
            if not await bot.signed_in():
                log("Not signed in to Google: joining as a guest (the host must admit the bot).")
            await bot.join(url, admit_timeout, log)
            log(f"In the call. Waiting up to {wait_candidate:.0f}s for the candidate to arrive...")
            if not await bot.wait_for_participants(2, wait_candidate):
                log("Candidate did not arrive (or participant count unreadable); starting anyway.")

            env = {**os.environ, "SENJIN_NOTES_PATH": str(out.with_suffix(".notes.jsonl"))}
            if candidate_dir:
                env["SENJIN_CANDIDATE_DIR"] = str(candidate_dir)
            tools = FilteredTools(
                await stack.enter_async_context(open_mcp_tools(env=env)), {"record_note"}
            )
            session = InterviewSession(
                OllamaClient(model=model),
                TranscriptLog(out),
                tools=tools,
                exemplars=ExemplarIndex.load(exemplars) if exemplars else None,
                prefetch=True,
                state=cfg.state(),
            )
            loop = VoiceLoop(
                session, SileroVAD(SILERO_PATH), stt, tts, player,
                VoiceConfig(
                    idle_prompt_s=cfg.idle_seconds or None,
                    filler_delay_ms=cfg.filler_ms or None,
                ),
                TurnConfig(end_silence_ms=cfg.silence_ms),
                latency_path=out.with_suffix(".latency.jsonl"),
                on_text=lambda who, text: log(f"{LABEL[who]}: {text}"),
            )  # fmt: skip
            run = asyncio.create_task(loop.run(mic_frames(bridge.hear_index, channels=2)))
            ended = asyncio.create_task(watch_call(bot, url, lambda: session.finished, log=log))
            done, pending = await asyncio.wait({run, ended}, return_when=asyncio.FIRST_COMPLETED)
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            if run in done and run.exception():
                raise run.exception()
            log("Interview over." if session.finished else "The call ended before the sign-off.")
            log(summarize(loop.records))
            for err in loop.errors:
                log(f"error: {err!r}")
        finally:
            await bot.leave()
            await bot.close()
            player.close()
            stt.stop()
    return out


def default_out() -> Path:
    return Path("data/sessions") / f"meet-{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
