"""`senjin-voice setup | devices | check | simulate | run | report`."""

import argparse
import asyncio
import json
import os
import time
from contextlib import AsyncExitStack
from pathlib import Path

from senjin.audio.loop import VoiceConfig, VoiceLoop, summarize
from senjin.audio.models import SILERO_PATH, ensure_models
from senjin.audio.stt import DEFAULT_WHISPER_MODEL, WhisperServerSTT
from senjin.audio.vad import SileroVAD, TurnConfig
from senjin.interview.exemplars import ExemplarIndex
from senjin.interview.session import InterviewSession
from senjin.interview.transcript import TranscriptLog
from senjin.llm.client import DEFAULT_MODEL, OllamaClient
from senjin.llm.tools import FilteredTools, open_mcp_tools

# record_note costs a full LLM round trip before the interviewer can speak; skip it live.
VOICE_HIDDEN = {"record_note"}


def make_tts(kind: str):
    from senjin.audio.tts import PiperTTS, SayTTS

    return SayTTS() if kind == "say" else PiperTTS()


async def check(args) -> None:
    """Round trip without a microphone: speak a sentence, transcribe it, report timings."""
    import numpy as np

    tts = make_tts(args.tts)
    text = "Tell me about the payments ledger you built, and the trade-offs you made."
    t = time.monotonic()
    pcm = await tts.synth(text)
    print(f"TTS  {time.monotonic() - t:.2f}s for {len(pcm) / tts.sample_rate:.1f}s of audio")
    x = pcm.astype(np.float32) / 32768
    idx = np.arange(0, len(x), tts.sample_rate / 16000)
    audio16 = np.interp(idx, np.arange(len(x)), x).astype(np.float32)
    stt = WhisperServerSTT(args.stt_model)
    await stt.start()
    try:
        await stt.transcribe(audio16)  # warm-up
        t = time.monotonic()
        heard = await stt.transcribe(audio16)
        print(f"STT  {time.monotonic() - t:.2f}s -> {heard}")
    finally:
        stt.stop()
    vad = SileroVAD(SILERO_PATH)
    probs = [vad(audio16[i : i + 512]) for i in range(0, len(audio16) - 512, 512)]
    print(f"VAD  max speech probability {max(probs):.2f}")


async def run(args) -> None:
    import numpy as np  # noqa: F401  (fail early if audio deps are missing)

    from senjin.audio.devices import Player, mic_frames

    ensure_models()
    out = args.out or Path("data/sessions") / f"voice-{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
    tts = make_tts(args.tts)
    stt = WhisperServerSTT(args.stt_model)
    print("Starting speech recognition...")
    await stt.start()
    player = Player(tts.sample_rate)
    async with AsyncExitStack() as stack:
        env = {**os.environ, "SENJIN_NOTES_PATH": str(out.with_suffix(".notes.jsonl"))}
        tools = FilteredTools(
            await stack.enter_async_context(open_mcp_tools(env=env)), VOICE_HIDDEN
        )
        session = InterviewSession(
            OllamaClient(model=args.model),
            TranscriptLog(out),
            tools=tools,
            prefetch=True,
            exemplars=ExemplarIndex.load(args.exemplars) if args.exemplars else None,
        )
        loop = VoiceLoop(
            session,
            SileroVAD(SILERO_PATH),
            stt,
            tts,
            player,
            VoiceConfig(barge_in=not args.no_barge_in, idle_prompt_s=args.idle_seconds or None),
            TurnConfig(end_silence_ms=args.silence_ms),
            latency_path=out.with_suffix(".latency.jsonl"),
            on_text=lambda who, text: print(
                f"\n{'Sen' if who == 'interviewer' else 'You'}: {text}"
            ),
        )
        print(f"Transcript: {out}\nUse headphones. Speak when Sen finishes; Ctrl-C to stop.")
        try:
            await loop.run(mic_frames())
        except (KeyboardInterrupt, asyncio.CancelledError):
            pass
        finally:
            stt.stop()
            player.close()
        print("\n" + summarize(loop.records))
        for err in loop.errors:
            print(f"error: {err!r}")


SIM_ANSWERS = [
    "Yes, I'm ready.",
    "At Northwind Pay I led the rebuild of our payments ledger in Go and Postgres. "
    "I designed the migration to event sourcing, using dual writes and a reconciliation job.",
    "We cut duplicate charges by ninety eight percent by adding idempotency keys to every request.",
    "We ran both systems in parallel for a month and compared balances every night.",
    "Sure. I would use a token bucket per customer in Redis, updated atomically with a Lua script.",
    "If Redis is down I would fail open with a small local limiter as a safety net.",
]


def to_16k(pcm, rate):
    import numpy as np

    x = pcm.astype(np.float32) / 32768
    idx = np.arange(0, len(x), rate / 16000)
    return np.interp(idx, np.arange(len(x)), x).astype(np.float32)


async def simulate(args) -> None:
    """Full pipeline with synthetic candidate speech instead of a microphone."""
    import numpy as np

    from senjin.audio.devices import SilentPlayer
    from senjin.audio.vad import FRAME

    ensure_models()
    out = args.out or Path("data/sessions") / f"voicesim-{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
    tts = make_tts(args.tts)
    stt = WhisperServerSTT(args.stt_model)
    await stt.start()
    clips = [to_16k(await tts.synth(t), tts.sample_rate) for t in SIM_ANSWERS[: args.turns]]
    async with AsyncExitStack() as stack:
        tools = FilteredTools(await stack.enter_async_context(open_mcp_tools()), VOICE_HIDDEN)
        session = InterviewSession(
            OllamaClient(model=args.model), TranscriptLog(out), tools=tools, prefetch=True
        )
        loop = VoiceLoop(
            session, SileroVAD(SILERO_PATH), stt, tts, SilentPlayer(tts.sample_rate),
            VoiceConfig(), TurnConfig(end_silence_ms=args.silence_ms),
            latency_path=out.with_suffix(".latency.jsonl"),
            on_text=lambda who, text: print(f"{'Sen' if who == 'interviewer' else 'You'}: {text}"),
        )  # fmt: skip
        quiet = np.zeros(FRAME, dtype=np.float32)

        async def frames():
            for clip in clips:
                while loop.busy or not loop.records:  # wait for the interviewer to finish
                    yield quiet
                    await asyncio.sleep(FRAME / 16000)
                for _ in range(25):  # a beat of silence before speaking
                    yield quiet
                    await asyncio.sleep(FRAME / 16000)
                for i in range(0, len(clip) - FRAME, FRAME):
                    yield clip[i : i + FRAME]
                    await asyncio.sleep(FRAME / 16000)
                seen = len(loop.records)
                while len(loop.records) <= seen or loop.busy:  # until the reply is done
                    yield quiet
                    await asyncio.sleep(FRAME / 16000)

        try:
            await loop.run(frames())
        finally:
            stt.stop()
        print("\n" + summarize(loop.records))
        print(f"transcript: {out}")
        for err in loop.errors:
            print(f"error: {err!r}")


def report(path: Path) -> None:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    print(summarize(rows))


def main() -> None:
    parser = argparse.ArgumentParser(prog="senjin-voice")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("setup", help="download the VAD and Piper voice models")
    sub.add_parser("devices", help="list audio devices")
    for name in ("check", "run", "simulate"):
        p = sub.add_parser(name)
        p.add_argument("--stt-model", default=DEFAULT_WHISPER_MODEL)
        p.add_argument("--tts", choices=["piper", "say"], default="piper")
    run_p = sub.choices["run"]
    run_p.add_argument("--model", default=DEFAULT_MODEL)
    run_p.add_argument("--exemplars", type=Path)
    run_p.add_argument(
        "--no-barge-in", action="store_true", help="use with speakers, not headphones"
    )
    run_p.add_argument("--silence-ms", type=int, default=800, help="silence that ends your turn")
    run_p.add_argument("--max-minutes", type=float, default=45, help="time box for the interview")
    run_p.add_argument(
        "--idle-seconds", type=float, default=25, help="silence before a nudge (0 = off)"
    )
    run_p.add_argument("--out", type=Path)
    sim_p = sub.choices["simulate"]
    sim_p.add_argument("--model", default=DEFAULT_MODEL)
    sim_p.add_argument("--turns", type=int, default=6, help="candidate answers to play (max 6)")
    sim_p.add_argument("--silence-ms", type=int, default=800)
    sim_p.add_argument("--out", type=Path)
    rep = sub.add_parser("report", help="summarize a .latency.jsonl file")
    rep.add_argument("path", type=Path)
    args = parser.parse_args()

    if args.cmd == "setup":
        for path in ensure_models():
            print(path)
    elif args.cmd == "devices":
        import sounddevice as sd

        print(sd.query_devices())
    elif args.cmd == "check":
        asyncio.run(check(args))
    elif args.cmd == "run":
        asyncio.run(run(args))
    elif args.cmd == "simulate":
        asyncio.run(simulate(args))
    else:
        report(args.path)


if __name__ == "__main__":
    main()
