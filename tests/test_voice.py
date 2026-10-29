import asyncio
import time

import numpy as np

from senjin.audio.loop import VoiceConfig, VoiceLoop, summarize
from senjin.audio.stt import clean_transcript, to_wav_bytes
from senjin.audio.tts import SentenceSplitter
from senjin.audio.vad import FRAME, TurnConfig, TurnDetector
from senjin.interview.session import InterviewSession
from senjin.llm.fake import FakeLLM

SPEECH = np.full(FRAME, 0.9, dtype=np.float32)
QUIET = np.zeros(FRAME, dtype=np.float32)


def fake_vad(frame: np.ndarray) -> float:
    return 1.0 if frame.max() > 0.5 else 0.0


def run_detector(pattern: str, **cfg):
    """pattern: 's' speech frame, '.' silence frame."""
    det = TurnDetector(TurnConfig(**cfg))
    events = []
    for ch in pattern:
        frame = SPEECH if ch == "s" else QUIET
        events += det.feed(frame, fake_vad(frame))
    return events


def test_detector_emits_start_then_end_after_silence():
    events = run_detector("." * 5 + "s" * 20 + "." * 30)
    assert [e.kind for e in events] == ["start", "end"]
    audio = events[1].audio
    assert audio is not None and len(audio) > 20 * FRAME  # includes pre-roll and some tail


def test_detector_waits_through_short_pauses():
    events = run_detector("s" * 15 + "." * 10 + "s" * 15 + "." * 30)  # 10 frames = 320 ms pause
    assert [e.kind for e in events] == ["start", "end"]


def test_detector_discards_short_bursts():
    events = run_detector("s" * 4 + "." * 40)  # 128 ms of noise
    assert [e.kind for e in events] == ["start", "discard"]


def test_detector_reports_speech_ms():
    det = TurnDetector()
    for _ in range(12):
        det.feed(SPEECH, 1.0)
    assert det.in_speech and abs(det.speech_ms - 12 * 32) < 1e-6


def test_sentence_splitter_streams_and_handles_abbreviations():
    s = SentenceSplitter()
    out = []
    for chunk in ["Great, thanks. Let ", "me ask, e.g. about caching. How do", "es it scale? Fine"]:
        out += s.feed(chunk)
    assert out == ["Great, thanks.", "Let me ask, e.g. about caching.", "How does it scale?"]
    assert s.flush() == "Fine"


def test_sentence_splitter_merges_tiny_sentences():
    s = SentenceSplitter(min_chars=12)
    assert s.feed("Okay. Tell me about your last project. ") == [
        "Okay. Tell me about your last project."
    ]


def test_clean_transcript_drops_noise_and_hallucinations():
    assert clean_transcript("[BLANK_AUDIO]") == ""
    assert clean_transcript(" (music) hello  there *cough* ") == "hello there"
    assert clean_transcript("Thank you.") == ""


def test_wav_bytes_header():
    data = to_wav_bytes(np.zeros(1600, dtype=np.float32))
    assert data[:4] == b"RIFF" and len(data) == 44 + 3200


class FakeSTT:
    def __init__(self, texts):
        self.texts = list(texts)
        self.called = False

    async def transcribe(self, audio):
        self.called = True
        await asyncio.sleep(0.005)
        return self.texts.pop(0) if self.texts else ""


class FakeTTS:
    sample_rate = 22050

    async def synth(self, text):
        await asyncio.sleep(0.005)
        return np.zeros(100, dtype=np.int16)


class FakePlayer:
    """Each clip plays for `clip_s` of wall time, back to back."""

    def __init__(self, clip_s=0.2):
        self.clip_s = clip_s
        self.clips: list[tuple[str, float]] = []  # (tag, start time)
        self.stopped = 0
        self.first_sample_at = None

    def enqueue(self, audio, tag):
        start = max([time.monotonic()] + [s + self.clip_s for _, s in self.clips[-1:]])
        self.clips.append((tag, start))
        if self.first_sample_at is None:
            self.first_sample_at = start

    @property
    def busy(self):
        return bool(self.clips) and time.monotonic() < self.clips[-1][1] + self.clip_s

    def stop(self):
        self.stopped += 1
        started = [t for t, s in self.clips if s <= time.monotonic()]
        self.clips = []
        return started

    def reset(self):
        self.clips = []
        self.first_sample_at = None

    async def wait_idle(self):
        while self.busy:
            await asyncio.sleep(0.005)


async def frames(script: str, loop_ref: dict, until, pace=0.004):
    """Play a script of 's'/'.' frames, then silence until `until()` is true."""
    for ch in script:
        await asyncio.sleep(pace)
        yield SPEECH if ch == "s" else QUIET
    while not until():
        await asyncio.sleep(pace)
        yield QUIET


def make_loop(llm_replies, stt_texts, clip_s=0.03, **kw):
    session = InterviewSession(FakeLLM(llm_replies))
    player = FakePlayer(clip_s)
    loop = VoiceLoop(session, fake_vad, FakeSTT(stt_texts), FakeTTS(), player, **kw)
    return loop, session, player


async def test_voice_turn_end_to_end_with_latency():
    loop, session, player = make_loop(
        ["Hello there, welcome. Ready to begin?", "Great. Tell me about your last project."],
        ["I am ready"],
    )
    ref: dict = {}
    script = "." * 40 + "s" * 20  # opening plays first, then the candidate speaks
    await asyncio.wait_for(
        loop.run(frames(script, ref, lambda: len(loop.records) >= 2 and not loop.busy)), 10
    )
    assert [m["role"] for m in session.history] == ["assistant", "user", "assistant"]
    assert session.history[1]["content"] == "I am ready"
    assert not loop.errors and loop.interruptions == 0
    turn = loop.records[1]
    assert turn["text"] == "I am ready"
    assert turn["to_first_audio_s"] >= 0 and turn["endpoint_ms"] == 800
    assert "end-of-speech" in summarize(loop.records)


async def test_barge_in_stops_playback_and_truncates_history():
    long_reply = (
        "Sentence number one is here. Sentence number two is here. Sentence number three is here."
    )
    loop, session, player = make_loop(
        [long_reply, "Sure, go ahead."],
        ["wait I have a question"],
        clip_s=0.2,
        config=VoiceConfig(),
    )
    ref: dict = {}
    script = "." * 30 + "s" * 25  # candidate cuts in ~150 ms into the opening
    await asyncio.wait_for(
        loop.run(frames(script, ref, lambda: len(loop.records) >= 2 and not loop.busy)), 10
    )
    assert loop.interruptions == 1 and player.stopped >= 1
    opening = session.history[0]
    assert opening["role"] == "assistant"
    assert opening["content"].startswith("Sentence number one is here.")
    assert "three" not in opening["content"]
    assert session.history[1] == {"role": "user", "content": "wait I have a question"}


async def test_no_barge_in_mode_lets_interviewer_finish():
    long_reply = "Sentence number one is here. Sentence number two is here."
    loop, session, player = make_loop(
        [long_reply, "Okay."], ["hello"], clip_s=0.2, config=VoiceConfig(barge_in=False)
    )
    ref: dict = {}
    script = "." * 10 + "s" * 25 + "." * 40  # speaking over the opening is ignored
    await asyncio.wait_for(
        loop.run(
            frames(script, ref, lambda: loop.records and not loop.busy and len(loop.records) >= 1)
        ),
        10,
    )
    assert loop.interruptions == 0
    assert session.history[0]["content"] == long_reply.strip()


async def test_stt_noise_is_ignored():
    loop, session, _ = make_loop(["Hi, ready when you are."], [""])
    ref: dict = {}
    script = "." * 40 + "s" * 20
    await asyncio.wait_for(
        loop.run(frames(script, ref, lambda: loop.stt.called and not loop.busy)), 10
    )
    assert [m["role"] for m in session.history] == ["assistant"]  # empty transcript, no turn
