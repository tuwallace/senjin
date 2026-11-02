import asyncio
import time

import numpy as np

from senjin.audio.loop import FILLER_TAG, FILLERS, VoiceConfig, VoiceLoop, summarize
from senjin.audio.stt import clean_transcript, to_wav_bytes
from senjin.audio.tts import SentenceSplitter
from senjin.audio.vad import FRAME, TurnConfig, TurnDetector
from senjin.interview.session import InterviewSession
from senjin.interview.state import InterviewState, Phase
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


async def test_silence_watchdog_prompts_then_ends_politely():
    loop, session, player = make_loop(
        ["Hello and welcome, are you ready?"],
        [],
        clip_s=0.01,
        config=VoiceConfig(idle_prompt_s=0.15, max_idle_prompts=2),
    )
    ref: dict = {}
    await asyncio.wait_for(
        loop.run(frames("", ref, lambda: session.finished and not loop.busy)), 10
    )
    said = [m["content"] for m in session.history if m["role"] == "assistant"]
    assert said[1] == "Are you still there?"
    assert "can't hear you" in said[2]
    assert "end the call here" in said[3]
    assert session.finished


async def test_candidate_speech_resets_the_idle_clock():
    loop, session, _ = make_loop(
        ["Welcome, are you ready?", "Great, tell me about yourself."],
        ["I am ready"],
        clip_s=0.01,
        config=VoiceConfig(idle_prompt_s=0.6),
    )
    ref: dict = {}
    script = "." * 20 + "s" * 20  # speaks well inside the idle window
    await asyncio.wait_for(
        loop.run(frames(script, ref, lambda: len(loop.records) >= 2 and not loop.busy)), 10
    )
    said = [m["content"] for m in session.history if m["role"] == "assistant"]
    assert "Are you still there?" not in said


async def test_failed_reply_gets_an_apology_then_recovers():
    class Flaky(FakeLLM):
        async def stream(self, messages, tools=None):
            if len(self.calls) == 1:  # the reply to the first candidate answer
                self.calls.append(messages)
                raise RuntimeError("ollama down")
            async for e in super().stream(messages, tools):
                yield e

    session = InterviewSession(Flaky(["Hi, are you ready to begin?", "unused"]))
    player = FakePlayer(0.01)
    loop = VoiceLoop(
        session, fake_vad, FakeSTT(["I am ready"]), FakeTTS(), player,
        VoiceConfig(idle_prompt_s=None),
    )  # fmt: skip
    ref: dict = {}
    script = "." * 40 + "s" * 20

    def apologized() -> bool:
        said = [m["content"] for m in session.history if m["role"] == "assistant"]
        return any(t.startswith("Sorry") for t in said) and not loop.busy

    await asyncio.wait_for(loop.run(frames(script, ref, apologized)), 10)
    assert len(loop.errors) == 1
    assert any(m["content"].startswith("Sorry, I had a technical problem") for m in session.history)


async def test_repeated_errors_end_the_call():
    class Broken(FakeLLM):
        async def stream(self, messages, tools=None):
            if self.calls:
                raise RuntimeError("down")
            async for e in super().stream(messages, tools):
                yield e
            self.calls.append(messages)

    session = InterviewSession(Broken(["Hi, are you ready to begin?"]))
    loop = VoiceLoop(
        session, fake_vad, FakeSTT(["a"] * 5), FakeTTS(), FakePlayer(0.01),
        VoiceConfig(idle_prompt_s=None, max_consecutive_errors=2),
    )  # fmt: skip
    ref: dict = {}
    script = ("." * 40 + "s" * 20) * 3
    await asyncio.wait_for(
        loop.run(frames(script, ref, lambda: session.finished and not loop.busy)), 15
    )
    assert session.finished
    said = [m["content"] for m in session.history if m["role"] == "assistant"]
    assert "technical difficulties" in said[-1]


async def test_talking_over_the_goodbye_does_not_cut_it_off():
    from senjin.interview.state import InterviewState, Phase

    llm = FakeLLM(["Thank you so much for your time today. The team will follow up soon. Goodbye."])
    session = InterviewSession(llm, state=InterviewState(phase=Phase.SIGNOFF))
    player = FakePlayer(0.2)
    loop = VoiceLoop(session, fake_vad, FakeSTT(["thanks bye"]), FakeTTS(), player)
    ref: dict = {}
    script = "." * 20 + "s" * 25  # the candidate speaks over the goodbye
    await asyncio.wait_for(
        loop.run(frames(script, ref, lambda: session.finished and not loop.busy)), 10
    )
    assert loop.interruptions == 0  # (the final stop() on exit is not an interruption)
    assert session.history[0]["content"].endswith("Goodbye.")
    assert session.finished


class SlowLLM(FakeLLM):
    """First text arrives `delay` seconds after the request (not for the opening)."""

    def __init__(self, replies, delay):
        super().__init__(replies)
        self.delay = delay

    async def stream(self, messages, tools=None):
        if self.calls:  # skip the opening
            await asyncio.sleep(self.delay)
        async for event in super().stream(messages, tools):
            yield event


def make_slow_loop(delay, filler_ms=100, replies=None):
    llm = SlowLLM(replies or ["Welcome, are you ready?", "Good. Tell me about a project."], delay)
    session = InterviewSession(llm)
    player = FakePlayer(0.03)
    config = VoiceConfig(idle_prompt_s=None, filler_delay_ms=filler_ms)
    loop = VoiceLoop(session, fake_vad, FakeSTT(["I am ready"]), FakeTTS(), player, config)
    return loop, session, player


async def run_one_turn(loop, session):
    script = "." * 40 + "s" * 20
    until = lambda: len(loop.records) >= 2 and not loop.busy  # noqa: E731
    await asyncio.wait_for(loop.run(frames(script, {}, until)), 10)


async def test_filler_plays_when_the_reply_is_slow_and_stays_out_of_history():
    loop, session, player = make_slow_loop(delay=0.5, filler_ms=100)
    played = []
    original = player.enqueue
    player.enqueue = lambda audio, tag: (played.append(tag), original(audio, tag))[1]
    await run_one_turn(loop, session)
    fillers = [t for t in played if t.startswith(FILLER_TAG)]
    assert len(fillers) == 1 and fillers[0][len(FILLER_TAG) :] in FILLERS
    assert played.index(fillers[0]) < len(played) - 1  # the real reply follows the filler
    assert loop.records[1]["filler"] in FILLERS
    spoken = [m["content"] for m in session.history]
    assert not any(f in " ".join(spoken) for f in FILLERS)
    assert "filler clip played" in summarize(
        loop.records + [{**loop.records[1], "to_first_audio_s": 1}]
    )


async def test_no_filler_when_the_reply_is_fast():
    loop, session, player = make_slow_loop(delay=0.0, filler_ms=400)
    await run_one_turn(loop, session)
    assert "filler" not in loop.records[1]


async def test_filler_can_be_turned_off():
    loop, session, player = make_slow_loop(delay=0.5, filler_ms=0)
    await run_one_turn(loop, session)
    assert "filler" not in loop.records[1]
    assert loop._fillers == []


def test_filler_phrases_are_plain_words_piper_can_say():
    assert FILLERS == ["Okay, right.", "I see."]


async def test_fillers_are_chosen_at_random_from_the_two_phrases():
    loop, session, player = make_slow_loop(delay=0.0, filler_ms=50)
    await loop._prepare_fillers()
    seen = []
    for _ in range(60):
        rec: dict = {}
        await loop._filler_after(rec, loop.clock() - 1)
        seen.append(rec["filler"])
    assert set(seen) == set(FILLERS)  # both appear
    assert seen != sorted(seen) and seen != seen[::-1]  # not a fixed pattern


async def test_barge_in_during_a_filler_does_not_become_interviewer_speech():
    loop, session, player = make_slow_loop(delay=1.0, filler_ms=50)
    player.clip_s = 0.5  # long filler so the candidate cuts in while it is playing
    script = "." * 40 + "s" * 20 + "." * 60 + "s" * 20  # second utterance interrupts the reply
    loop.stt.texts = ["I am ready", "wait, one more thing"]
    until = lambda: loop.interruptions >= 1 and not loop.busy and len(loop.stt.texts) == 0  # noqa: E731
    await asyncio.wait_for(loop.run(frames(script, {}, until)), 15)
    assert loop.interruptions >= 1
    assert not any(f in m["content"] for m in session.history for f in FILLERS)


async def test_no_filler_on_two_turns_in_a_row():
    llm = SlowLLM(["Welcome, ready?", "Okay. Tell me more.", "Thanks. And then?"], 0.5)
    session = InterviewSession(llm)
    config = VoiceConfig(idle_prompt_s=None, filler_delay_ms=100)
    loop = VoiceLoop(
        session,
        fake_vad,
        FakeSTT(["my first answer", "my second answer"]),
        FakeTTS(),
        FakePlayer(0.03),
        config,
    )
    script = "." * 40 + "s" * 20 + "." * 120 + "s" * 20
    until = lambda: len(loop.records) >= 3 and not loop.busy  # noqa: E731
    await asyncio.wait_for(loop.run(frames(script, {}, until)), 20)
    assert "filler" in loop.records[1]
    assert "filler" not in loop.records[2]


async def test_unfinished_fragment_is_merged_with_the_rest_of_the_answer(tmp_path):
    from senjin.interview.transcript import TranscriptLog

    llm = SlowLLM(["Welcome, are you ready?", "Understood. Tell me about a project."], 0.5)
    session = InterviewSession(llm, TranscriptLog(tmp_path / "s.jsonl"))
    config = VoiceConfig(idle_prompt_s=None, filler_delay_ms=None)
    stt = FakeSTT(["I would set a timer", "I would set a timer or push from the server"])
    loop = VoiceLoop(session, fake_vad, stt, FakeTTS(), FakePlayer(0.03), config)
    # first part, a pause long enough to end the turn, then the rest while the model is thinking
    script = "." * 40 + "s" * 20 + "." * 45 + "s" * 20
    until = lambda: not stt.texts and not loop.busy  # noqa: E731
    await asyncio.wait_for(loop.run(frames(script, {}, until)), 15)
    users = [m["content"] for m in session.history if m["role"] == "user"]
    assert users == ["I would set a timer or push from the server"]
    rows = TranscriptLog.load(tmp_path / "s.jsonl")
    assert "retract" in [r["role"] for r in rows]


class SlowOpeningLLM(FakeLLM):
    """The model is still loading when the greeting is requested: its first text comes late."""

    async def stream(self, messages, tools=None):
        if not self.calls:
            await asyncio.sleep(0.6)
        async for event in super().stream(messages, tools):
            yield event


async def test_opening_killed_by_noise_is_said_again():
    llm = SlowOpeningLLM(["Hello, I am Senjin. Are you ready to begin?"])
    session = InterviewSession(llm)
    player = FakePlayer(0.05)
    config = VoiceConfig(idle_prompt_s=None, filler_delay_ms=None)
    loop = VoiceLoop(session, fake_vad, FakeSTT([""]), FakeTTS(), player, config)
    script = "." * 5 + "s" * 25  # a cough while the model is still preparing the greeting
    until = lambda: bool(session.history) and not loop.busy  # noqa: E731
    await asyncio.wait_for(loop.run(frames(script, {}, until)), 10)
    assert loop.interruptions == 1
    assert [m["role"] for m in session.history] == ["assistant"]
    assert session.history[0]["content"].startswith("Hello, I am Senjin")


async def test_no_filler_after_a_one_word_utterance():
    loop, session, player = make_slow_loop(0.5, filler_ms=100)
    loop.stt.texts = ["Hello?"]
    await run_one_turn(loop, session)
    assert "filler" not in loop.records[-1]


def test_looks_unfinished():
    from senjin.audio.loop import looks_unfinished

    assert looks_unfinished("I created a data change capture and")
    assert looks_unfinished("I would use the")
    assert looks_unfinished("it scales well")  # no closing mark
    assert not looks_unfinished("I created a data change capture.")


def _hold_loop(stt_texts, hold_ms, phase=Phase.TECHNICAL):
    llm = SlowLLM(["Welcome.", "Understood. Next question?"], 0.2)
    state = InterviewState(phase=phase)
    session = InterviewSession(llm, state=state)
    config = VoiceConfig(idle_prompt_s=None, filler_delay_ms=None, short_answer_hold_ms=hold_ms)
    loop = VoiceLoop(session, fake_vad, FakeSTT(stt_texts), FakeTTS(), FakePlayer(0.03), config)
    return loop, session


async def test_short_answer_is_held_so_a_pause_does_not_cut_it_off():
    loop, session = _hold_loop(
        ["I did it.", "I did it. I built a change data capture pipeline."], 800
    )
    # a short answer, a pause that ends the turn, then the candidate carries on during the hold
    script = "." * 40 + "s" * 20 + "." * 45 + "s" * 20
    until = lambda: not loop.stt.texts and not loop.busy  # noqa: E731
    await asyncio.wait_for(loop.run(frames(script, {}, until)), 15)
    users = [m["content"] for m in session.history if m["role"] == "user"]
    assert users == ["I did it. I built a change data capture pipeline."]
    # the held answer never reached the session, so the interviewer's question was not undone
    assert [m["role"] for m in session.history].count("assistant") == 2


async def test_no_hold_during_the_intro_or_when_switched_off():
    for phase, hold in ((Phase.INTRO, 800), (Phase.TECHNICAL, 0)):
        loop, session = _hold_loop(["Yes."], hold, phase)
        script = "." * 40 + "s" * 20
        until = lambda lp=loop: len(lp.records) >= 2 and not lp.busy  # noqa: E731
        await asyncio.wait_for(loop.run(frames(script, {}, until)), 15)
        assert "held_s" not in loop.records[-1]


def test_speaker_change_marks_are_dropped_from_transcripts():
    assert (
        clean_transcript(">> Well, for page speed it is fine.")
        == "Well, for page speed it is fine."
    )
    assert clean_transcript("- I'm ready.") == "I'm ready."


async def test_no_hold_after_a_complete_non_answer():
    for said in ("I don't know.", "No.", "Not sure."):
        loop, session = _hold_loop([said], 800)
        script = "." * 40 + "s" * 20
        until = lambda lp=loop: len(lp.records) >= 2 and not lp.busy  # noqa: E731
        await asyncio.wait_for(loop.run(frames(script, {}, until)), 15)
        assert "held_s" not in loop.records[-1], said


async def test_an_unfinished_dont_know_is_still_held():
    loop, session = _hold_loop(["I don't know how I would keep it fresh and"], 300)
    script = "." * 40 + "s" * 20
    until = lambda: len(loop.records) >= 2 and not loop.busy  # noqa: E731
    await asyncio.wait_for(loop.run(frames(script, {}, until)), 15)
    assert loop.records[-1].get("held_s", 0) > 0.2


def test_player_falls_back_to_the_device_rate_when_it_refuses_the_voice_rate(monkeypatch):
    import sys
    import types

    import numpy as np

    from senjin.audio.devices import Player, resample_int16

    class PortAudioError(Exception):
        pass

    opened: list[int] = []

    class Stream:
        def __init__(self, samplerate, **kw):
            if samplerate != 48000:
                raise PortAudioError("-9986")
            opened.append(samplerate)

        def start(self):
            pass

    fake = types.SimpleNamespace(
        OutputStream=Stream,
        PortAudioError=PortAudioError,
        query_devices=lambda device, kind: {"default_samplerate": 48000.0},
    )
    monkeypatch.setitem(sys.modules, "sounddevice", fake)
    player = Player(22050, device=5, channels=2)
    assert opened == [48000] and player.sample_rate == 22050  # callers still pass 22.05 kHz clips
    player.enqueue(np.ones(2205, dtype=np.int16), "a")
    assert len(player._clips[0][1]) == 4800  # 0.1 s at the device rate
    assert resample_int16(np.arange(5, dtype=np.int16), 8, 8).tolist() == [0, 1, 2, 3, 4]
