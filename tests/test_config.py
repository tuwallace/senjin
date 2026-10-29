import pytest

from senjin.config import ConfigError, InterviewConfig
from senjin.interview.session import InterviewSession
from senjin.interview.state import InterviewState, Phase
from senjin.llm.fake import FakeLLM


def test_defaults_are_five_questions_split_two_and_three():
    cfg = InterviewConfig.from_env({})
    assert (cfg.max_questions, cfg.resume, cfg.technical) == (5, 2, 3)
    budgets = cfg.budgets()
    turns = {p: b.max_turns for p, b in budgets.items()}
    assert budgets[Phase.CANDIDATE_QA].max_turns is None  # the candidate decides
    assert budgets[Phase.CANDIDATE_QA].max_seconds is None
    del turns[Phase.CANDIDATE_QA]
    assert turns == {
        Phase.INTRO: 1,
        Phase.RESUME_DIVE: 2,
        Phase.TECHNICAL: 3,
        Phase.SIGNOFF: 1,
    }


@pytest.mark.parametrize(
    "n, resume, technical", [(2, 1, 1), (3, 2, 1), (5, 2, 3), (8, 4, 4), (10, 4, 6), (30, 12, 18)]
)
def test_question_split_always_has_both_parts(n, resume, technical):
    cfg = InterviewConfig.from_env({"SENJIN_MAX_QUESTIONS": str(n)})
    assert (cfg.resume, cfg.technical) == (resume, technical)
    assert cfg.resume + cfg.technical == n


def test_environment_overrides_every_limit():
    cfg = InterviewConfig.from_env(
        {
            "SENJIN_MAX_QUESTIONS": "7",
            "SENJIN_RESUME_QUESTIONS": "1",
            "SENJIN_MAX_MINUTES": "20",
            "SENJIN_IDLE_SECONDS": "0",
        }
    )
    assert (cfg.resume, cfg.technical) == (1, 6)
    assert cfg.max_minutes == 20 and cfg.idle_seconds == 0
    assert cfg.state().max_total_seconds == 1200


@pytest.mark.parametrize(
    "env, name",
    [
        ({"SENJIN_MAX_QUESTIONS": "abc"}, "SENJIN_MAX_QUESTIONS"),
        ({"SENJIN_MAX_QUESTIONS": "1"}, "SENJIN_MAX_QUESTIONS"),
        ({"SENJIN_MAX_QUESTIONS": "500"}, "SENJIN_MAX_QUESTIONS"),
        ({"SENJIN_MAX_QUESTIONS": "5", "SENJIN_RESUME_QUESTIONS": "5"}, "SENJIN_RESUME_QUESTIONS"),
        ({"SENJIN_MAX_MINUTES": "-3"}, "SENJIN_MAX_MINUTES"),
    ],
)
def test_bad_values_name_the_variable(env, name):
    with pytest.raises(ConfigError, match=name):
        InterviewConfig.from_env(env)


def test_blank_values_fall_back_to_defaults():
    assert InterviewConfig.from_env({"SENJIN_MAX_QUESTIONS": " "}).max_questions == 5


def test_flags_override_environment():
    base = InterviewConfig.from_env({"SENJIN_MAX_QUESTIONS": "9", "SENJIN_MAX_MINUTES": "30"})
    cfg = base.override(max_questions=4, idle_seconds=10)
    assert (cfg.max_questions, cfg.max_minutes, cfg.idle_seconds) == (4, 30, 10)
    assert base.override() == base
    with pytest.raises(ConfigError):
        base.override(max_questions=1)


def test_default_interview_state_reads_the_environment(monkeypatch):
    monkeypatch.setenv("SENJIN_MAX_QUESTIONS", "4")
    budgets = InterviewState().budgets
    assert budgets[Phase.RESUME_DIVE].max_turns + budgets[Phase.TECHNICAL].max_turns == 4
    monkeypatch.delenv("SENJIN_MAX_QUESTIONS")
    budgets = InterviewState().budgets
    assert budgets[Phase.RESUME_DIVE].max_turns + budgets[Phase.TECHNICAL].max_turns == 5


class Tools:
    tools: list = []

    async def call(self, name, arguments):
        if name == "get_next_technical_question":
            return (
                '{"id": "q", "topic": "t", "difficulty": "easy", "prompt": "Q?", '
                '"follow_ups": ["a?", "b?"]}'
            )
        return "text"


@pytest.mark.parametrize("n", [3, 5, 8])
async def test_interview_asks_exactly_n_questions_then_hands_over_to_candidate(n):
    cfg = InterviewConfig.from_env({"SENJIN_MAX_QUESTIONS": str(n)})
    session = InterviewSession(FakeLLM([]), state=cfg.state(), tools=Tools(), prefetch=True)
    spoken_by_phase: dict[str, int] = {}

    async def say(stream):
        text = "".join([c async for c in stream])
        spoken_by_phase[session.state.phase.value] = (
            spoken_by_phase.get(session.state.phase.value, 0) + 1
        )
        return text

    await say(session.open())
    guard = 0
    while not session.finished and guard < 100:
        in_qa = session.state.phase.value == "candidate_qa"
        await say(session.respond("No, I'm good." if in_qa else "an answer"))
        guard += 1
    questions = spoken_by_phase["resume_dive"] + spoken_by_phase["technical"]
    assert questions == n
    assert spoken_by_phase["intro"] == 1 and spoken_by_phase["signoff"] == 1
    assert spoken_by_phase["candidate_qa"] == 1  # just the invitation, then "no, I'm good"


def test_voice_timing_defaults_and_overrides():
    cfg = InterviewConfig.from_env({})
    assert (cfg.silence_ms, cfg.filler_ms) == (1000, 1100)
    cfg = InterviewConfig.from_env({"SENJIN_SILENCE_MS": "500", "SENJIN_FILLER_MS": "0"})
    assert (cfg.silence_ms, cfg.filler_ms) == (500, 0)
    assert cfg.override(silence_ms=650, filler_ms=1200).silence_ms == 650


@pytest.mark.parametrize("env", [{"SENJIN_SILENCE_MS": "50"}, {"SENJIN_FILLER_MS": "-1"}])
def test_voice_timing_bounds(env):
    with pytest.raises(ConfigError):
        InterviewConfig.from_env(env)


async def test_candidate_qa_has_no_limit_until_the_candidate_is_done():
    cfg = InterviewConfig.from_env({"SENJIN_MAX_QUESTIONS": "2"})
    llm = FakeLLM([])
    session = InterviewSession(llm, state=cfg.state(), tools=Tools(), prefetch=True)
    async for _ in session.open():
        pass
    while session.state.phase.value != "candidate_qa":
        async for _ in session.respond("an answer"):
            pass
    answered = 0
    for i in range(25):  # far more than any old cap
        async for _ in session.respond(f"What about thing number {i}?"):
            pass
        assert session.state.phase.value == "candidate_qa" and not session.finished
        answered += 1
    assert answered == 25
    async for _ in session.respond("No, that's all, thank you."):
        pass
    assert session.state.phase.value == "signoff" and session.finished


@pytest.mark.parametrize(
    "text, done",
    [
        ("No, I'm good.", True),
        ("Nope, that was it.", True),
        ("Thanks, that's all.", True),
        ("No questions.", True),
        ("Okay, thank you.", True),
        ("No, but what about the team size?", False),
        ("Thanks. One more question.", False),
        ("Actually, tell me about the on-call rotation", False),
        ("Is there a bonus?", False),
        ("What is the salary range for this role at this level?", False),
    ],
)
def test_done_detector_only_fires_when_the_candidate_is_really_done(text, done):
    session = InterviewSession(FakeLLM([]))
    session.state.phase = Phase.CANDIDATE_QA
    session.state.turns_in_phase = 1
    assert session._candidate_is_done(text) is done


@pytest.fixture
def isolated_environ(monkeypatch):
    """load_env_file writes to os.environ; give the test its own copy so nothing leaks."""
    import os

    monkeypatch.setattr(os, "environ", os.environ.copy())
    for key in [k for k in os.environ if k.startswith("SENJIN_") and k != "SENJIN_ENV_FILE"]:
        del os.environ[key]


def write_env(tmp_path, text):
    path = tmp_path / ".env"
    path.write_text(text)
    return path


def test_dotenv_values_are_loaded(tmp_path, isolated_environ):
    import os

    from senjin.config import load_env_file

    path = write_env(
        tmp_path,
        '# a comment\n\nSENJIN_MAX_QUESTIONS=7\nexport SENJIN_SILENCE_MS="1200"\n'
        "SENJIN_FILLER_MS='0'\nSENJIN_MAX_MINUTES=30 # half an hour\n"
        "OTHER_THING=ignored\nnot a pair\n",
    )
    loaded = load_env_file([path])
    assert loaded == {
        "SENJIN_MAX_QUESTIONS": "7",
        "SENJIN_SILENCE_MS": "1200",
        "SENJIN_FILLER_MS": "0",
        "SENJIN_MAX_MINUTES": "30",
    }
    cfg = InterviewConfig.from_env(loaded)
    assert (cfg.max_questions, cfg.silence_ms, cfg.filler_ms, cfg.max_minutes) == (7, 1200, 0, 30)
    assert "OTHER_THING" not in os.environ


def test_exported_variables_beat_the_dotenv_file(tmp_path, isolated_environ, monkeypatch):
    from senjin.config import load_env_file

    monkeypatch.setenv("SENJIN_MAX_QUESTIONS", "3")
    path = write_env(tmp_path, "SENJIN_MAX_QUESTIONS=9\nSENJIN_SILENCE_MS=1500\n")
    assert load_env_file([path]) == {"SENJIN_SILENCE_MS": "1500"}
    assert InterviewConfig.from_env().max_questions == 3  # the shell value stayed


def test_dotenv_feeds_the_default_config_and_can_be_disabled(
    tmp_path, isolated_environ, monkeypatch
):
    path = write_env(tmp_path, "SENJIN_MAX_QUESTIONS=8\n")
    monkeypatch.setenv("SENJIN_ENV_FILE", str(path))
    assert InterviewConfig.from_env().max_questions == 8
    monkeypatch.delenv("SENJIN_MAX_QUESTIONS")
    monkeypatch.setenv("SENJIN_ENV_FILE", "")  # empty = ignore .env files
    assert InterviewConfig.from_env().max_questions == 5


def test_dotenv_bad_value_is_reported_by_name(tmp_path, isolated_environ, monkeypatch):
    monkeypatch.setenv("SENJIN_ENV_FILE", str(write_env(tmp_path, "SENJIN_SILENCE_MS=fast\n")))
    with pytest.raises(ConfigError, match="SENJIN_SILENCE_MS"):
        InterviewConfig.from_env()


def test_missing_dotenv_is_fine(tmp_path):
    from senjin.config import load_env_file

    assert load_env_file([tmp_path / "nope.env"]) == {}
