from senjin.interview.session import InterviewSession
from senjin.interview.state import Budget, InterviewState, Phase
from senjin.interview.transcript import TranscriptLog
from senjin.llm.fake import FakeLLM


class Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


async def collect(stream) -> str:
    return "".join([c async for c in stream]).strip()


def one_turn_budgets() -> dict[Phase, Budget]:
    return {p: Budget(max_turns=1, max_seconds=1e9) for p in Phase}


async def test_full_interview_walks_every_phase(tmp_path):
    llm = FakeLLM([f"line {i}" for i in range(10)])
    log = TranscriptLog(tmp_path / "s.jsonl")
    session = InterviewSession(llm, log, InterviewState(budgets=one_turn_budgets()))

    await collect(session.open())
    phases = [session.state.phase]
    while not session.finished:
        await collect(session.respond("answer"))
        phases.append(session.state.phase)

    assert phases[0] == Phase.INTRO
    assert Phase.TECHNICAL in phases
    assert session.state.phase == Phase.SIGNOFF
    rows = TranscriptLog.load(tmp_path / "s.jsonl")
    assert [r["role"] for r in rows][:2] == ["assistant", "user"]
    assert {r["phase"] for r in rows} == {p.value for p in Phase}


async def test_turn_budget_holds_phase():
    budgets = one_turn_budgets()
    budgets[Phase.INTRO] = Budget(max_turns=3, max_seconds=1e9)
    session = InterviewSession(FakeLLM([]), state=InterviewState(budgets=budgets))
    await collect(session.open())
    await collect(session.respond("a"))
    await collect(session.respond("b"))
    assert session.state.phase == Phase.INTRO
    await collect(session.respond("c"))
    assert session.state.phase == Phase.RESUME_DIVE


async def test_time_budget_advances_phase():
    clock = Clock()
    budgets = one_turn_budgets()
    budgets[Phase.INTRO] = Budget(max_turns=99, max_seconds=60)
    session = InterviewSession(FakeLLM([]), state=InterviewState(budgets=budgets), clock=clock)
    await collect(session.open())
    clock.t = 61
    await collect(session.respond("hi"))
    assert session.state.phase == Phase.RESUME_DIVE


async def test_prompt_shape_and_transition_note():
    llm = FakeLLM([])
    session = InterviewSession(llm, state=InterviewState(budgets=one_turn_budgets()))
    await collect(session.open())
    await collect(session.respond("ready"))
    first, second = llm.calls
    assert first[0]["role"] == "system" and "Sen" in first[0]["content"]
    assert first[-1]["role"] == "user"
    assert [m["role"] for m in second].count("system") == 1
    assert "Transition" not in first[0]["content"]
    assert "Transition" in second[0]["content"]
    assert second[-1] == {"role": "user", "content": "ready"}


async def test_signoff_finishes_session():
    session = InterviewSession(FakeLLM([]), state=InterviewState(budgets=one_turn_budgets()))
    await collect(session.open())
    while not session.finished:
        await collect(session.respond("x"))
    assert session.finished
    assert [m["role"] for m in session.history].count("assistant") == len(Phase)


class StubTools:
    tools = [{"type": "function", "function": {"name": "get_candidate_resume"}}]

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    async def call(self, name, arguments):
        self.calls.append((name, arguments))
        if name == "boom":
            raise RuntimeError("down")
        return "RESUME TEXT"


async def test_tool_call_round_trip_keeps_speech_clean():
    from senjin.llm.client import ToolCall

    llm = FakeLLM([[ToolCall("get_candidate_resume")], "Tell me about your ledger."])
    tools = StubTools()
    session = InterviewSession(llm, tools=tools)
    spoken = await collect(session.open())
    assert spoken == "Tell me about your ledger."
    assert tools.calls == [("get_candidate_resume", {})]
    assert llm.tools_seen[0] == tools.tools
    # second model call saw the tool result
    assert llm.calls[1][-1]["role"] == "tool" and llm.calls[1][-1]["content"] == "RESUME TEXT"
    assert [m["role"] for m in session.history] == ["assistant", "tool", "assistant"]
    assert session.history[-1]["content"] == "Tell me about your ledger."


async def test_tool_failure_is_reported_to_model_not_raised():
    from senjin.llm.client import ToolCall

    llm = FakeLLM([[ToolCall("boom")], "Sorry, let's continue."])
    session = InterviewSession(llm, tools=StubTools())
    assert await collect(session.open()) == "Sorry, let's continue."
    assert "ERROR: down" in llm.calls[1][-1]["content"]


async def test_tool_loop_is_bounded():
    from senjin.llm.client import ToolCall

    llm = FakeLLM([[ToolCall("get_candidate_resume")]] * 10)
    session = InterviewSession(llm, tools=StubTools())
    await collect(session.open())
    assert len(llm.calls) == 4  # initial + 3 tool rounds
    assert llm.tools_seen[-1] is None  # last round forces a spoken reply
