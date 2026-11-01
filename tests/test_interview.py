import pytest

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


async def test_prompt_shape_and_stable_system_prefix_within_a_phase():
    llm = FakeLLM([])
    budgets = one_turn_budgets()
    budgets[Phase.RESUME_DIVE] = Budget(max_turns=2, max_seconds=1e9)
    session = InterviewSession(llm, state=InterviewState(budgets=budgets))
    await collect(session.open())
    await collect(session.respond("ready"))  # -> resume_dive
    await collect(session.respond("I built a ledger"))  # still resume_dive
    first, second, third = llm.calls
    assert first[0]["role"] == "system" and "Senjin" in first[0]["content"]
    assert first[-1]["role"] == "user"
    assert [m["role"] for m in second].count("system") == 1
    assert first[0]["content"] != second[0]["content"]  # phase changed
    assert second[0]["content"] == third[0]["content"]  # same phase: cacheable prefix
    assert third[-1] == {"role": "user", "content": "I built a ledger"}


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


QUESTION = (
    '{"id": "q1", "topic": "system_design", "difficulty": "medium", '
    '"prompt": "How would you design a rate limiter?", "follow_ups": ["FU-one?", "FU-two?"]}'
)
QUESTION2 = (
    '{"id": "q2", "topic": "databases", "difficulty": "easy", '
    '"prompt": "Second topic?", "follow_ups": []}'
)


class OrchestratedTools(StubTools):
    def __init__(self, questions=(QUESTION, QUESTION2)):
        super().__init__()
        self.questions = list(questions)

    async def call(self, name, arguments):
        self.calls.append((name, arguments))
        if name == "get_next_technical_question":
            return self.questions.pop(0) if self.questions else '{"exhausted": true}'
        return f"RESULT:{name}"


def orchestrated(llm, tools=None, **budget):
    budgets = {p: Budget(max_turns=budget.get(p.value, 1), max_seconds=1e9) for p in Phase}
    return InterviewSession(
        llm,
        state=InterviewState(budgets=budgets, follow_ups_per_question=None),
        tools=tools or OrchestratedTools(),
        prefetch=True,
    )


def last_user(call):
    return call[-1]["content"]


async def test_orchestrated_resume_goes_in_system_message_not_history():
    llm = FakeLLM([])
    session = orchestrated(llm)
    await collect(session.open())
    await collect(session.respond("ready"))
    messages = llm.calls[-1]
    assert "RESULT:get_candidate_resume" in messages[0]["content"]
    assert all(m["role"] in {"user", "assistant"} for m in messages[1:])  # no tool messages
    assert llm.tools_seen[-1] is None  # the model gets no tools: no extra round trips
    assert len(llm.calls) == 2


async def test_technical_asks_question_then_follow_ups_once_each_then_next_question():
    llm = FakeLLM([])
    tools = OrchestratedTools()
    session = orchestrated(llm, tools, technical=6)
    await collect(session.open())
    await collect(session.respond("ready"))  # -> resume_dive
    await collect(session.respond("my project"))  # -> technical, asks the question
    assert "How would you design a rate limiter?" in last_user(llm.calls[-1])
    assert "FU-one" not in str(llm.calls[-1])  # follow-ups are not leaked to the model
    await collect(session.respond("answer 1"))
    assert "FU-one?" in last_user(llm.calls[-1])
    await collect(session.respond("answer 2"))
    assert "FU-two?" in last_user(llm.calls[-1])
    await collect(session.respond("answer 3"))  # topic exhausted: next question
    assert "Second topic?" in last_user(llm.calls[-1])
    notes = [m["content"] for call in llm.calls for m in call if "Interviewer note" in m["content"]]
    assert sum("FU-one?" in n for n in notes) == len({n for n in notes if "FU-one?" in n}) >= 1
    await collect(session.respond("answer 4"))  # bank exhausted: probe
    assert "probing follow-up" in last_user(llm.calls[-1])
    # the note is never stored in history
    assert all("Interviewer note" not in m["content"] for m in session.history)


async def test_candidate_qa_uses_job_description_and_ends_when_candidate_is_done():
    llm = FakeLLM([])
    session = orchestrated(llm)
    await collect(session.open())
    for text in ["ready", "project", "answer"]:
        await collect(session.respond(text))
    assert session.state.phase == Phase.CANDIDATE_QA
    assert "RESULT:get_job_description" in llm.calls[-1][0]["content"]
    assert "invite any questions" in last_user(llm.calls[-1])
    await collect(session.respond("No, I'm good."))
    assert session.state.phase == Phase.SIGNOFF
    assert "goodbye" in last_user(llm.calls[-1]) and "Do not ask a question" in last_user(
        llm.calls[-1]
    )
    assert session.finished


async def test_qa_question_is_answered_not_treated_as_done():
    llm = FakeLLM([])
    session = orchestrated(llm, candidate_qa=3)
    await collect(session.open())
    for text in ["ready", "project", "answer"]:
        await collect(session.respond(text))
    await collect(session.respond("What does on-call look like?"))
    assert session.state.phase == Phase.CANDIDATE_QA
    assert "Answer their question" in last_user(llm.calls[-1])


async def test_orchestration_is_off_by_default():
    tools = OrchestratedTools()
    llm = FakeLLM([])
    session = InterviewSession(llm, state=InterviewState(budgets=one_turn_budgets()), tools=tools)
    await collect(session.open())
    await collect(session.respond("ready"))
    assert tools.calls == []
    assert llm.tools_seen[-1] == tools.tools  # model-driven mode still offers tools


async def test_cut_off_goodbye_does_not_finish_the_interview():
    state = InterviewState(budgets=one_turn_budgets(), phase=Phase.SIGNOFF)
    session = InterviewSession(FakeLLM(["Thank you for your time today and goodbye."]), state=state)
    stream = session.respond("thanks")
    assert (await stream.__anext__()).strip()  # the goodbye begins...
    await stream.aclose()  # ...and is cancelled before it ends
    assert not session.finished
    complete = InterviewSession(FakeLLM(["Goodbye."]), state=InterviewState(phase=Phase.SIGNOFF))
    await collect(complete.open())
    assert complete.finished


async def test_qa_note_forbids_inventing_facts():
    from senjin.interview.prompts import NOTES

    assert "never invent" in NOTES["qa_answer"] and "salary" in NOTES["qa_answer"]


async def test_interrupted_unspoken_reply_does_not_use_up_a_question_or_a_turn():
    llm = FakeLLM([])
    tools = OrchestratedTools()
    session = orchestrated(llm, tools, technical=6)
    await collect(session.open())
    await collect(session.respond("ready"))
    await collect(session.respond("my project"))  # technical: the question is asked
    await collect(session.respond("answer 1"))  # follow-up 1 is asked
    turns = session.state.turns_in_phase
    assert "FU-one?" in last_user(llm.calls[-1])
    session.interrupt("")  # the candidate cut in before any of it was spoken
    assert session.state.turns_in_phase == turns - 1
    assert session.history[-1]["role"] == "user"
    await collect(session.respond("answer 1 again"))
    assert "FU-one?" in last_user(llm.calls[-1])  # asked again, not skipped
    session.interrupt("Okay, so")  # partly spoken: it counts
    assert session.history[-1]["content"] == "Okay, so"


async def test_empty_model_reply_is_not_counted_as_a_turn():
    class Silent(FakeLLM):
        async def stream(self, messages, tools=None):
            return
            yield

    session = InterviewSession(Silent([]))
    await collect(session.open())
    assert session.state.turns_in_phase == 0


async def test_orchestrated_system_prompt_never_changes_between_phases():
    llm = FakeLLM([])
    session = orchestrated(llm, technical=2, candidate_qa=2)
    await collect(session.open())
    for text in ["ready", "my project", "answer", "answer 2", "What stack?"]:
        await collect(session.respond(text))
    systems = {call[0]["content"] for call in llm.calls}
    assert len(systems) == 1  # resume and job description are in it from the start
    assert "RESULT:get_candidate_resume" in systems.copy().pop()
    assert "RESULT:get_job_description" in systems.copy().pop()
    assert "Phase: technical" in str([call[-1]["content"] for call in llm.calls])


async def test_qa_note_carries_matching_job_description_lines():
    class JobTools(OrchestratedTools):
        async def call(self, name, arguments):
            if name == "get_job_description":
                return (
                    "Intro line about the company\n- Deploy to AWS GovCloud or Azure Government\n"
                )
            return await super().call(name, arguments)

    llm = FakeLLM([])
    session = orchestrated(llm, JobTools(), candidate_qa=3)
    await collect(session.open())
    for text in ["ready", "project", "answer"]:
        await collect(session.respond(text))
    await collect(session.respond("Is the solution Google-based or AWS?"))
    assert "Deploy to AWS GovCloud" in last_user(llm.calls[-1])


async def test_opening_praise_is_not_spoken_or_recorded():
    llm = FakeLLM(["Great! Let's start with your background. Tell me about a project."])
    session = InterviewSession(llm)
    spoken = await collect(session.open())
    assert spoken.startswith("Let's start")
    assert session.history[-1]["content"].startswith("Let's start")


async def test_repeat_request_asks_the_same_question_again_without_using_a_turn():
    llm = FakeLLM([])
    session = orchestrated(llm, technical=6)
    await collect(session.open())
    await collect(session.respond("ready"))
    await collect(session.respond("my project"))  # technical: the question is asked
    turns = session.state.turns_in_phase
    await collect(session.respond("Can you repeat the question?"))
    assert "repeat or clarify" in last_user(llm.calls[-1])
    assert "How would you design a rate limiter?" in last_user(llm.calls[-1])
    assert session.state.turns_in_phase == turns  # not counted
    await collect(session.respond("answer 1"))
    assert "FU-one?" in last_user(llm.calls[-1])  # the follow-up was not skipped or spent


async def test_a_long_answer_mentioning_repeat_is_still_an_answer():
    llm = FakeLLM([])
    session = orchestrated(llm, technical=6)
    await collect(session.open())
    await collect(session.respond("ready"))
    await collect(session.respond("my project"))
    answer = (
        "I would retry and repeat the request with backoff, then queue it if the store stays down"
    )
    await collect(session.respond(answer))
    assert "repeat or clarify" not in last_user(llm.calls[-1])


async def test_background_follow_ups_rotate_through_angles_and_never_ask_which_project():
    llm = FakeLLM([])
    session = orchestrated(llm, resume_dive=4)
    await collect(session.open())
    await collect(session.respond("ready"))  # -> resume_dive, asks for a project
    await collect(session.respond("I built a ledger"))
    first = last_user(llm.calls[-1])
    await collect(session.respond("it was event sourced"))
    second = last_user(llm.calls[-1])
    assert "decision" in first and "trade-off" in second
    assert "Do not ask which project they mean" in first


def test_retracted_fragment_is_dropped_from_the_judged_transcript():
    from senjin.assess.judge import render_transcript

    rows = [
        {"role": "assistant", "text": "Q?", "phase": "technical"},
        {"role": "user", "text": "I would set a timer", "phase": "technical"},
        {"role": "retract", "text": "", "phase": "technical"},
        {"role": "user", "text": "I would set a timer or push", "phase": "technical"},
    ]
    dialogue, candidate = render_transcript(rows)
    assert list(candidate.values()) == ["I would set a timer or push"]
    assert "[1] Candidate: I would set a timer or push" in dialogue


async def test_praise_in_later_sentences_is_also_removed():
    llm = FakeLLM(
        ["The technical part is done. Thank you for your detailed responses. Any questions?"]
    )
    session = InterviewSession(llm)
    spoken = await collect(session.open())
    assert "detailed" not in spoken and spoken.endswith("Any questions?")
    assert "detailed" not in session.history[-1]["content"]


async def test_warm_up_loads_context_and_sends_the_fixed_prompt_before_the_call():
    llm = FakeLLM([])
    session = orchestrated(llm)
    await session.warm()
    assert len(llm.warmed) == 1 and llm.calls == []  # a warm-up is not a turn
    system = llm.warmed[0][0]["content"]
    assert "RESULT:get_candidate_resume" in system and "RESULT:get_job_description" in system
    await collect(session.open())
    assert llm.calls[0][0]["content"] == system  # the real prompt starts the same way


def test_prompts_no_time_of_day_greeting_and_yes_to_listed_technologies():
    from senjin.interview.prompts import NOTES, PHASE_INSTRUCTIONS

    intro = PHASE_INSTRUCTIONS[Phase.INTRO]
    assert "never say good morning" in intro
    assert "FedRAMP" in NOTES["qa_answer"] and "say yes" in NOTES["qa_answer"]


@pytest.mark.parametrize(
    "said",
    [
        "I don't have any at the moment.",
        "I do not have any other questions and I meet all that criteria.",
        "No questions from me.",
        "I have no more questions.",
        "Nope, I'm all set.",
    ],
)
async def test_candidate_who_has_no_questions_ends_the_interview(said):
    llm = FakeLLM([])
    session = orchestrated(llm, candidate_qa=3)
    await collect(session.open())
    for text in ["ready", "project", "answer"]:
        await collect(session.respond(text))
    assert session.state.phase == Phase.CANDIDATE_QA
    await collect(session.respond(said))
    assert session.state.phase == Phase.SIGNOFF


@pytest.mark.parametrize(
    "said",
    [
        "I don't have any questions about pay, but what is the team size?",
        "What does on-call look like?",
        "Do you know whether it is remote?",
    ],
)
async def test_candidate_who_still_asks_is_answered(said):
    llm = FakeLLM([])
    session = orchestrated(llm, candidate_qa=3)
    await collect(session.open())
    for text in ["ready", "project", "answer"]:
        await collect(session.respond(text))
    await collect(session.respond(said))
    assert session.state.phase == Phase.CANDIDATE_QA


@pytest.mark.parametrize(
    "said",
    [
        "Thank you for the detailed response. I have no more questions and appreciate the "
        "opportunity to learn more about the role and team. Have a great day!",
        "That was really helpful, thank you so much for your time today. I don't have any more "
        "questions, and I look forward to hearing from the team.",
        "No, that's everything for me. Thanks for walking me through all of it, it was a pleasure.",
    ],
)
async def test_a_long_polite_sign_off_ends_the_candidate_qa(said):
    llm = FakeLLM([])
    session = orchestrated(llm, candidate_qa=3)
    await collect(session.open())
    for text in ["ready", "project", "answer"]:
        await collect(session.respond(text))
    assert session.state.phase == Phase.CANDIDATE_QA
    await collect(session.respond(said))
    assert session.state.phase == Phase.SIGNOFF


@pytest.mark.parametrize(
    "said",
    [
        "Thank you. I have one question: how does the team approach deployment? "
        "I have no more questions after that.",
        "I have no more questions about pay, but it would help to know how big the team is.",
        "Thanks, that is helpful. Also, I wonder about the on-call rotation and how often it is.",
    ],
)
async def test_a_candidate_who_still_asks_something_is_not_cut_off(said):
    llm = FakeLLM([])
    session = orchestrated(llm, candidate_qa=3)
    await collect(session.open())
    for text in ["ready", "project", "answer"]:
        await collect(session.respond(text))
    await collect(session.respond(said))
    assert session.state.phase == Phase.CANDIDATE_QA
