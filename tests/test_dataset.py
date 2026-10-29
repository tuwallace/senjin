import json

from senjin.eval.dataset import classify_move, extract_moves, parse_turns
from senjin.eval.style import measure
from senjin.interview.exemplars import Exemplar, ExemplarIndex, good_exemplar
from senjin.interview.session import InterviewSession
from senjin.interview.state import Phase
from senjin.llm.fake import FakeLLM

RAW = """Assistant: Hi there! I'm Claude. Ready?

User: Yes.

AI: Tell me about how AI fits into your typical workday?

User: I mostly use it for drafting sales emails and summarizing call notes.

AI: When you say summarizing call notes, what does that look like in practice?

User: I paste the transcript and ask for action items.

AI: It sounds like that saves you real time. How do you check the results?

User: I skim them.

AI: Thank you so much for your time today!
"""


def test_parse_turns_roles_and_text():
    turns = parse_turns(RAW)
    assert [t.role for t in turns][:3] == ["interviewer", "participant", "interviewer"]
    assert turns[0].text.startswith("Hi there")
    assert len(turns) == 9


def test_extract_moves_classifies_kinds_and_context():
    moves = extract_moves("t1", "workforce", parse_turns(RAW))
    assert [m.kind for m in moves] == ["opener", "probe", "clarify", "reflect", "wrap_up"]
    assert moves[2].context.startswith("I mostly use it")
    assert moves[0].context == ""


def test_classify_transition_and_last():
    assert classify_move("Let's move on to your tools.", 3, False) == "transition"
    assert classify_move("Anything?", 3, True) == "wrap_up"


def test_good_exemplar_filter():
    moves = extract_moves("t1", "workforce", parse_turns(RAW))
    assert good_exemplar(moves[2])  # clarify, one open question, real context
    assert not good_exemplar(moves[4])  # wrap-up


def test_style_metrics():
    pairs = [
        ("How did you test the migration?", "We migrated the ledger with dual writes."),
        ("Did you test it? Was it hard?", "We migrated the ledger."),
        ("**Great.** What next?", "unrelated"),
    ]
    s = measure(pairs)
    assert s.n == 3
    assert abs(s.single_question - 2 / 3) < 1e-9
    assert abs(s.open_ended - 2 / 3) < 1e-9
    assert abs(s.speakable - 2 / 3) < 1e-9
    assert abs(s.ties_to_answer - 1 / 3) < 1e-9


async def test_index_retrieves_nearest_for_probing_phases_only():
    ex = [
        Exemplar("probe", "about cats", "Tell me more about cats?", [1.0, 0.0]),
        Exemplar("probe", "about dogs", "Tell me more about dogs?", [0.0, 1.0]),
        Exemplar("reflect", "x", "y", [0.5, 0.5]),
    ]
    index = ExemplarIndex(ex, k=1, embed=lambda texts: [[0.9, 0.1] for _ in texts])
    got = await index.examples("my cat", Phase.TECHNICAL)
    assert len(got) == 1 and "cats" in got[0]
    assert await index.examples("hi", Phase.INTRO) == []
    assert await index.examples("hi", Phase.CANDIDATE_QA) == []


async def test_session_injects_exemplars_into_system_message(tmp_path):
    class Fixed:
        async def examples(self, text, phase):
            return ['Participant: "a"\nInterviewer: "b"'] if phase == Phase.RESUME_DIVE else []

    from senjin.interview.state import Budget, InterviewState

    budgets = {p: Budget(1, 1e9) for p in Phase}
    llm = FakeLLM([])
    session = InterviewSession(llm, state=InterviewState(budgets=budgets), exemplars=Fixed())
    async for _ in session.open():
        pass
    async for _ in session.respond("ready"):
        pass
    assert "skilled interviewers" not in llm.calls[0][0]["content"]  # intro: none
    assert "skilled interviewers" in llm.calls[1][0]["content"]
    assert 'Interviewer: "b"' in llm.calls[1][0]["content"]


def test_exemplar_file_roundtrip(tmp_path):
    path = tmp_path / "m.jsonl"
    row = {"kind": "probe", "context": "c", "text": "t?", "embedding": [0.1, 0.2]}
    path.write_text(json.dumps(row) + "\n")
    assert len(ExemplarIndex.load(path).exemplars) == 1
