import json
import sys
from pathlib import Path

from senjin.llm.tools import open_mcp_tools
from senjin.mcp_server.store import InterviewStore, load_questions

ROOT = Path(__file__).resolve().parents[1]


def store(tmp_path=None) -> InterviewStore:
    return InterviewStore(ROOT / "examples/candidate", ROOT / "question_bank", tmp_path)


def test_question_bank_is_valid():
    qs = load_questions(ROOT / "question_bank")
    assert len(qs) >= 9
    assert all(q.rubric and q.prompt for q in qs)


def test_questions_never_repeat_and_hide_rubric():
    s = store()
    seen = []
    while True:
        q = s.next_question(topic="algorithms")
        if "id" not in q:
            break
        assert "rubric" not in q
        seen.append(q["id"])
    assert len(seen) == len(set(seen)) == 3
    assert q["exhausted"] is False  # other topics remain
    assert s.rubric(seen[0])


def test_note_and_resume(tmp_path):
    s = store(tmp_path / "n.jsonl")
    assert "Northwind" in s.resume()
    s.record_note("strong on idempotency")
    assert json.loads((tmp_path / "n.jsonl").read_text())["note"] == "strong on idempotency"


async def test_mcp_server_over_stdio():
    async with open_mcp_tools(sys.executable, ["-m", "senjin.mcp_server.server"]) as runner:
        names = {t["function"]["name"] for t in runner.tools}
        assert names == {
            "get_candidate_resume",
            "get_job_description",
            "get_next_technical_question",
            "record_note",
        }
        assert "Northwind" in await runner.call("get_candidate_resume", {})
        q = json.loads(await runner.call("get_next_technical_question", {"difficulty": "hard"}))
        assert q["difficulty"] == "hard"


def test_question_set_limits_the_bank():
    only = load_questions(ROOT / "question_bank", ["fullstack"])
    assert only and all(q.id.startswith("fs-") for q in only)
