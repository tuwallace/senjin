import json
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
    async with open_mcp_tools(
        ROOT / "examples/candidate", bank_dir=ROOT / "question_bank"
    ) as runner:
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


async def test_mcp_server_asks_from_the_candidates_questions_file(tmp_path):
    candidate = tmp_path / "c"
    candidate.mkdir()
    (candidate / "resume.md").write_text("Resume of Pat")
    (candidate / "job_description.md").write_text("A job")
    (candidate / "questions.yaml").write_text(
        "- id: gen-01-x\n  topic: x\n  difficulty: easy\n  prompt: Only question?\n"
        '  follow_ups: ["Why?"]\n  rubric: [a, b]\n  basis: because\n  origin: generated\n'
    )
    async with open_mcp_tools(candidate) as runner:
        q = json.loads(await runner.call("get_next_technical_question", {}))
        assert q["id"] == "gen-01-x" and q["prompt"] == "Only question?"
        assert "basis" not in q and "origin" not in q and "rubric" not in q
        assert "Pat" in await runner.call("get_candidate_resume", {})


def test_server_stops_with_a_clear_message_without_the_folder_or_questions(tmp_path):
    import subprocess
    import sys

    def run(*args):
        return subprocess.run(
            [sys.executable, "-m", "senjin.mcp_server.server", *args],
            capture_output=True,
            text=True,
            timeout=60,
        )

    assert "--candidate-dir is required" in run().stderr
    assert "candidate folder not found" in run("--candidate-dir", str(tmp_path / "nope")).stderr
    assert "senjin-review" in run("--candidate-dir", str(tmp_path)).stderr


def test_every_question_file_is_text_only_and_a_colon_line_is_rejected(tmp_path):
    import pytest

    for q in load_questions(ROOT / "question_bank"):
        assert all(isinstance(t, str) for t in [*q.follow_ups, *q.rubric]), q.id
    (tmp_path / "bad.yaml").write_text(
        '- id: x\n  topic: t\n  difficulty: easy\n  prompt: Q?\n  follow_ups: ["a?"]\n'
        "  rubric:\n    - CI stages: lint, tests\n"
    )
    with pytest.raises(ValueError, match="put the whole line in quotes"):
        load_questions(tmp_path)
