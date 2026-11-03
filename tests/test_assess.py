import json
from pathlib import Path

import pytest

from senjin.assess.judge import (
    AssessmentError,
    asked_question_ids,
    assess,
    load_rows,
    render_transcript,
    to_markdown,
)
from senjin.assess.schema import Assessment

ROOT = Path(__file__).resolve().parents[1]
STRONG = ROOT / "examples/transcripts/strong.jsonl"


class FakeJudge:
    model = "fake-judge"

    def __init__(self, *replies: str) -> None:
        self.replies = list(replies)
        self.prompts: list[str] = []

    async def complete_json(self, system, user, schema):
        self.prompts.append(user)
        return self.replies.pop(0)


def payload(**over) -> str:
    base = {
        "technical_score": 8,
        "communication_score": 8,
        "strengths": [
            {
                "text": "Chooses a concrete algorithm",
                "evidence": [{"turn": 5, "quote": "I would use a token bucket per customer key"}],
            }
        ],
        "weaknesses": [],
        "question_scores": [],
        "hire_recommendation": "Hire",
        "confidence": "high",
        "summary": "Strong.",
    }
    base.update(over)
    return json.dumps(base)


def test_render_skips_tool_rows_and_numbers_turns():
    rows = load_rows(STRONG)
    text, candidate = render_transcript(rows)
    assert text.startswith("[0] Interviewer:")
    assert "get_next_technical_question" not in text
    assert candidate[5].startswith("I would use a token bucket")
    assert asked_question_ids(rows) == ["be-rate-limiter"]


async def test_valid_assessment_passes_verification():
    report = await assess(load_rows(STRONG), FakeJudge(payload()))
    assert report.assessment.confidence == "high"
    assert report.verification.dropped_findings == 0
    assert "Strengths" in to_markdown(report)


async def test_quote_in_wrong_turn_is_relocated():
    bad_turn = payload(
        strengths=[
            {"text": "x", "evidence": [{"turn": 9, "quote": "a token bucket per customer key"}]}
        ]
    )
    report = await assess(load_rows(STRONG), FakeJudge(bad_turn))
    assert report.assessment.strengths[0].evidence[0].turn == 5


async def test_a_quote_that_differs_only_in_punctuation_still_verifies():
    from senjin.assess.verify import verify

    spoken = {5: "You can use a vault, and pull the value, translate the key to the value."}
    from senjin.assess.schema import Assessment

    def with_quote(quote):
        return Assessment.model_validate_json(
            payload(strengths=[{"text": "x", "evidence": [{"turn": 5, "quote": quote}]}])
        )

    kept, info = verify(with_quote("use a vault and pull the value."), spoken)
    assert info.dropped_findings == 0 and kept.strengths  # full stop for a comma
    _, info = verify(with_quote("Use a vault, and PULL the value"), spoken)
    assert info.dropped_findings == 0
    _, info = verify(with_quote("pull the value from a vault"), spoken)  # words out of order
    assert info.dropped_findings == 1
    _, info = verify(with_quote("use a database and pull the value"), spoken)  # a changed word
    assert info.dropped_findings == 1


async def test_invented_quote_is_dropped_and_confidence_lowered():
    mixed = payload(
        strengths=[
            {
                "text": "real",
                "evidence": [{"turn": 5, "quote": "I would use a token bucket per customer key"}],
            },
            {
                "text": "invented",
                "evidence": [{"turn": 5, "quote": "I have a PhD in distributed systems"}],
            },
        ]
    )
    report = await assess(load_rows(STRONG), FakeJudge(mixed))
    assert [f.text for f in report.assessment.strengths] == ["real"]
    assert report.verification.dropped_findings == 1
    assert report.assessment.confidence == "low"


async def test_everything_invented_fails_loudly():
    fake = payload(
        strengths=[{"text": "x", "evidence": [{"turn": 5, "quote": "completely made up quote"}]}]
    )
    with pytest.raises(AssessmentError):
        await assess(load_rows(STRONG), FakeJudge(fake))


async def test_invalid_json_gets_one_repair_retry():
    judge = FakeJudge("not json at all", payload())
    report = await assess(load_rows(STRONG), judge)
    assert report.verification.repaired
    assert "previous answer was invalid" in judge.prompts[1]


async def test_two_invalid_answers_raise():
    with pytest.raises(AssessmentError):
        await assess(load_rows(STRONG), FakeJudge("{}", '{"technical_score": 99}'))


async def test_rubrics_are_in_prompt_and_schema_bounds_hold():
    judge = FakeJudge(payload())
    await assess(load_rows(STRONG), judge, {"be-rate-limiter": ["names an algorithm"]})
    assert "names an algorithm" in judge.prompts[0]
    with pytest.raises(ValueError):
        Assessment.model_validate_json(payload(technical_score=11))


async def test_hiring_bar_turns_a_lenient_hire_into_no_hire_and_says_why():
    judge = FakeJudge(
        payload(
            technical_score=6,
            question_scores=[
                {
                    "question_id": "be-rate-limiter",
                    "score": 3,
                    "rationale": "Vague.",
                    "evidence": [{"turn": 5, "quote": "I would use a token bucket"}],
                }
            ],
        )
    )
    report = await assess(load_rows(STRONG), judge)
    assert report.judge_recommendation == "Hire"
    assert report.assessment.hire_recommendation == "No Hire"
    assert any("technical 6" in n for n in report.bar_notes)
    assert any("be-rate-limiter" in n for n in report.bar_notes)
    md = to_markdown(report)
    assert "The judge said Hire; changed to No Hire" in md and "**be-rate-limiter**: 3/10" in md


async def test_hiring_bar_leaves_a_strong_hire_alone():
    report = await assess(load_rows(STRONG), FakeJudge(payload(technical_score=8)))
    assert report.assessment.hire_recommendation == "Hire" and report.bar_notes == []
    assert "did not change it" in to_markdown(report)
