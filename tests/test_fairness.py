"""Offline fairness guarantees. Run alone with `pytest -m fairness`."""

from pathlib import Path

import pytest

from senjin.assess.judge import load_rows, render_transcript
from senjin.assess.redact import candidate_names, redact, redact_rows
from senjin.eval.fairness import VARIANTS, make_variant

pytestmark = pytest.mark.fairness
ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("base", ["strong", "mediocre", "weak", "evasive"])
def test_redacted_variants_are_identical(base):
    rows = load_rows(ROOT / f"examples/transcripts/{base}.jsonl")
    seen = {}
    for label, (name, cue) in VARIANTS.items():
        variant = redact_rows(make_variant(rows, name, cue), [name])
        seen[label] = render_transcript(variant)[0]
    leaked = {k for k, v in seen.items() if v != seen["control"]}
    assert not leaked, f"cues survived redaction in: {sorted(leaked)}"


def test_unredacted_variants_do_differ():
    rows = load_rows(ROOT / "examples/transcripts/strong.jsonl")
    texts = {render_transcript(make_variant(rows, n, c))[0] for n, c in VARIANTS.values()}
    assert len(texts) == len(VARIANTS)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("I'm a mother of three. I built a ledger in Go.", "I built a ledger in Go."),
        ("My wife and I moved. We use Kafka.", "We use Kafka."),
        ("I'm 58 years old. I like Redis.", "I like Redis."),
        ("I'm originally from Lagos. Kafka is great.", "Kafka is great."),
        (
            "My manager said she liked it, so he approved.",
            "My manager said they liked it, so they approved.",
        ),
        ("Hello, my name is Jamal Washington. I use Go.", "Hello, my name is [name]. I use Go."),
        (
            "I have 10 years of experience and ran 58 percent faster.",
            "I have 10 years of experience and ran 58 percent faster.",
        ),
        ("I'm ready to start.", "I'm ready to start."),
    ],
)
def test_redact_patterns(text, expected):
    assert redact(text) == expected


def test_known_names_are_masked_everywhere():
    assert redact("Jane led it, said Doe.", ["Jane Doe"]) == "[name] led it, said [name]."


def test_candidate_name_from_resume_heading():
    assert candidate_names("# Jane Doe (sample candidate)\n\nbody") == ["Jane Doe"]
    assert candidate_names("") == []


def test_redaction_leaves_interviewer_text_structure_intact():
    rows = [{"role": "assistant", "text": "Tell me about it?"}, {"role": "tool", "text": "x"}]
    assert redact_rows(rows) == rows
