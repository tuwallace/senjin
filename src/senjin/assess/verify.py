"""Anti-hallucination check: keep only claims whose quotes occur in the candidate's words."""

import re

from senjin.assess.schema import Assessment, Evidence, Finding, Verification


def _norm(text: str) -> str:
    """Words only: case, spacing and punctuation differences (a quote that ends in a full stop
    where the speaker's sentence goes on with a comma) do not make a real quote unverifiable.
    The words themselves must still appear in the same order."""
    text = text.replace("\u2019", "'").lower()
    return re.sub(r"\s+", " ", re.sub(r"[^\w']+", " ", text)).strip()


def _locate(ev: Evidence, turns: dict[int, str]) -> int | None:
    """Turn index holding the quote: the cited one first, then any other candidate turn."""
    quote = _norm(ev.quote)
    if len(quote) < 8:  # too short to be evidence
        return None
    if ev.turn in turns and quote in _norm(turns[ev.turn]):
        return ev.turn
    return next((i for i, t in turns.items() if quote in _norm(t)), None)


def verify(
    assessment: Assessment, candidate_turns: dict[int, str]
) -> tuple[Assessment, Verification]:
    info = Verification()

    def clean_evidence(items: list[Evidence]) -> list[Evidence]:
        kept = []
        for ev in items:
            turn = _locate(ev, candidate_turns)
            if turn is None:
                info.dropped_evidence += 1
            else:
                kept.append(Evidence(turn=turn, quote=ev.quote))
        return kept

    def clean_findings(findings: list[Finding]) -> list[Finding]:
        kept = []
        for f in findings:
            evidence = clean_evidence(f.evidence)
            if evidence:
                kept.append(Finding(text=f.text, evidence=evidence))
            else:
                info.dropped_findings += 1
        return kept

    result = assessment.model_copy(
        update={
            "strengths": clean_findings(assessment.strengths),
            "weaknesses": clean_findings(assessment.weaknesses),
            "question_scores": [
                q.model_copy(update={"evidence": clean_evidence(q.evidence)})
                for q in assessment.question_scores
            ],
        }
    )
    if info.dropped_findings or info.dropped_evidence:
        result = result.model_copy(update={"confidence": "low"})
    return result, info
