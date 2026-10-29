"""Assessment output schema. Every claim must cite a quote from the transcript."""

from typing import Literal

from pydantic import BaseModel, Field

Score = Field(ge=1, le=10)


class Evidence(BaseModel):
    turn: int = Field(description="Index of the candidate turn the quote comes from")
    quote: str = Field(description="Exact words copied from that turn")


class Finding(BaseModel):
    text: str
    evidence: list[Evidence] = Field(min_length=1)


class QuestionScore(BaseModel):
    question_id: str
    score: int = Score
    rationale: str
    evidence: list[Evidence] = Field(default_factory=list)


class Assessment(BaseModel):
    technical_score: int = Score
    communication_score: int = Score
    strengths: list[Finding]
    weaknesses: list[Finding]
    question_scores: list[QuestionScore] = Field(default_factory=list)
    hire_recommendation: Literal["Hire", "No Hire"]
    confidence: Literal["low", "medium", "high"]
    summary: str


class Verification(BaseModel):
    dropped_findings: int = 0
    dropped_evidence: int = 0
    repaired: bool = False


class Report(BaseModel):
    assessment: Assessment
    verification: Verification
    judge_model: str
    transcript: str
    redacted: bool = True
