"""What the models must return."""

from typing import Literal

from pydantic import BaseModel, Field


class GeneratedQuestion(BaseModel):
    topic: str = Field(description="Short snake_case area, e.g. web_rendering")
    difficulty: Literal["easy", "medium", "hard"]
    requirement: str = Field(description="The requirement it tests, copied exactly from the list")
    prompt: str = Field(description="One spoken question to the candidate, second person")
    follow_ups: list[str] = Field(description="Two or three spoken follow-up questions")
    rubric: list[str] = Field(description="Three things a good answer covers (never read aloud)")


class GeneratedBank(BaseModel):
    questions: list[GeneratedQuestion]


class FileChoice(BaseModel):
    file: str = Field(description="A question file name from the list, without .yaml")
    reason: str


class FileSelection(BaseModel):
    files: list[FileChoice]


class Pick(BaseModel):
    id: str = Field(description="A question id from the list")
    relevance: str = Field(description="One short sentence: why this question suits this role")


class Ranking(BaseModel):
    selected: list[Pick] = Field(description="The chosen questions, most relevant first")


class Requirement(BaseModel):
    name: str = Field(description="Short skill or duty, e.g. Next.js rendering (SSR, SSG)")
    kind: Literal["must_have", "nice_to_have"]
    evidence: str = Field(description="The job description line it comes from")


class Requirements(BaseModel):
    requirements: list[Requirement]


class CoverageItem(BaseModel):
    requirement: str = Field(description="A requirement name copied exactly from the list")
    question_ids: list[str] = Field(description="Ids of questions that test it, or an empty list")


class CoverageMap(BaseModel):
    items: list[CoverageItem]


class ProfileChoice(BaseModel):
    profile: str = Field(description="A profile name from the list")
    reason: str = Field(description="One short sentence: why this profile fits the role")
