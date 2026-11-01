"""The shape of one candidate's interview, chosen from the job and saved with the candidate.

`senjin-review` reads the job description, picks a profile from `profiles/` (junior generalist,
senior full-stack, staff architect, ...) and writes `<candidate-dir>/interview.yaml`. The
interview commands read that file, so a staff-level role gets fewer, deeper questions and a
longer time box than a junior one, and every candidate for the same job gets the same shape.

Precedence for the interview's shape: command-line flags, then this file, then SENJIN_* settings
(which only apply to tools that read no candidate, such as `senjin-chat --no-tools`). Audio
settings (silence, hold, filler, idle) are not part of the plan: they depend on the machine.
"""

from dataclasses import dataclass, replace
from pathlib import Path

import yaml

from senjin.candidate import QUESTIONS, CandidateError
from senjin.config import InterviewConfig
from senjin.mcp_server.store import ROOT, load_questions

PLAN = "interview.yaml"
PROFILES_DIR = ROOT / "profiles"
DEFAULT_PROFILE = "default"
SENIORITIES = ("junior", "mid", "senior", "lead")

HEADER = """\
# The shape of this candidate's interview, written by senjin-review from the job description.
# Edit freely, or run senjin-review --profile NAME to switch. `minutes` is the time box for the
# whole interview; delete the line to have it computed from the question counts.
"""


class PlanError(ValueError):
    pass


@dataclass(frozen=True)
class Profile:
    name: str
    description: str
    seniority: str
    background_questions: int
    technical_questions: int
    follow_ups: int
    minutes_per_turn: float
    extra_minutes: float


@dataclass(frozen=True)
class InterviewPlan:
    profile: str
    seniority: str
    background_questions: int
    technical_questions: int
    follow_ups: int
    minutes_per_turn: float
    extra_minutes: float
    minutes: float | None = None  # None: computed from the counts
    reason: str = ""

    @property
    def max_questions(self) -> int:
        return self.background_questions + self.technical_questions

    def apply(self, base: InterviewConfig) -> InterviewConfig:
        """The plan's shape on top of the environment's audio settings."""
        return replace(
            base,
            max_questions=self.max_questions,
            resume_questions=self.background_questions,
            follow_ups=self.follow_ups,
            minutes_per_turn=self.minutes_per_turn,
            extra_minutes=self.extra_minutes,
            max_minutes=self.minutes,
        )

    def describe(self, source: Path | None = None) -> str:
        where = f"{source}, " if source else ""
        return (
            f"Interview plan ({where}profile {self.profile}, {self.seniority}): "
            f"{self.background_questions} background + {self.technical_questions} technical "
            f"questions, {self.follow_ups} follow-up(s) each."
        )


def _number(data: dict, key: str, low: float, high: float, where: str) -> float:
    try:
        value = float(data[key])
    except (KeyError, TypeError, ValueError):
        raise PlanError(f"{where}: {key} is missing or not a number") from None
    if not low <= value <= high:
        raise PlanError(f"{where}: {key} must be between {low:g} and {high:g}, got {value:g}")
    return value


def _shape(data: dict, where: str) -> dict:
    background = int(_number(data, "background_questions", 1, 29, where))
    technical = int(_number(data, "technical_questions", 1, 29, where))
    if not 2 <= background + technical <= 30:
        raise PlanError(f"{where}: background + technical questions must be 2 to 30")
    seniority = str(data.get("seniority", ""))
    if seniority not in SENIORITIES:
        raise PlanError(f"{where}: seniority must be one of {', '.join(SENIORITIES)}")
    return {
        "seniority": seniority,
        "background_questions": background,
        "technical_questions": technical,
        "follow_ups": int(_number(data, "follow_ups", 0, 3, where)),
        "minutes_per_turn": _number(data, "minutes_per_turn", 0.5, 10, where),
        "extra_minutes": _number(data, "extra_minutes", 0, 60, where),
    }


def load_profiles(directory: Path = PROFILES_DIR) -> dict[str, Profile]:
    profiles: dict[str, Profile] = {}
    for path in sorted(directory.glob("*.yaml")):
        where = path.name
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            shape = _shape(data, where)
            description = str(data["description"]).strip()
        except (KeyError, AttributeError, yaml.YAMLError) as exc:
            raise PlanError(f"{where}: not a valid profile ({exc})") from None
        profiles[path.stem] = Profile(name=path.stem, description=description, **shape)
    if DEFAULT_PROFILE not in profiles:
        raise PlanError(f"{directory}/{DEFAULT_PROFILE}.yaml is missing")
    return profiles


def plan_from_profile(profile: Profile, reason: str = "") -> InterviewPlan:
    return InterviewPlan(
        profile=profile.name,
        seniority=profile.seniority,
        background_questions=profile.background_questions,
        technical_questions=profile.technical_questions,
        follow_ups=profile.follow_ups,
        minutes_per_turn=profile.minutes_per_turn,
        extra_minutes=profile.extra_minutes,
        reason=reason,
    )


def save_plan(path: Path, plan: InterviewPlan) -> None:
    data: dict = {
        "profile": plan.profile,
        "seniority": plan.seniority,
        "background_questions": plan.background_questions,
        "technical_questions": plan.technical_questions,
        "follow_ups": plan.follow_ups,
        "minutes_per_turn": plan.minutes_per_turn,
        "extra_minutes": plan.extra_minutes,
    }
    if plan.minutes is not None:
        data["minutes"] = plan.minutes
    if plan.reason:
        data["reason"] = plan.reason
    path.write_text(HEADER + yaml.safe_dump(data, sort_keys=False, width=100), encoding="utf-8")


def load_plan(path: Path) -> InterviewPlan:
    if not path.is_file():
        raise PlanError(
            f"{path} not found: run `senjin-review --candidate-dir {path.parent}` first"
        )
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        shape = _shape(data, path.name)
        minutes = _number(data, "minutes", 5, 600, path.name) if "minutes" in data else None
        return InterviewPlan(
            profile=str(data.get("profile", "custom")),
            minutes=minutes,
            reason=str(data.get("reason", "")),
            **shape,
        )
    except (AttributeError, yaml.YAMLError) as exc:
        raise PlanError(f"{path} is not a valid interview plan: {exc}") from None


def plan_for_interview(candidate_dir: Path) -> InterviewPlan:
    """The plan an interview will follow, checked against the questions that were written for it."""
    plan = load_plan(candidate_dir / PLAN)
    questions = len(load_questions(candidate_dir, [Path(QUESTIONS).stem]))
    if questions < plan.technical_questions:
        raise PlanError(
            f"{candidate_dir / QUESTIONS} has {questions} questions but the plan asks for "
            f"{plan.technical_questions}: run `senjin-review --candidate-dir {candidate_dir} "
            "--force` again"
        )
    return plan


def plan_from_args(candidate_dir: Path) -> InterviewPlan:
    """For command-line entry points: a missing or inconsistent plan ends with one error line."""
    try:
        return plan_for_interview(candidate_dir)
    except (PlanError, CandidateError) as exc:
        raise SystemExit(f"error: {exc}") from None
