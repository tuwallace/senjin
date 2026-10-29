"""Interview limits, managed through environment variables.

SENJIN_MAX_QUESTIONS     questions asked in the background + technical parts (default 5, 2 to 30)
SENJIN_RESUME_QUESTIONS  how many of those are about the resume (default: 40%, at least 1)
SENJIN_QA_TURNS          replies reserved for the candidate's own questions (default 3, 1 to 10)
SENJIN_MAX_MINUTES       time box for the whole interview (default 45)
SENJIN_IDLE_SECONDS      silence before "Are you still there?" (default 25, 0 turns it off)

The greeting, the candidate Q&A and the goodbye are not counted as interview questions.
"""

import math
import os
from collections.abc import Mapping
from dataclasses import dataclass, replace

from senjin.interview.state import DEFAULT_BUDGETS, Budget, InterviewState, Phase


class ConfigError(ValueError):
    pass


def exit_on_config_error(main):
    """Decorator for CLI entry points: a bad SENJIN_* value ends with one clear line."""
    import functools

    @functools.wraps(main)
    def wrapper(*args, **kwargs):
        try:
            return main(*args, **kwargs)
        except ConfigError as exc:
            raise SystemExit(f"error: {exc}") from None

    return wrapper


def _number(env: Mapping[str, str], name: str, default: float, low: float, high: float) -> float:
    raw = env.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = float(raw)
    except ValueError:
        raise ConfigError(f"{name} must be a number, got {raw!r}") from None
    if not low <= value <= high:
        raise ConfigError(f"{name} must be between {low:g} and {high:g}, got {raw!r}")
    return value


@dataclass(frozen=True)
class InterviewConfig:
    max_questions: int = 5
    resume_questions: int | None = None  # None: derive from max_questions
    qa_turns: int = 3
    max_minutes: float = 45.0
    idle_seconds: float = 25.0

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "InterviewConfig":
        env = os.environ if env is None else env
        questions = int(_number(env, "SENJIN_MAX_QUESTIONS", 5, 2, 30))
        resume = None
        if env.get("SENJIN_RESUME_QUESTIONS", "").strip():
            resume = int(_number(env, "SENJIN_RESUME_QUESTIONS", 0, 1, questions - 1))
        return cls(
            max_questions=questions,
            resume_questions=resume,
            qa_turns=int(_number(env, "SENJIN_QA_TURNS", 3, 1, 10)),
            max_minutes=_number(env, "SENJIN_MAX_MINUTES", 45, 1, 600),
            idle_seconds=_number(env, "SENJIN_IDLE_SECONDS", 25, 0, 3600),
        )

    def with_max_questions(self, n: int) -> "InterviewConfig":
        if not 2 <= n <= 30:
            raise ConfigError(f"max questions must be between 2 and 30, got {n}")
        return replace(self, max_questions=n, resume_questions=None)

    def override(
        self,
        max_questions: int | None = None,
        max_minutes: float | None = None,
        idle_seconds: float | None = None,
    ) -> "InterviewConfig":
        """Command-line flags win over the environment."""
        cfg = self.with_max_questions(max_questions) if max_questions is not None else self
        if max_minutes is not None:
            cfg = replace(cfg, max_minutes=max_minutes)
        if idle_seconds is not None:
            cfg = replace(cfg, idle_seconds=idle_seconds)
        return cfg

    @property
    def resume(self) -> int:
        if self.resume_questions is not None:
            return min(self.resume_questions, self.max_questions - 1)
        return min(max(1, math.ceil(self.max_questions * 0.4)), self.max_questions - 1)

    @property
    def technical(self) -> int:
        return self.max_questions - self.resume

    def budgets(self) -> dict[Phase, Budget]:
        turns = {
            Phase.INTRO: 1,
            Phase.RESUME_DIVE: self.resume,
            Phase.TECHNICAL: self.technical,
            Phase.CANDIDATE_QA: self.qa_turns,
            Phase.SIGNOFF: 1,
        }
        return {p: Budget(turns[p], DEFAULT_BUDGETS[p].max_seconds) for p in Phase}

    def state(self) -> InterviewState:
        return InterviewState(budgets=self.budgets(), max_total_seconds=self.max_minutes * 60)

    def describe(self) -> str:
        return (
            f"Interview plan: {self.max_questions} questions ({self.resume} background + "
            f"{self.technical} technical), then up to {self.qa_turns} replies to the candidate, "
            f"within {self.max_minutes:g} minutes."
        )
