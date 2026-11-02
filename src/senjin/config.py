"""Interview limits, managed through environment variables.

Settings can also be kept in a .env file (see .env.example); exported variables win over it.

SENJIN_MAX_QUESTIONS     questions asked in the background + technical parts (default 5, 2 to 30)
SENJIN_RESUME_QUESTIONS  how many of those are about the resume (default: 40%, at least 1)
SENJIN_FOLLOW_UPS        follow-ups after each technical question (default 1, 0 to 3)
SENJIN_MAX_MINUTES       time box for the whole interview (default: scaled to the questions:
                         about 2 minutes per question and follow-up, plus 10; 30 for the
                         default 5 questions)
SENJIN_IDLE_SECONDS      silence before "Are you still there?" (default 25, 0 turns it off)
SENJIN_SILENCE_MS        silence that ends the candidate's turn (default 1000, 200 to 3000)
SENJIN_HOLD_MS           after a short or unfinished answer, wait this much longer for the candidate
                         to carry on before replying (default 1500, 0 turns it off)
SENJIN_DECISION_MODEL    a local decision model (for example nimble) that double-checks whether the
                         candidate has finished the Q&A when the rules say they have not (default:
                         off, rules only). It can only end the Q&A, never change a question.
SENJIN_VOICE_PACE        how slowly Senjin speaks: 1.0 is the voice's normal speed, 1.2 is 20%
                         slower (default 1.2, 0.8 to 1.6)
SENJIN_FILLER_MS         play "Okay, right." or "I see." if no reply has started this long
                         after the candidate stops (default 1100, 0 turns it off)

The greeting, the candidate's own Q&A and the goodbye are not counted as interview questions.
The candidate Q&A has no limit: it goes on until the candidate says they are done, the time box
runs out, or the candidate goes silent.
"""

import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from senjin.interview.state import DEFAULT_BUDGETS, Budget, InterviewState, Phase


class ConfigError(ValueError):
    pass


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_VOICE_PACE = 1.2  # slower than the voice's own pace: easier to follow over a call
MINUTES_PER_TURN = 2.0  # allowance per interviewer turn when the interview is long
EXTRA_MINUTES = 10  # greeting, the candidate's own questions and the goodbye


def _parse_env_line(line: str) -> tuple[str, str] | None:
    line = line.strip()
    if not line or line.startswith("#") or "=" not in line:
        return None
    if line.startswith("export "):
        line = line[len("export ") :]
    key, _, value = line.partition("=")
    key, value = key.strip(), value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        value = value[1:-1]
    elif " #" in value:  # inline comment on an unquoted value
        value = value.split(" #", 1)[0].rstrip()
    return key, value


def load_env_file(paths: Sequence[Path] | None = None) -> dict[str, str]:
    """Read SENJIN_* settings from a .env file into os.environ. Returns what was set.

    The first file found wins: `$SENJIN_ENV_FILE` (empty value disables .env), else `.env` in the
    current directory, else `.env` in the project root. Values already in the environment are
    never overwritten, so a variable exported in the shell beats the file. Only SENJIN_* keys
    are read.
    """
    if paths is None:
        chosen = os.environ.get("SENJIN_ENV_FILE")
        if chosen is not None:
            paths = [Path(chosen)] if chosen else []
        else:
            paths = [Path.cwd() / ".env", PROJECT_ROOT / ".env"]
    loaded: dict[str, str] = {}
    for path in paths:
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            parsed = _parse_env_line(line)
            if parsed and parsed[0].startswith("SENJIN_") and parsed[0] not in os.environ:
                os.environ[parsed[0]] = loaded[parsed[0]] = parsed[1]
        break
    return loaded


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
    max_minutes: float | None = None  # None: scaled with the number of questions
    idle_seconds: float = 25.0
    silence_ms: int = 1000
    filler_ms: int = 1100
    hold_ms: int = 1500
    voice_pace: float = DEFAULT_VOICE_PACE
    decision_model: str = ""  # "": the rules decide alone
    follow_ups: int = 1  # per technical question; each main question gets its own turn first
    minutes_per_turn: float = MINUTES_PER_TURN
    extra_minutes: float = EXTRA_MINUTES

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "InterviewConfig":
        if env is None:
            load_env_file()
            env = os.environ
        questions = int(_number(env, "SENJIN_MAX_QUESTIONS", 5, 2, 30))
        resume = None
        if env.get("SENJIN_RESUME_QUESTIONS", "").strip():
            resume = int(_number(env, "SENJIN_RESUME_QUESTIONS", 0, 1, questions - 1))
        return cls(
            max_questions=questions,
            resume_questions=resume,
            max_minutes=(
                _number(env, "SENJIN_MAX_MINUTES", 0, 1, 600)
                if env.get("SENJIN_MAX_MINUTES", "").strip()
                else None
            ),
            idle_seconds=_number(env, "SENJIN_IDLE_SECONDS", 25, 0, 3600),
            silence_ms=int(_number(env, "SENJIN_SILENCE_MS", 1000, 200, 3000)),
            filler_ms=int(_number(env, "SENJIN_FILLER_MS", 1100, 0, 10000)),
            hold_ms=int(_number(env, "SENJIN_HOLD_MS", 1500, 0, 5000)),
            voice_pace=_number(env, "SENJIN_VOICE_PACE", DEFAULT_VOICE_PACE, 0.8, 1.6),
            decision_model=env.get("SENJIN_DECISION_MODEL", "").strip(),
            follow_ups=int(_number(env, "SENJIN_FOLLOW_UPS", 1, 0, 3)),
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
        silence_ms: int | None = None,
        filler_ms: int | None = None,
        hold_ms: int | None = None,
        follow_ups: int | None = None,
    ) -> "InterviewConfig":
        """Command-line flags win over the environment."""
        cfg = self.with_max_questions(max_questions) if max_questions is not None else self
        if max_minutes is not None:
            cfg = replace(cfg, max_minutes=max_minutes)
        if idle_seconds is not None:
            cfg = replace(cfg, idle_seconds=idle_seconds)
        if silence_ms is not None:
            cfg = replace(cfg, silence_ms=silence_ms)
        if filler_ms is not None:
            cfg = replace(cfg, filler_ms=filler_ms)
        if hold_ms is not None:
            cfg = replace(cfg, hold_ms=hold_ms)
        if follow_ups is not None:
            if not 0 <= follow_ups <= 3:
                raise ConfigError(f"follow-ups must be between 0 and 3, got {follow_ups}")
            cfg = replace(cfg, follow_ups=follow_ups)
        return cfg

    @property
    def resume(self) -> int:
        if self.resume_questions is not None:
            return min(self.resume_questions, self.max_questions - 1)
        return min(max(1, math.ceil(self.max_questions * 0.4)), self.max_questions - 1)

    @property
    def technical(self) -> int:
        return self.max_questions - self.resume

    def _phase_minutes(self, phase: Phase, turns: int) -> float:
        """Time cap for a phase: the default, or about 2 minutes per interviewer turn if more."""
        return max((DEFAULT_BUDGETS[phase].max_seconds or 0) / 60, self.minutes_per_turn * turns)

    @property
    def minutes(self) -> float:
        """The whole-interview time box: SENJIN_MAX_MINUTES, else scaled to the question count."""
        if self.max_minutes is not None:
            return self.max_minutes
        scaled = (
            self._phase_minutes(Phase.RESUME_DIVE, self.resume)
            + self._phase_minutes(Phase.TECHNICAL, self.technical * (1 + self.follow_ups))
            + self.extra_minutes
        )
        return math.ceil(scaled)

    def budgets(self) -> dict[Phase, Budget]:
        turns = {
            Phase.INTRO: 1,
            Phase.RESUME_DIVE: self.resume,
            Phase.TECHNICAL: self.technical * (1 + self.follow_ups),  # question + its follow-ups
            Phase.SIGNOFF: 1,
        }
        budgets = {p: Budget(n, DEFAULT_BUDGETS[p].max_seconds) for p, n in turns.items()}
        for phase in (Phase.RESUME_DIVE, Phase.TECHNICAL):
            budgets[phase] = Budget(turns[phase], self._phase_minutes(phase, turns[phase]) * 60)
        budgets[Phase.CANDIDATE_QA] = DEFAULT_BUDGETS[Phase.CANDIDATE_QA]  # unlimited
        return {p: budgets[p] for p in Phase}

    def state(self) -> InterviewState:
        return InterviewState(
            budgets=self.budgets(),
            max_total_seconds=self.minutes * 60,
            follow_ups_per_question=self.follow_ups,
        )

    def describe(self) -> str:
        return (
            f"Interview plan: {self.max_questions} questions ({self.resume} background + "
            f"{self.technical} separate technical questions, each with up to {self.follow_ups} "
            f"follow-up), then as many answers to the candidate's own questions as they want, "
            f"within {self.minutes:g} minutes."
        )
