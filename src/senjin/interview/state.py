"""Interview phases. The state machine, not the LLM, decides when to advance."""

from dataclasses import dataclass, field
from enum import StrEnum


class Phase(StrEnum):
    INTRO = "intro"
    RESUME_DIVE = "resume_dive"
    TECHNICAL = "technical"
    CANDIDATE_QA = "candidate_qa"
    SIGNOFF = "signoff"


@dataclass(frozen=True)
class Budget:
    max_turns: int | None  # interviewer utterances allowed in this phase (None = no limit)
    max_seconds: float | None  # time allowed in this phase (None = no limit)


DEFAULT_BUDGETS: dict[Phase, Budget] = {
    Phase.INTRO: Budget(max_turns=1, max_seconds=90),
    Phase.RESUME_DIVE: Budget(max_turns=4, max_seconds=5 * 60),
    Phase.TECHNICAL: Budget(max_turns=8, max_seconds=15 * 60),
    Phase.CANDIDATE_QA: Budget(max_turns=None, max_seconds=None),  # the candidate decides
    Phase.SIGNOFF: Budget(max_turns=1, max_seconds=60),
}

ORDER = list(Phase)


def _env_budgets() -> dict[Phase, Budget]:
    """Default budgets honour SENJIN_MAX_QUESTIONS and friends (imported late: config uses us)."""
    from senjin.config import InterviewConfig

    return InterviewConfig.from_env().budgets()


def _env_follow_ups() -> int | None:
    from senjin.config import InterviewConfig

    return InterviewConfig.from_env().follow_ups


@dataclass
class InterviewState:
    budgets: dict[Phase, Budget] = field(default_factory=lambda: _env_budgets())
    phase: Phase = Phase.INTRO
    turns_in_phase: int = 0
    phase_started_at: float = 0.0
    finished: bool = False
    max_total_seconds: float | None = None  # hard time box for the whole interview
    started_at: float = 0.0
    # follow-ups asked after each technical question (None: every follow-up the bank has)
    follow_ups_per_question: int | None = field(default_factory=lambda: _env_follow_ups())

    def start(self, now: float) -> None:
        self.started_at = self.phase_started_at = now

    def record_interviewer_turn(self) -> None:
        self.turns_in_phase += 1

    def due_to_advance(self, now: float) -> bool:
        budget = self.budgets[self.phase]
        turns_up = budget.max_turns is not None and self.turns_in_phase >= budget.max_turns
        time_up = (
            budget.max_seconds is not None and now - self.phase_started_at >= budget.max_seconds
        )
        return turns_up or time_up

    def advance(self, now: float) -> Phase | None:
        """Move to the next phase; return it, or None (and finish) after the last one."""
        index = ORDER.index(self.phase)
        if index == len(ORDER) - 1:
            self.finished = True
            return None
        self.phase = ORDER[index + 1]
        self.turns_in_phase = 0
        self.phase_started_at = now
        return self.phase

    def wrap_up_target(self, now: float) -> Phase | None:
        """Where to jump when the whole-interview time box runs low, or None.

        At 85% of the time box, skip ahead to the candidate's questions; at 100%, sign off.
        """
        if self.max_total_seconds is None:
            return None
        used = (now - self.started_at) / self.max_total_seconds
        if used >= 1.0 and self.phase != Phase.SIGNOFF:
            return Phase.SIGNOFF
        if used >= 0.85 and ORDER.index(self.phase) < ORDER.index(Phase.CANDIDATE_QA):
            return Phase.CANDIDATE_QA
        return None

    def jump(self, phase: Phase, now: float) -> None:
        self.phase = phase
        self.turns_in_phase = 0
        self.phase_started_at = now
