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
    max_turns: int  # interviewer utterances allowed in this phase
    max_seconds: float


DEFAULT_BUDGETS: dict[Phase, Budget] = {
    Phase.INTRO: Budget(max_turns=1, max_seconds=90),
    Phase.RESUME_DIVE: Budget(max_turns=4, max_seconds=5 * 60),
    Phase.TECHNICAL: Budget(max_turns=8, max_seconds=15 * 60),
    Phase.CANDIDATE_QA: Budget(max_turns=3, max_seconds=4 * 60),
    Phase.SIGNOFF: Budget(max_turns=1, max_seconds=60),
}

ORDER = list(Phase)


@dataclass
class InterviewState:
    budgets: dict[Phase, Budget] = field(default_factory=lambda: dict(DEFAULT_BUDGETS))
    phase: Phase = Phase.INTRO
    turns_in_phase: int = 0
    phase_started_at: float = 0.0
    finished: bool = False

    def start(self, now: float) -> None:
        self.phase_started_at = now

    def record_interviewer_turn(self) -> None:
        self.turns_in_phase += 1

    def due_to_advance(self, now: float) -> bool:
        budget = self.budgets[self.phase]
        return (
            self.turns_in_phase >= budget.max_turns
            or now - self.phase_started_at >= budget.max_seconds
        )

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
