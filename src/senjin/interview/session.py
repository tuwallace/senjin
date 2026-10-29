"""One interview: owns history, phase control, and the transcript."""

import time
from collections.abc import AsyncIterator, Callable

from senjin.interview.prompts import KICKOFF, SYSTEM_PROMPT, phase_message
from senjin.interview.state import InterviewState
from senjin.interview.transcript import TranscriptLog
from senjin.llm.client import LLMClient, Message


class InterviewSession:
    def __init__(
        self,
        llm: LLMClient,
        log: TranscriptLog | None = None,
        state: InterviewState | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.llm = llm
        self.log = log
        self.state = state or InterviewState()
        self.clock = clock
        self.history: list[Message] = []
        self._entering = True
        self.state.start(self.clock())

    @property
    def finished(self) -> bool:
        return self.state.finished

    async def open(self) -> AsyncIterator[str]:
        """The interviewer speaks first."""
        async for chunk in self._reply():
            yield chunk

    async def respond(self, candidate_text: str) -> AsyncIterator[str]:
        """Record the candidate's turn, advance the phase if due, stream the reply."""
        self._record("user", candidate_text)
        if self.state.due_to_advance(self.clock()):
            if self.state.advance(self.clock()) is None:
                return
            self._entering = True
        async for chunk in self._reply():
            yield chunk

    async def _reply(self) -> AsyncIterator[str]:
        # One system message: some chat templates (qwen) mishandle several or trailing ones.
        phase = phase_message(self.state.phase, entering=self._entering)
        messages: list[Message] = [
            {"role": "system", "content": f"{SYSTEM_PROMPT}\nCurrent instruction: {phase}"},
            *self.history,
        ]
        if not self.history:
            messages.append({"role": "user", "content": KICKOFF})
        self._entering = False
        parts: list[str] = []
        async for chunk in self.llm.stream(messages):
            parts.append(chunk)
            yield chunk
        self._record("assistant", "".join(parts).strip())
        self.state.record_interviewer_turn()
        if self.state.phase.value == "signoff":
            self.state.finished = True

    def _record(self, role: str, text: str) -> None:
        self.history.append({"role": role, "content": text})
        if self.log:
            self.log.append(role, text, self.state.phase.value)
