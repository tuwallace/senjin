"""One interview: owns history, phase control, and the transcript."""

import time
from collections.abc import AsyncIterator, Callable

from senjin.interview.prompts import KICKOFF, SYSTEM_PROMPT, phase_message
from senjin.interview.state import InterviewState
from senjin.interview.transcript import TranscriptLog
from senjin.llm.client import LLMClient, Message, ToolCall
from senjin.llm.tools import ToolRunner

MAX_TOOL_ROUNDS = 3


class InterviewSession:
    def __init__(
        self,
        llm: LLMClient,
        log: TranscriptLog | None = None,
        state: InterviewState | None = None,
        clock: Callable[[], float] = time.monotonic,
        tools: ToolRunner | None = None,
    ) -> None:
        self.llm = llm
        self.tools = tools
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
        self._entering = False
        system = {"role": "system", "content": f"{SYSTEM_PROMPT}\nCurrent instruction: {phase}"}
        tools = self.tools.tools if self.tools else None
        parts: list[str] = []
        for _ in range(MAX_TOOL_ROUNDS + 1):
            messages: list[Message] = [system, *self.history]
            if not self.history:
                messages.append({"role": "user", "content": KICKOFF})
            calls: list[ToolCall] = []
            async for event in self.llm.stream(messages, tools):
                if isinstance(event, ToolCall):
                    calls.append(event)
                else:
                    parts.append(event)
                    yield event
            if not calls or not self.tools:
                break
            await self._run_tools(calls)
            tools = self.tools.tools if _ < MAX_TOOL_ROUNDS - 1 else None  # force a spoken reply
        self._record("assistant", "".join(parts).strip())
        self.state.record_interviewer_turn()
        if self.state.phase.value == "signoff":
            self.state.finished = True

    async def _run_tools(self, calls: list[ToolCall]) -> None:
        assert self.tools
        self.history.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"function": {"name": c.name, "arguments": c.arguments}} for c in calls
                ],
            }
        )
        for call in calls:
            try:
                result = await self.tools.call(call.name, call.arguments)
            except Exception as exc:  # a failed tool must not end the interview
                result = f"ERROR: {exc}"
            self.history.append({"role": "tool", "tool_name": call.name, "content": result})
            if self.log:
                self.log.append("tool", f"{call.name}({call.arguments}) -> {result[:500]}", "")

    def _record(self, role: str, text: str) -> None:
        self.history.append({"role": role, "content": text})
        if self.log:
            self.log.append(role, text, self.state.phase.value)
