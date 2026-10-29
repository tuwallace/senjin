"""One interview: owns history, phase control, and the transcript."""

import time
from collections.abc import AsyncIterator, Callable

from senjin.interview.exemplars import ExemplarProvider
from senjin.interview.prompts import EXEMPLAR_HEADER, KICKOFF, SYSTEM_PROMPT, phase_message
from senjin.interview.state import InterviewState, Phase
from senjin.interview.transcript import TranscriptLog
from senjin.llm.client import LLMClient, Message, ToolCall
from senjin.llm.tools import ToolRunner

MAX_TOOL_ROUNDS = 3

# Tools the orchestrator runs itself on entering a phase, saving an LLM round trip.
PREFETCH: dict[Phase, list[ToolCall]] = {
    Phase.RESUME_DIVE: [ToolCall("get_candidate_resume")],
    Phase.TECHNICAL: [ToolCall("get_next_technical_question")],
}


class InterviewSession:
    def __init__(
        self,
        llm: LLMClient,
        log: TranscriptLog | None = None,
        state: InterviewState | None = None,
        clock: Callable[[], float] = time.monotonic,
        tools: ToolRunner | None = None,
        exemplars: ExemplarProvider | None = None,
        prefetch: bool = False,
    ) -> None:
        self.prefetch = prefetch and tools is not None
        self._prefetched = False
        self.exemplars = exemplars
        self.llm = llm
        self.tools = tools
        self.log = log
        self.state = state or InterviewState()
        self.clock = clock
        self.history: list[Message] = []
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
        now = self.clock()
        if target := self.state.wrap_up_target(now):
            self.state.jump(target, now)
            await self._enter_phase()
        elif self.state.due_to_advance(now):
            if self.state.advance(now) is None:
                return
            await self._enter_phase()
        examples: list[str] = []
        if self.exemplars:
            examples = await self.exemplars.examples(candidate_text, self.state.phase)
        async for chunk in self._reply(examples):
            yield chunk

    async def _enter_phase(self) -> None:
        self._prefetched = False
        if self.prefetch and self.state.phase in PREFETCH:
            await self._run_tools(list(PREFETCH[self.state.phase]))
            self._prefetched = True

    def say(self, text: str, *, final: bool = False) -> None:
        """Record a fixed line the orchestrator spoke itself (watchdog prompts, apologies)."""
        self._record("assistant", text)
        if final:
            self.state.finished = True

    async def _reply(self, examples: list[str] | None = None) -> AsyncIterator[str]:
        # One system message: some chat templates (qwen) mishandle several or trailing ones.
        phase = phase_message(self.state.phase, prefetched=self._prefetched)
        content = f"{SYSTEM_PROMPT}\nCurrent instruction: {phase}"
        if examples:
            content += EXEMPLAR_HEADER + "\n\n".join(examples)
        system = {"role": "system", "content": content}
        tools = self.tools.tools if self.tools else None
        parts: list[str] = []
        try:
            for round_ in range(MAX_TOOL_ROUNDS + 1):
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
                # last round: no tools, so the model must speak
                tools = self.tools.tools if round_ < MAX_TOOL_ROUNDS - 1 else None
        finally:
            # Also runs when the reply is cancelled (barge-in): keep what was generated.
            text = "".join(parts).strip()
            if text:
                self._record("assistant", text)
            self.state.record_interviewer_turn()
            if self.state.phase.value == "signoff":
                self.state.finished = True

    def interrupt(self, spoken: str) -> None:
        """The candidate cut the interviewer off: history keeps only what was actually said."""
        last = self.history[-1] if self.history else None
        if last and last["role"] == "assistant" and not last.get("tool_calls"):
            if spoken.strip():
                last["content"] = spoken.strip()
            else:
                self.history.pop()
        if self.log:
            self.log.append("interrupt", spoken.strip(), self.state.phase.value)

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
