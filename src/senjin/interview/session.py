"""One interview: owns history, phase control, and the transcript.

Two modes. With `prefetch=False` the model drives the tools itself (text chat, simulations).
With `prefetch=True` the orchestrator drives: it fetches the resume, the next question and the
job description itself, feeds follow-ups one at a time, and tells the model exactly what to do
each turn in a short note on the candidate's last message. That keeps the model from answering
its own questions, repeating follow-ups or skipping the candidate's Q&A.
"""

import json
import re
import time
from collections.abc import AsyncIterator, Callable

from senjin.interview.exemplars import ExemplarProvider
from senjin.interview.prompts import (
    EXEMPLAR_HEADER,
    KICKOFF,
    NOTE_FORMAT,
    NOTES,
    SYSTEM_PROMPT,
    phase_message,
)
from senjin.interview.state import InterviewState, Phase
from senjin.interview.transcript import TranscriptLog
from senjin.llm.client import LLMClient, Message, ToolCall
from senjin.llm.tools import ToolRunner

MAX_TOOL_ROUNDS = 3

_NO_MORE_QUESTIONS = re.compile(
    r"^(no|nope|nah|nothing)\b|no (more |other |further )?questions|that'?s (all|it)|"
    r"that is (all|it)|i'?m (good|all set|done|fine)|we'?re (good|done)|"
    r"nothing (else|more|further)|all good|^(thanks|thank you)\b|"
    r"^(ok|okay),? (thanks|thank you|that'?s)",
    re.IGNORECASE,
)

# Signs the candidate is still asking something, so "No, but what about..." keeps Q&A going.
_STILL_ASKING = re.compile(
    r"\?|\b(but|actually|wait|also|one more|another|what|how|when|where|who|why|which|"
    r"can you|could you|tell me|do you|is there|are there)\b",
    re.IGNORECASE,
)


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
        self.prefetch = prefetch and tools is not None  # orchestrated mode
        self._context = ""  # resume / job description for the current phase
        self._question: dict | None = None
        self._followups: list[str] = []
        self._question_new = False
        self._undo_note: Callable[[], None] | None = None  # re-arms the last note if unspoken
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
        elif self._candidate_is_done(candidate_text):
            self.state.jump(Phase.SIGNOFF, now)
            await self._enter_phase()
        elif self.state.due_to_advance(now):
            if self.state.advance(now) is None:
                return
            await self._enter_phase()
        examples: list[str] = []
        if self.exemplars:
            examples = await self.exemplars.examples(candidate_text, self.state.phase)
        self._undo_note = None
        note = await self._turn_note() if self.prefetch else None
        async for chunk in self._reply(examples, note):
            yield chunk

    def _candidate_is_done(self, text: str) -> bool:
        """In Q&A, "no, I'm good" ends the interview: the candidate decides when Q&A is over."""
        in_qa = self.state.phase == Phase.CANDIDATE_QA and self.state.turns_in_phase >= 1
        if not in_qa or len(text.split()) > 12 or _STILL_ASKING.search(text):
            return False
        return bool(_NO_MORE_QUESTIONS.search(text.strip()))

    async def _enter_phase(self) -> None:
        """Orchestrated mode: load what the new phase needs, without a model round trip."""
        self._context = ""
        if not self.prefetch:
            return
        phase = self.state.phase
        if phase == Phase.RESUME_DIVE:
            self._context = "\n\nCandidate resume:\n" + await self._fetch("get_candidate_resume")
        elif phase == Phase.TECHNICAL:
            self._absorb_question(await self._fetch("get_next_technical_question"))
        elif phase == Phase.CANDIDATE_QA:
            self._context = "\n\nJob description:\n" + await self._fetch("get_job_description")

    async def _fetch(self, name: str) -> str:
        assert self.tools
        try:
            result = await self.tools.call(name, {})
        except Exception as exc:  # a failed lookup must not end the interview
            result = f"ERROR: {exc}"
        if self.log:
            self.log.append("tool", f"{name}() -> {result[:500]}", self.state.phase.value)
        return result

    def _absorb_question(self, result: str) -> None:
        try:
            data = json.loads(result)
        except ValueError:
            data = {}
        if isinstance(data, dict) and data.get("prompt"):
            self._question = data
            self._followups = list(data.get("follow_ups") or [])
            self._question_new = True
        else:  # bank exhausted or lookup failed
            self._question, self._followups, self._question_new = None, [], False

    def _rearm_question(self) -> None:
        self._question_new = True

    async def _turn_note(self) -> str | None:
        """What the model must do this turn, decided by code rather than by the model."""
        phase = self.state.phase
        if phase == Phase.TECHNICAL:
            if self._question_new and self._question:
                self._question_new = False
                self._undo_note = self._rearm_question
                return NOTES["ask"].format(q=self._question["prompt"])
            if self._followups:
                follow_up = self._followups.pop(0)
                self._undo_note = lambda: self._followups.insert(0, follow_up)
                return NOTES["follow_up"].format(fu=follow_up)
            self._absorb_question(await self._fetch("get_next_technical_question"))
            if self._question_new and self._question:
                self._question_new = False
                self._undo_note = self._rearm_question
                return NOTES["next_question"].format(q=self._question["prompt"])
            return NOTES["probe"]
        if phase == Phase.CANDIDATE_QA:
            return NOTES["qa_open"] if self.state.turns_in_phase == 0 else NOTES["qa_answer"]
        if phase == Phase.SIGNOFF:
            return NOTES["signoff"]
        return None

    def say(self, text: str, *, final: bool = False) -> None:
        """Record a fixed line the orchestrator spoke itself (watchdog prompts, apologies)."""
        self._record("assistant", text)
        if final:
            self.state.finished = True

    async def _reply(
        self, examples: list[str] | None = None, note: str | None = None
    ) -> AsyncIterator[str]:
        # One system message: some chat templates (qwen) mishandle several or trailing ones.
        phase = phase_message(self.state.phase, orchestrated=self.prefetch)
        content = f"{SYSTEM_PROMPT}\nCurrent instruction: {phase}{self._context}"
        if examples:
            content += EXEMPLAR_HEADER + "\n\n".join(examples)
        system = {"role": "system", "content": content}
        tools = self.tools.tools if self.tools and not self.prefetch else None
        parts: list[str] = []
        completed = False
        try:
            for round_ in range(MAX_TOOL_ROUNDS + 1):
                messages: list[Message] = [system, *self.history]
                if not self.history:
                    messages.append({"role": "user", "content": KICKOFF})
                elif note and messages[-1]["role"] == "user":
                    last = messages[-1]
                    messages[-1] = {**last, "content": last["content"] + NOTE_FORMAT.format(note)}
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
            completed = True
        finally:
            # Also runs when the reply is cancelled (barge-in): keep what was generated.
            text = "".join(parts).strip()
            if text:
                self._record("assistant", text)
                self.state.record_interviewer_turn()
            # A goodbye that was cut off must not end the interview as if it had been said.
            if self.state.phase == Phase.SIGNOFF and completed:
                self.state.finished = True

    def interrupt(self, spoken: str) -> None:
        """The candidate cut the interviewer off: history keeps only what was actually said."""
        last = self.history[-1] if self.history else None
        if last and last["role"] == "assistant" and not last.get("tool_calls"):
            if spoken.strip():
                last["content"] = spoken.strip()
            else:  # nothing was said: the turn, and the question it carried, are not used up
                self.history.pop()
                self.state.turns_in_phase = max(0, self.state.turns_in_phase - 1)
                if self._undo_note:
                    self._undo_note()
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
