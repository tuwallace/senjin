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
from senjin.interview.lookup import relevant_lines
from senjin.interview.praise import PraiseFilter
from senjin.interview.prompts import (
    EXEMPLAR_HEADER,
    GREETING,
    HINT_FORMAT,
    KICKOFF,
    NOTE_FORMAT,
    NOTES,
    SYSTEM_PROMPT,
    phase_message,
)
from senjin.interview.state import InterviewState, Phase
from senjin.interview.transcript import TranscriptLog
from senjin.llm.client import LLMClient, Message, ToolCall
from senjin.llm.decide import Decider
from senjin.llm.tools import ToolRunner

MAX_TOOL_ROUNDS = 3

# Asking the decision model only when the rules say "not done", and acting only on a confident
# yes: ending the interview early is worse than one more question.
DONE_QUESTION = (
    "The candidate has said they have no more questions and the interview can end. Answer low "
    "if they asked a question, want to ask something, or are still thinking."
)
DONE_THRESHOLD = 0.8

_NO_MORE_QUESTIONS = re.compile(
    r"^(no|nope|nah|nothing)\b|no (more |other |further )?questions|that'?s (all|it)|"
    r"that is (all|it)|i'?m (good|all set|done|fine)|we'?re (good|done)|"
    r"nothing (else|more|further)|all good|^(thanks|thank you)\b|"
    r"^(ok|okay),? (thanks|thank you|that'?s)|"
    r"(don'?t|do not|have no|haven'?t got|got no)\s+(have\s+)?(any\s+)?"
    r"(other\s+|more\s+|further\s+|additional\s+)?questions?|"
    r"\bi (don'?t|do not) have any\b|\bno questions\b|\bnone\b",
    re.IGNORECASE,
)

# Signs the candidate is still asking something, so "No, but what about..." keeps Q&A going.
_STILL_ASKING = re.compile(
    r"\?|\b(but|actually|wait|also|one more|another|what|how|when|where|who|why|which|"
    r"can you|could you|tell me|do you|is there|are there)\b",
    re.IGNORECASE,
)

# A longer, polite sign-off that says so outright still ends the Q&A.
_EXPLICITLY_DONE = re.compile(
    r"no (more|other|further|additional) questions|"
    r"(don'?t|do not) have (any )?(more |other |further )?questions|"
    r"that'?s (all|everything)|nothing (else|more|further)|i'?m (all set|done)",
    re.IGNORECASE,
)
_CARRIES_ON = re.compile(r"\b(but|actually|wait|also|one more|another)\b", re.IGNORECASE)

# "Can you repeat the question?" is a request, not an answer: it must not use up a question.
_REPEAT_REQUEST = re.compile(
    r"\b(repeat|say (that|it|the question) again|come again|pardon|rephrase|"
    r"(didn'?t|did not|couldn'?t|could not) (catch|hear|get|understand)|"
    r"(what|which) (was|is) (the|that) question|sorry,? what|i don'?t understand "
    r"(the|that|your) question)",
    re.IGNORECASE,
)

# "I forgot the last part of the question": the candidate answered some of it. Not "I forgot to
# mention ...", which is more answer.
_FORGOT_PART = re.compile(
    r"\bi'?(ve)? ?(have )?(forgot|forgotten|forget|lost track of|"
    r"(don'?t|can'?t|do not|cannot) remember)\b[^.?!]{0,40}"
    r"\b(question|part|what you (asked|said))\b|"
    r"what was the (last|second|other|rest|first) part|remind me (of )?(the|what)",
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
        decider: Decider | None = None,
    ) -> None:
        self.decider = decider
        self.prefetch = prefetch and tools is not None  # orchestrated mode
        self._context = ""  # resume and job description: fixed for the whole interview
        self._job_description = ""
        self._last_candidate_text = ""
        self._last_asked = ""  # the question or follow-up most recently put to the candidate
        self._free_turn = False  # the next interviewer turn does not use up the phase budget
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
        await self._load_context()
        if self.prefetch:  # orchestrated (voice, Meet): the notice must never be paraphrased away
            self._record("assistant", GREETING)
            self.state.record_interviewer_turn()
            yield GREETING
            return
        async for chunk in self._reply():
            yield chunk

    async def respond(self, candidate_text: str) -> AsyncIterator[str]:
        """Record the candidate's turn, advance the phase if due, stream the reply."""
        self._record("user", candidate_text)
        self._last_candidate_text = candidate_text
        await self._load_context()
        now = self.clock()
        forgot = self.forgot_part(candidate_text)
        repeat = self.asks_to_repeat(candidate_text) or forgot
        if target := self.state.wrap_up_target(now):
            self.state.jump(target, now)
            await self._enter_phase()
            repeat = False
        elif await self._is_done(candidate_text):
            self.state.jump(Phase.SIGNOFF, now)
            await self._enter_phase()
            repeat = False
        elif repeat:
            pass  # stay on the same question: it is asked again, not skipped
        elif self.state.due_to_advance(now):
            if self.state.advance(now) is None:
                return
            await self._enter_phase()
        examples: list[str] = []
        if self.exemplars:
            examples = await self.exemplars.examples(candidate_text, self.state.phase)
        self._undo_note = None
        if repeat:
            self._free_turn = True
            note = NOTES["forgot" if forgot else "repeat"].format(
                previous=self._previous_interviewer_text()
            )
        else:
            note = await self._turn_note() if self.prefetch else None
        async for chunk in self._reply(examples, note):
            yield chunk

    def forgot_part(self, text: str) -> bool:
        in_questions = self.state.phase in (Phase.RESUME_DIVE, Phase.TECHNICAL)
        return bool(self.prefetch and in_questions and _FORGOT_PART.search(text))

    def asks_to_repeat(self, text: str) -> bool:
        in_questions = self.state.phase in (Phase.RESUME_DIVE, Phase.TECHNICAL)
        short = len(text.split()) <= 14
        return bool(self.prefetch and in_questions and short and _REPEAT_REQUEST.search(text))

    def _previous_interviewer_text(self) -> str:
        if self.state.phase == Phase.TECHNICAL and self._last_asked:
            return self._last_asked
        for message in reversed(self.history[:-1]):
            if message["role"] == "assistant" and message.get("content"):
                return message["content"]
        return ""

    def retract_candidate(self) -> None:
        """The last candidate turn was an unfinished fragment; it is re-sent with the rest."""
        if self.history and self.history[-1]["role"] == "user":
            self.history.pop()
            if self.log:
                self.log.append("retract", "", self.state.phase.value)

    async def _is_done(self, text: str) -> bool:
        """The rules decide; a decision model may only add a sign-off the rules missed."""
        if self._candidate_is_done(text):
            return True
        in_qa = self.state.phase == Phase.CANDIDATE_QA and self.state.turns_in_phase >= 1
        if not (self.decider and in_qa) or "?" in text or len(text.split()) > 60:
            return False
        p = await self.decider.probability(
            "done",
            {"interviewer_said": self._previous_interviewer_text()[-300:], "candidate_said": text},
            DONE_QUESTION,
        )
        return p is not None and p >= DONE_THRESHOLD

    def _candidate_is_done(self, text: str) -> bool:
        """In Q&A, "no, I'm good" ends the interview: the candidate decides when Q&A is over."""
        in_qa = self.state.phase == Phase.CANDIDATE_QA and self.state.turns_in_phase >= 1
        if not in_qa or "?" in text:
            return False
        words = len(text.split())
        if words <= 20 and not _STILL_ASKING.search(text) and _NO_MORE_QUESTIONS.search(text):
            return True
        # "Thank you, that was helpful. I have no more questions. Have a great day!"
        return words <= 60 and bool(_EXPLICITLY_DONE.search(text)) and not _CARRIES_ON.search(text)

    @property
    def context(self) -> str:
        """The resume and job description in the system prompt (empty until loaded)."""
        return self._context

    async def warm(self) -> None:
        """Before the call: load the model and have it read the long, fixed system prompt."""
        await self._load_context()
        warm = getattr(self.llm, "warm", None)
        if self.prefetch and warm:
            system = {"role": "system", "content": f"{SYSTEM_PROMPT}{self._context}"}
            await warm([system, {"role": "user", "content": KICKOFF}])
        if self.decider:  # a cold decision model can miss its 3 s limit on the first real call
            await self.decider.probability(
                "ready", {"candidate_said": "Yes."}, "Is the candidate ready?"
            )

    async def _load_context(self) -> None:
        """Orchestrated mode: put the resume and job description in the system prompt up front.

        The system prompt then never changes, so Ollama reuses its cache across phase changes
        instead of re-reading the whole prompt (that cost 5 to 9 seconds on phase-change turns).
        """
        if not self.prefetch or self._context:
            return
        resume = await self._fetch("get_candidate_resume")
        self._job_description = await self._fetch("get_job_description")
        self._context = (
            f"\n\nCandidate resume:\n{resume}\n\nJob description:\n{self._job_description}"
        )

    async def _enter_phase(self) -> None:
        """Orchestrated mode: load what the new phase needs, without a model round trip."""
        if self.prefetch and self.state.phase == Phase.TECHNICAL:
            self._absorb_question(await self._fetch("get_next_technical_question"))

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
            limit = self.state.follow_ups_per_question
            if limit is not None:
                self._followups = self._followups[:limit]
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
                self._last_asked = self._question["prompt"]
                return NOTES["ask"].format(q=self._question["prompt"])
            if self._followups:
                follow_up = self._followups.pop(0)
                self._undo_note = lambda: self._followups.insert(0, follow_up)
                self._last_asked = follow_up
                return NOTES["follow_up"].format(fu=follow_up)
            self._absorb_question(await self._fetch("get_next_technical_question"))
            if self._question_new and self._question:
                self._question_new = False
                self._undo_note = self._rearm_question
                self._last_asked = self._question["prompt"]
                return NOTES["next_question"].format(q=self._question["prompt"])
            return NOTES["probe"]
        if phase == Phase.RESUME_DIVE and self.state.turns_in_phase >= 1:
            angles = NOTES["resume_angles"]
            return NOTES["resume_probe"].format(angle=angles[(self.state.turns_in_phase - 1) % 4])
        if phase == Phase.CANDIDATE_QA:
            if self.state.turns_in_phase == 0:
                return NOTES["qa_open"]
            lines = relevant_lines(self._job_description, self._last_candidate_text)
            hint = HINT_FORMAT.format(" | ".join(lines)) if lines else ""
            return NOTES["qa_answer"] + hint
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
        if self.prefetch:  # the phase rides in the note so the system prompt stays identical
            content = f"{SYSTEM_PROMPT}{self._context}"
            note = f"{phase} {note}" if note else phase
        else:
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
                    kickoff = KICKOFF + (NOTE_FORMAT.format(note) if note else "")
                    messages.append({"role": "user", "content": kickoff})
                elif note and messages[-1]["role"] == "user":
                    last = messages[-1]
                    messages[-1] = {**last, "content": last["content"] + NOTE_FORMAT.format(note)}
                calls: list[ToolCall] = []
                praise = PraiseFilter()  # holds each sentence back until praise is stripped
                async for event in self.llm.stream(messages, tools):
                    if isinstance(event, ToolCall):
                        calls.append(event)
                    elif text := praise.feed(event):
                        parts.append(text)
                        yield text
                if tail := praise.flush():
                    parts.append(tail)
                    yield tail
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
                if self._free_turn:  # a repeated question does not count as another one
                    self._free_turn = False
                else:
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
        elif last and last["role"] == "user" and not spoken.strip() and self._undo_note:
            # Cut in before the model had produced a word: the follow-up or question chosen for
            # this turn was never put to the candidate, so it must not be used up.
            self._undo_note()
            self._undo_note = None
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
