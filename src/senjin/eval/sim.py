"""Simulated candidates: a second local model plays the candidate so interviews run unattended."""

import argparse
import asyncio
import os
import time
from contextlib import AsyncExitStack
from pathlib import Path

from senjin.interview.exemplars import ExemplarIndex
from senjin.interview.session import InterviewSession
from senjin.interview.state import DEFAULT_BUDGETS, Budget, InterviewState, Phase
from senjin.interview.transcript import TranscriptLog
from senjin.llm.client import DEFAULT_MODEL, LLMClient, Message, OllamaClient
from senjin.llm.tools import open_mcp_tools

ROOT = Path(__file__).resolve().parents[3]

PERSONAS = {
    "strong": (
        "You are an excellent senior backend engineer. Answer concretely: name specific "
        "techniques, trade-offs, failure modes and numbers from experience. Reason aloud."
    ),
    "average": (
        "You are a competent but unremarkable engineer. Give reasonable but general answers, "
        "miss edge cases and failure modes unless prompted, and stay a little vague."
    ),
    "weak": (
        "You are a junior engineer out of your depth. Give short, shallow or partly wrong "
        "answers, and often say you are not sure or have not done that before."
    ),
}


class SimulatedCandidate:
    def __init__(self, llm: LLMClient, persona: str, resume: str) -> None:
        system = (
            f"You are a job candidate in a spoken video interview. {PERSONAS[persona]}\n"
            "Speak naturally in 2 to 5 sentences, first person, no markdown, no lists. "
            "Never mention that you are an AI or a simulation.\n\n"
            f"Your resume:\n{resume}"
        )
        self.llm = llm
        self.messages: list[Message] = [{"role": "system", "content": system}]

    async def reply(self, interviewer_text: str) -> str:
        self.messages.append({"role": "user", "content": interviewer_text})
        text = "".join([e async for e in self.llm.stream(self.messages) if isinstance(e, str)])
        text = text.strip() or "Could you repeat that?"
        self.messages.append({"role": "assistant", "content": text})
        return text


def quick_state() -> InterviewState:
    short = {
        Phase.INTRO: 1,
        Phase.RESUME_DIVE: 2,
        Phase.TECHNICAL: 3,
        Phase.CANDIDATE_QA: 1,
        Phase.SIGNOFF: 1,
    }
    budgets = {p: Budget(short[p], DEFAULT_BUDGETS[p].max_seconds) for p in Phase}
    return InterviewState(budgets=budgets)


async def run_simulation(
    persona: str,
    model: str,
    out: Path,
    exemplars: ExemplarIndex | None = None,
    quick: bool = False,
    orchestrate: bool = False,
    max_exchanges: int = 40,
) -> Path:
    candidate_dir = ROOT / "examples/candidate"
    resume = (candidate_dir / "resume.md").read_text(encoding="utf-8")
    candidate = SimulatedCandidate(OllamaClient(model=model, temperature=0.8), persona, resume)
    async with AsyncExitStack() as stack:
        env = {**os.environ, "SENJIN_NOTES_PATH": str(out.with_suffix(".notes.jsonl"))}
        tools = await stack.enter_async_context(open_mcp_tools(env=env))
        session = InterviewSession(
            OllamaClient(model=model),
            TranscriptLog(out),
            state=quick_state() if quick else None,
            tools=tools,
            exemplars=exemplars,
            prefetch=orchestrate,
        )
        said = "".join([c async for c in session.open()])
        for _ in range(max_exchanges):
            if session.finished:
                break
            answer = await candidate.reply(said)
            said = "".join([c async for c in session.respond(answer)])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an unattended simulated interview")
    parser.add_argument("--persona", choices=sorted(PERSONAS), default="average")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--exemplars", type=Path, help="exemplar JSONL (omit to run without)")
    parser.add_argument("--quick", action="store_true", help="short phase budgets (about 9 turns)")
    parser.add_argument(
        "--orchestrate", action="store_true", help="code-driven questions, as in voice and Meet"
    )
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    out = args.out or Path("data/sessions") / f"sim-{args.persona}-{time.strftime('%H%M%S')}.jsonl"
    index = ExemplarIndex.load(args.exemplars) if args.exemplars else None
    asyncio.run(run_simulation(args.persona, args.model, out, index, args.quick, args.orchestrate))
    print(out)


if __name__ == "__main__":
    main()
