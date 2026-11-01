"""Simulated candidates: a second local model plays the candidate so interviews run unattended."""

import argparse
import asyncio
import time
from contextlib import AsyncExitStack
from pathlib import Path

from senjin.candidate import JOB_DESCRIPTION, QUESTIONS, RESUME, candidate_from_args
from senjin.config import InterviewConfig
from senjin.interview.exemplars import ExemplarIndex
from senjin.interview.praise import plain_speech
from senjin.interview.session import InterviewSession
from senjin.interview.state import DEFAULT_BUDGETS, Budget, InterviewState, Phase
from senjin.interview.transcript import TranscriptLog
from senjin.llm.client import DEFAULT_MODEL, LLMClient, Message, OllamaClient
from senjin.llm.decide import decider_for
from senjin.llm.tools import open_mcp_tools
from senjin.mcp_server.store import bank_label
from senjin.plan import PLAN, plan_from_args

ROOT = Path(__file__).resolve().parents[3]

PERSONAS = {
    "resume": (
        "You are the person described by your resume below. Answer as that person would: "
        "specific about the work, tools and results the resume supports, and honest about what "
        "it does not cover (say you have not done that, or reason from related experience). "
        "Never invent employers, degrees or numbers that are not in the resume."
    ),
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
            "Never mention that you are an AI or a simulation. When you are asked whether you have "
            "questions, ask at most one short question about the role, and after the answer say "
            "that you have no more questions and thank them.\n\n"
            f"Your resume:\n{resume}"
        )
        self.llm = llm
        self.messages: list[Message] = [{"role": "system", "content": system}]

    async def reply(self, interviewer_text: str) -> str:
        self.messages.append({"role": "user", "content": interviewer_text})
        text = "".join([e async for e in self.llm.stream(self.messages) if isinstance(e, str)])
        text = plain_speech(text).strip() or "Could you repeat that?"  # speech has no markdown
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
    max_exchanges: int = 60,
    candidate_dir: Path | None = None,
    log=print,
) -> Path:
    """A simulated interview. With `candidate_dir` it is that candidate's real interview setup:
    their resume plays the candidate, their questions.yaml and interview.yaml drive the
    interviewer exactly as in a Meet call. Without it, the built-in sample and the whole bank."""
    own = candidate_dir is not None
    folder = candidate_dir or ROOT / "examples/candidate"
    resume = (folder / "resume.md").read_text(encoding="utf-8")
    candidate = SimulatedCandidate(OllamaClient(model=model, temperature=0.8), persona, resume)
    state = quick_state() if quick else None
    if own:
        plan = plan_from_args(folder)
        state = plan.apply(InterviewConfig.from_env()).state()
        log(plan.describe(folder / PLAN))
        log(bank_label(folder))
        orchestrate = True  # the same code-driven flow as voice and Meet
    async with AsyncExitStack() as stack:
        tools = await stack.enter_async_context(
            open_mcp_tools(
                folder,
                bank_dir=None if own else ROOT / "question_bank",
                notes_path=out.with_suffix(".notes.jsonl"),
            )
        )
        session = InterviewSession(
            OllamaClient(model=model),
            TranscriptLog(out),
            state=state,
            tools=tools,
            exemplars=exemplars,
            prefetch=orchestrate,
            decider=decider_for(InterviewConfig.from_env().decision_model),
        )
        if own:
            await session.warm()
        said = "".join([c async for c in session.open()])
        log(f"Senjin: {said.strip()}")
        repeats = 0
        for _ in range(max_exchanges):
            if session.finished:
                break
            answer = await candidate.reply(said)
            log(f"Candidate: {answer}")
            reply = "".join([c async for c in session.respond(answer)])
            log(f"Senjin: {reply.strip()}")
            repeats = repeats + 1 if reply.strip() == said.strip() else 0
            if repeats >= 2:  # the interview is not progressing: stop instead of looping to the cap
                raise RuntimeError("the interview is stuck repeating the same reply")
            said = reply
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an unattended simulated interview")
    parser.add_argument(
        "--persona",
        choices=sorted(PERSONAS),
        help="how the candidate answers (default: resume with --candidate-dir, else average)",
    )
    parser.add_argument(
        "--candidate-dir",
        type=Path,
        help=f"simulate this candidate's interview (default: the built-in sample); needs {RESUME}, "
        f"{JOB_DESCRIPTION}, {QUESTIONS} and {PLAN} from senjin-review",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--exemplars", type=Path, help="exemplar JSONL (omit to run without)")
    parser.add_argument("--quick", action="store_true", help="short phase budgets (about 9 turns)")
    parser.add_argument(
        "--orchestrate", action="store_true", help="code-driven questions, as in voice and Meet"
    )
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    folder = None
    if args.candidate_dir is not None:
        folder = candidate_from_args(
            args.candidate_dir, files=(RESUME, JOB_DESCRIPTION), questions=True
        )
    persona = args.persona or ("resume" if folder else "average")
    out = args.out or Path("data/sessions") / f"sim-{persona}-{time.strftime('%H%M%S')}.jsonl"
    index = ExemplarIndex.load(args.exemplars) if args.exemplars else None
    asyncio.run(
        run_simulation(
            persona, args.model, out, index, args.quick, args.orchestrate, candidate_dir=folder
        )
    )
    print(f"transcript: {out}")


if __name__ == "__main__":
    main()
