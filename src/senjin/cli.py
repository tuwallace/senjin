"""Terminal interview: `uv run senjin-chat [--model M] [--out PATH]`."""

import argparse
import asyncio
import time
from contextlib import AsyncExitStack
from pathlib import Path

from senjin.candidate import JOB_DESCRIPTION, RESUME, add_candidate_arg, candidate_from_args
from senjin.config import InterviewConfig, exit_on_config_error
from senjin.interview.exemplars import ExemplarIndex
from senjin.interview.session import InterviewSession
from senjin.interview.transcript import TranscriptLog
from senjin.llm.client import DEFAULT_MODEL, OllamaClient
from senjin.llm.tools import open_mcp_tools
from senjin.plan import PLAN, InterviewPlan, plan_from_args


async def _say(stream) -> None:
    print("\nSen: ", end="", flush=True)
    async for chunk in stream:
        print(chunk, end="", flush=True)
    print()


async def run(
    model: str,
    out: Path,
    use_tools: bool,
    exemplars: Path | None = None,
    max_questions: int | None = None,
    candidate_dir: Path | None = None,
    plan: InterviewPlan | None = None,
) -> None:
    base = InterviewConfig.from_env()
    cfg = (plan.apply(base) if plan else base).override(max_questions)
    notes = out.with_suffix(".notes.jsonl")
    async with AsyncExitStack() as stack:
        tools = None
        if use_tools:
            assert candidate_dir is not None
            tools = await stack.enter_async_context(open_mcp_tools(candidate_dir, notes_path=notes))
        session = InterviewSession(
            OllamaClient(model=model),
            TranscriptLog(out),
            state=cfg.state(),
            tools=tools,
            exemplars=ExemplarIndex.load(exemplars) if exemplars else None,
        )
        print(cfg.describe())
        print(f"Transcript: {out}\nType your answers. Ctrl-D or 'quit' to stop.")
        await _say(session.open())
        while not session.finished:
            try:
                text = (await asyncio.to_thread(input, "\nYou: ")).strip()
            except EOFError:
                break
            if text.lower() in {"quit", "exit"}:
                break
            if text:
                await _say(session.respond(text))


@exit_on_config_error
def main() -> None:
    parser = argparse.ArgumentParser(description="Text-only Senjin interview")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--exemplars", type=Path, help="exemplar JSONL from senjin-data moves")
    parser.add_argument("--max-questions", type=int, help="env SENJIN_MAX_QUESTIONS, default 5")
    parser.add_argument("--no-tools", action="store_true", help="skip the MCP server")
    add_candidate_arg(parser)
    args = parser.parse_args()
    candidate_dir = plan = None
    if not args.no_tools:  # the tools read the candidate's files and questions
        candidate_dir = candidate_from_args(
            args.candidate_dir, files=(RESUME, JOB_DESCRIPTION), questions=True
        )
        plan = plan_from_args(candidate_dir)
        print(plan.describe(candidate_dir / PLAN))
    out = args.out or Path("data/sessions") / f"{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
    asyncio.run(
        run(
            args.model,
            out,
            not args.no_tools,
            args.exemplars,
            args.max_questions,
            candidate_dir,
            plan,
        )
    )


if __name__ == "__main__":
    main()
