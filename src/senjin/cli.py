"""Terminal interview: `uv run senjin-chat [--model M] [--out PATH]`."""

import argparse
import asyncio
import os
import time
from contextlib import AsyncExitStack
from pathlib import Path

from senjin.interview.exemplars import ExemplarIndex
from senjin.interview.session import InterviewSession
from senjin.interview.transcript import TranscriptLog
from senjin.llm.client import DEFAULT_MODEL, OllamaClient
from senjin.llm.tools import open_mcp_tools


async def _say(stream) -> None:
    print("\nSen: ", end="", flush=True)
    async for chunk in stream:
        print(chunk, end="", flush=True)
    print()


async def run(model: str, out: Path, use_tools: bool, exemplars: Path | None = None) -> None:
    notes = out.with_suffix(".notes.jsonl")
    async with AsyncExitStack() as stack:
        tools = None
        if use_tools:
            env = {**os.environ, "SENJIN_NOTES_PATH": str(notes)}
            tools = await stack.enter_async_context(open_mcp_tools(env=env))
        session = InterviewSession(
            OllamaClient(model=model),
            TranscriptLog(out),
            tools=tools,
            exemplars=ExemplarIndex.load(exemplars) if exemplars else None,
        )
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Text-only Senjin interview")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--exemplars", type=Path, help="exemplar JSONL from senjin-data moves")
    parser.add_argument("--no-tools", action="store_true", help="skip the MCP server")
    args = parser.parse_args()
    out = args.out or Path("data/sessions") / f"{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
    asyncio.run(run(args.model, out, not args.no_tools, args.exemplars))


if __name__ == "__main__":
    main()
