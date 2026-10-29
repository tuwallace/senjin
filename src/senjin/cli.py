"""Terminal interview: `uv run senjin-chat [--model M] [--out PATH]`."""

import argparse
import asyncio
import time
from pathlib import Path

from senjin.interview.session import InterviewSession
from senjin.interview.transcript import TranscriptLog
from senjin.llm.client import DEFAULT_MODEL, OllamaClient


async def _say(stream) -> None:
    print("\nSen: ", end="", flush=True)
    async for chunk in stream:
        print(chunk, end="", flush=True)
    print()


async def run(model: str, out: Path) -> None:
    session = InterviewSession(OllamaClient(model=model), TranscriptLog(out))
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
    args = parser.parse_args()
    out = args.out or Path("data/sessions") / f"{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
    asyncio.run(run(args.model, out))


if __name__ == "__main__":
    main()
