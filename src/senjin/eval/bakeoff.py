"""Model bake-off: does each model call the right tool at the right moment?

Usage: uv run python -m senjin.eval.bakeoff qwen3:8b-16k llama3.1:8b-instruct-q8_0
Needs Ollama running. Prints tool-call accuracy and time to first token per scenario.
"""

import asyncio
import sys
import time

from senjin.interview.prompts import SYSTEM_PROMPT, phase_message
from senjin.interview.state import Phase
from senjin.llm.client import OllamaClient, ToolCall
from senjin.llm.tools import open_mcp_tools

# (phase, history, acceptable first actions: tool names, or None for "speak, no tool")
SCENARIOS = [
    (Phase.RESUME_DIVE, [{"role": "user", "content": "Yes, I'm ready."}], {"get_candidate_resume"}),
    (
        Phase.TECHNICAL,
        [{"role": "user", "content": "Sure, let's move on to the technical part."}],
        {"get_next_technical_question"},
    ),
    (
        Phase.INTRO,
        [{"role": "user", "content": "[The call has connected. Begin the interview.]"}],
        {None},
    ),
    (
        Phase.CANDIDATE_QA,
        [{"role": "user", "content": "What does the on-call rotation look like?"}],
        {None, "get_job_description"},  # looking up the role before answering is fine
    ),
]
REPEATS = 3


async def run_model(model: str, tools: list[dict]) -> None:
    llm = OllamaClient(model=model)
    ok = total = 0
    ttft: list[float] = []
    for phase, history, expected in SCENARIOS:
        for _ in range(REPEATS):
            instruction = phase_message(phase, entering=False)
            system = {
                "role": "system",
                "content": f"{SYSTEM_PROMPT}\nCurrent instruction: {instruction}",
            }
            start, first, called = time.monotonic(), None, None
            async for event in llm.stream([system, *history], tools):
                first = first or time.monotonic() - start
                if isinstance(event, ToolCall):
                    called = event.name
            ttft.append(first or 0.0)
            total += 1
            ok += called in expected
    print(f"{model:36} tool accuracy {ok}/{total}  mean first-event {sum(ttft) / len(ttft):.2f}s")


async def main(models: list[str]) -> None:
    async with open_mcp_tools() as runner:
        for model in models:
            await run_model(model, runner.tools)


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1:] or ["qwen3:8b-16k"]))
