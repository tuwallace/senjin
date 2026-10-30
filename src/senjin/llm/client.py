"""Chat clients. The interviewer depends only on the `LLMClient` protocol."""

import os
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Protocol

import ollama

DEFAULT_MODEL = "qwen3:8b-16k"
DEFAULT_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")  # container-friendly

Message = dict[str, Any]


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


Event = str | ToolCall


class LLMClient(Protocol):
    def stream(
        self, messages: list[Message], tools: list[dict] | None = None
    ) -> AsyncIterator[Event]:
        """Yield text chunks and tool calls as the model produces them."""
        ...


class OllamaClient:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        host: str = DEFAULT_HOST,
        temperature: float = 0.6,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self._client = ollama.AsyncClient(host=host)

    async def warm(self, messages: list[Message]) -> None:
        """Load the model and read the fixed start of the prompt, so the first real turn is fast."""
        await self._client.chat(
            model=self.model,
            messages=messages,
            stream=False,
            think=False,
            keep_alive="30m",
            options={"temperature": self.temperature, "num_predict": 1},
        )

    async def stream(
        self, messages: list[Message], tools: list[dict] | None = None
    ) -> AsyncIterator[Event]:
        # think=False: qwen3 reasoning tokens would be spoken aloud later and add latency.
        response = await self._client.chat(
            model=self.model,
            messages=messages,
            tools=tools or None,
            stream=True,
            think=False,
            keep_alive="30m",  # avoid multi-second reloads between turns
            options={"temperature": self.temperature},
        )
        async for chunk in response:
            if chunk.message.content:
                yield chunk.message.content
            for call in chunk.message.tool_calls or []:
                yield ToolCall(call.function.name, dict(call.function.arguments or {}))
