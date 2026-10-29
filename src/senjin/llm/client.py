"""Chat clients. The interviewer depends only on the `LLMClient` protocol."""

from collections.abc import AsyncIterator
from typing import Protocol

import ollama

DEFAULT_MODEL = "qwen3:8b-16k"
DEFAULT_HOST = "http://localhost:11434"

Message = dict[str, str]


class LLMClient(Protocol):
    def stream(self, messages: list[Message]) -> AsyncIterator[str]:
        """Yield the assistant reply as text chunks."""
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

    async def stream(self, messages: list[Message]) -> AsyncIterator[str]:
        # think=False: qwen3 reasoning tokens would be spoken aloud later and add latency.
        response = await self._client.chat(
            model=self.model,
            messages=messages,
            stream=True,
            think=False,
            options={"temperature": self.temperature},
        )
        async for chunk in response:
            text = chunk.message.content
            if text:
                yield text
