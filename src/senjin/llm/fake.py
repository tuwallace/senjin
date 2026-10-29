"""Scripted LLM for tests and offline demos."""

from collections.abc import AsyncIterator

from senjin.llm.client import Message


class FakeLLM:
    def __init__(self, replies: list[str]) -> None:
        self._replies = list(replies)
        self.calls: list[list[Message]] = []

    async def stream(self, messages: list[Message]) -> AsyncIterator[str]:
        self.calls.append([dict(m) for m in messages])
        reply = self._replies.pop(0) if self._replies else "Okay, tell me more."
        for word in reply.split(" "):
            yield word + " "
