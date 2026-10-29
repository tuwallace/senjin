"""Scripted LLM for tests and offline demos."""

from collections.abc import AsyncIterator

from senjin.llm.client import Event, Message, ToolCall

Reply = str | list[Event]


class FakeLLM:
    """Each scripted reply is a string or a list of events (text chunks and ToolCalls)."""

    def __init__(self, replies: list[Reply]) -> None:
        self._replies = list(replies)
        self.calls: list[list[Message]] = []
        self.tools_seen: list[list[dict] | None] = []

    async def stream(
        self, messages: list[Message], tools: list[dict] | None = None
    ) -> AsyncIterator[Event]:
        self.calls.append([dict(m) for m in messages])
        self.tools_seen.append(tools)
        reply = self._replies.pop(0) if self._replies else "Okay, tell me more."
        if isinstance(reply, str):
            for word in reply.split(" "):
                yield word + " "
        else:
            for event in reply:
                yield event


__all__ = ["FakeLLM", "ToolCall"]
