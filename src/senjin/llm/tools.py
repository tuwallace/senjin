"""Tool runners: bridge from the model's tool calls to an MCP server."""

import sys
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Any, Protocol

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class ToolRunner(Protocol):
    tools: list[dict]

    async def call(self, name: str, arguments: dict[str, Any]) -> str: ...


class McpToolRunner:
    """Spawns an MCP server over stdio and exposes its tools in Ollama's tool format."""

    def __init__(self, session: ClientSession, tools: list[dict]) -> None:
        self._session = session
        self.tools = tools

    async def call(self, name: str, arguments: dict[str, Any]) -> str:
        result = await self._session.call_tool(name, arguments)
        text = "\n".join(getattr(block, "text", "") for block in result.content)
        return f"ERROR: {text}" if result.is_error else text


@asynccontextmanager
async def open_mcp_tools(
    command: str = sys.executable,
    args: list[str] | None = None,
    env: dict[str, str] | None = None,
) -> AsyncIterator[McpToolRunner]:
    params = StdioServerParameters(
        command=command, args=args or ["-m", "senjin.mcp_server.server"], env=env
    )
    async with AsyncExitStack() as stack:
        read, write = await stack.enter_async_context(stdio_client(params))
        session = await stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        listed = await session.list_tools()
        tools = [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description or "",
                    "parameters": t.input_schema,
                },
            }
            for t in listed.tools
        ]
        yield McpToolRunner(session, tools)
