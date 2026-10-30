"""MCP server exposing interview context. One process per interview (stdio).

Env: SENJIN_CANDIDATE_DIR, SENJIN_QUESTION_BANK, SENJIN_QUESTION_SET, SENJIN_NOTES_PATH.
Debug with: ollmcp -s src/senjin/mcp_server/server.py -m qwen3:8b-16k
"""

import json
import os
from pathlib import Path

from mcp.server.mcpserver import MCPServer

from senjin.mcp_server.store import InterviewStore

ROOT = Path(__file__).resolve().parents[3]

store = InterviewStore(
    candidate_dir=Path(os.environ.get("SENJIN_CANDIDATE_DIR", ROOT / "examples/candidate")),
    bank_dir=Path(os.environ.get("SENJIN_QUESTION_BANK", ROOT / "question_bank")),
    notes_path=Path(p) if (p := os.environ.get("SENJIN_NOTES_PATH")) else None,
    question_sets=[
        n.strip() for n in os.environ.get("SENJIN_QUESTION_SET", "").split(",") if n.strip()
    ]
    or None,
)

mcp = MCPServer("senjin-interview")


@mcp.tool()
def get_candidate_resume() -> str:
    """Return the candidate's resume. Call once at the start of the background discussion."""
    return store.resume()


@mcp.tool()
def get_job_description() -> str:
    """Return the job description for the role being interviewed for."""
    return store.job_description()


@mcp.tool()
def get_next_technical_question(topic: str | None = None, difficulty: str | None = None) -> str:
    """Get the next unasked technical question and mark it asked.

    topic: system_design, algorithms, databases, or engineering_practice (optional).
    difficulty: easy, medium, or hard (optional).
    Returns JSON with the question prompt and optional follow-ups.
    """
    return json.dumps(store.next_question(topic, difficulty))


@mcp.tool()
def record_note(text: str) -> str:
    """Save a short private note about the candidate's answer for the later assessment."""
    return store.record_note(text)


if __name__ == "__main__":
    mcp.run()
