"""MCP server exposing interview context. One process per interview (stdio).

Args: --candidate-dir DIR (required), --bank-dir DIR (ask from every file in DIR instead of the
candidate's questions.yaml; used by the evaluation tools), --notes-path FILE.
Debug with: ollmcp -j examples/ollmcp-servers.json -m qwen3:8b-16k
"""

import argparse
import json
from pathlib import Path

from mcp.server.mcpserver import MCPServer

from senjin.candidate import QUESTIONS, CandidateError, require_candidate_dir
from senjin.mcp_server.store import InterviewStore

store: InterviewStore  # built in main(); one interview per process

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


def build_store(
    candidate_dir: Path, bank_dir: Path | None = None, notes_path: Path | None = None
) -> InterviewStore:
    if bank_dir is not None:
        return InterviewStore(candidate_dir, bank_dir, notes_path)
    require_candidate_dir(candidate_dir, questions=True)
    return InterviewStore(candidate_dir, candidate_dir, notes_path, [Path(QUESTIONS).stem])


def main() -> None:
    global store
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-dir", type=Path)
    parser.add_argument("--bank-dir", type=Path)
    parser.add_argument("--notes-path", type=Path)
    args = parser.parse_args()
    try:
        require_candidate_dir(args.candidate_dir)
        store = build_store(args.candidate_dir, args.bank_dir, args.notes_path)
    except CandidateError as exc:
        raise SystemExit(f"error: {exc}") from None
    mcp.run()


if __name__ == "__main__":
    main()
