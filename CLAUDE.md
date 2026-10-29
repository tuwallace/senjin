# Senjin

Local AI technical interviewer (Google Meet bot). Plan: `docs/ROADMAP.md`. Spec: `docs/AI Technical Interviewer Blueprint.md`.

## Commands
- Setup: `uv sync`
- Test: `uv run pytest`
- Lint/format: `uv run ruff check . && uv run ruff format .`
- MCP debug client (from repo root): `ollmcp -j examples/ollmcp-servers.json -m qwen3:8b-16k`

## Conventions
- Python 3.12, `src/` layout, package `senjin`.
- Everything runs locally; no cloud LLM calls in the product path. Ollama runs natively (not in Docker) on macOS.
- Tests must not call Ollama or touch audio devices; use fakes. CI runs no model calls.
- `data/` is git-ignored (transcripts, resumes, HF cache). Never commit interview data.
- Work phase by phase per the roadmap; tag each phase `v0.N`.
