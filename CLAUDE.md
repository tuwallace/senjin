# Senjin

Local AI technical interviewer (Google Meet bot). Plan: `docs/ROADMAP.md`. Spec: `docs/AI Technical Interviewer Blueprint.md`.

## Commands
- Setup: `uv sync`
- Test: `uv run pytest`
- Lint/format: `uv run ruff check . && uv run ruff format .`
- Questions for one candidate: `uv run senjin-review --candidate-dir DIR [--dry-run]`. Every command that reads a candidate needs `--candidate-dir` (no environment variable); see `docs/COMMANDS.md`
- MCP debug client (from repo root): `ollmcp -j examples/ollmcp-servers.json -m qwen3:8b-16k`

## Conventions
- Python 3.12, `src/` layout, package `senjin`.
- Everything runs locally; no cloud LLM calls in the product path. Ollama runs natively (not in Docker) on macOS.
- Tests must not call Ollama or touch audio devices; use fakes. CI runs no model calls.
- `data/` is git-ignored (transcripts, resumes, HF cache). Never commit interview data.
- Work phase by phase per the roadmap; tag each phase `v0.N`.
- Each candidate's interview shape (question counts, follow-ups, time box) comes from `<candidate-dir>/interview.yaml`, written by `senjin-review` from a profile in `profiles/` (`src/senjin/plan.py`). Flags override it. `SENJIN_*` settings in `src/senjin/config.py` (a git-ignored `.env`, see `.env.example`) cover audio timing and tools that read no candidate. See `docs/COMMANDS.md`.
