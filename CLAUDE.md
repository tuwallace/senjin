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

## graphify

This project has a knowledge graph at graphify-out/ (git-ignored) with god nodes, community structure and cross-file relationships, plus a generated wiki at graphify-out/wiki/. The hand-written "why" lives in `Session_Logs.md` and `BRAINS.md` at the project root. There is **no `vault/` folder** (a per-symbol Obsidian export does not scale); open graphify-out/wiki/ in Obsidian if wanted.

### Reading (you do this)
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for a concept. These return a scoped subgraph, much smaller than GRAPH_REPORT.md or raw grep output.
- Before starting a coding task, read `Session_Logs.md` for prior decisions and unfinished work.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing. Read GRAPH_REPORT.md only for a broad architecture review.

### Keeping it current
- After a meaningful change run `graphify update .` (code only, free). A full rebuild or community naming needs a model: always pass `--backend ollama` so nothing leaves this Mac; see `BRAINS.md` for the exact sequence. The `graphify-vault-sync` skill runs update then the wiki export; run it at the end of a session or milestone, not mid-session.
- `.graphifyignore` keeps `data/` (interview data, caches) out of the graph. Never add interview data to the graph, the wiki or the logs.

### Writing declarative memory (you do this)
- Before ending a session or finishing a phase, append to `Session_Logs.md`: what was done, why, dead ends, next steps. Use Obsidian `[[Wikilinks]]` to connect related notes and docs.

