# Repository Instructions & Memory Orchestration

Structural memory: ./graphify-out/  |  Declarative memory: ./graphify-out/wiki/ (generated) + ./Session_Logs.md (hand-maintained)

Before starting work: read ./Session_Logs.md, then query the graph instead of grepping the whole repo
(`graphify query`, `graphify explain`, `graphify path`).
After any meaningful change: `graphify update .` (free, code only). For a full rebuild: `graphify . --backend ollama` -> `graphify cluster-only . --backend ollama` -> `graphify export wiki --graph graphify-out/graph.json`.
Before ending a session: append to ./Session_Logs.md with what got done, decisions made, and what's next.
Use native [[Wikilinks]] when linking within Session_Logs.md.

Rules of this setup:
- `graphify-out/` is generated and git-ignored. `Session_Logs.md` is hand-written and committed, and sits at the project root, never inside `graphify-out/wiki/` (the wiki is regenerated on every export).
- Do not create a `vault/` folder: the per-symbol Obsidian export does not scale and can feed itself back into the graph. `.graphifyignore` lists `data/`, `.venv/` and media so interview data never enters the graph.
- Always pass `--backend ollama` for anything that calls a model, so no document leaves this Mac. (`graphify update .` needs no model.)
- Interview data (`data/`) is never part of the graph, the wiki or the session logs.
