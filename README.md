# senjin
AI employee capable of joining a Google Meet, conducting a technical interview (asking questions, listening, and responding dynamically), and providing a post-interview assessment.

## Data and attribution

Phase 4 uses transcripts from the [Anthropic Interviewer dataset](https://huggingface.co/datasets/Anthropic/AnthropicInterviewer) (MIT per the dataset card; the card also mentions CC-BY, so attribute it) to study interviewer style. The raw data is downloaded locally to `data/` and is never committed. See `docs/ROADMAP.md` and `docs/COMMANDS.md`.

## Status

Phases 0 to 7 are built (tags `v0.1` to `v0.7`). Not yet run end to end against a real Google Meet with a human candidate; see `docs/COMMANDS.md`, `docs/KNOWN_ISSUES.md` and `docs/adr/`. Hiring recommendations are advisory only.
