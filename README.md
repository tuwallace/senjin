# senjin
AI employee capable of joining a Google Meet, conducting a technical interview (asking questions, listening, and responding dynamically), and providing a post-interview assessment.

## Data and attribution

Phase 4 uses transcripts from the [Anthropic Interviewer dataset](https://huggingface.co/datasets/Anthropic/AnthropicInterviewer) (MIT per the dataset card; the card also mentions CC-BY, so attribute it) to study interviewer style. The raw data is downloaded locally to `data/` and is never committed. See `docs/ROADMAP.md` and `docs/COMMANDS.md`.
