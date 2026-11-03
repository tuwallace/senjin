# senjin

A local AI technical interviewer. It joins a Google Meet, greets the candidate, asks questions drawn from the candidate's resume and the job description, listens, asks follow-ups, answers the candidate's own questions, and then writes an advisory assessment. Everything runs on this Mac: the language models (Ollama), speech recognition (whisper.cpp) and the voice (Piper). There are no cloud model calls in the product path.

## How it works

1. **Prepare a candidate folder** under `data/candidates/NAME/` with `resume.md` and `job_description.md`.
2. **Plan the interview.** `senjin-review --candidate-dir DIR` picks an interview shape (a profile from `profiles/`), lists the job's requirements, chooses and writes questions from `question_bank/`, and saves `interview.yaml`, `questions.yaml` and a coverage report. Edit them freely.
3. **Run the interview.** `senjin-meet run URL --candidate-dir DIR --assess` joins the Meet, interviews, hangs up and scores the transcript. Other ways to run it: `senjin-voice` (laptop microphone), `senjin-chat` (typed) and `senjin-sim` (a second model plays the candidate).
4. **Read the report** in `data/reports/`. It lists strengths, weaknesses, per-question scores and quoted evidence, and applies a fixed hiring bar.

Every command that reads a candidate needs `--candidate-dir`. See [docs/COMMANDS.md](docs/COMMANDS.md) for all commands and settings.

## Quick start

```bash
uv sync
ollama pull qwen3:8b-16k
uv run senjin-review --candidate-dir data/candidates/jane
uv run senjin-sim --candidate-dir data/candidates/jane      # try it without a call
uv run senjin-meet run https://meet.google.com/xxx-xxxx-xxx --candidate-dir data/candidates/jane --assess
```

The Meet bot needs a one-time setup (BlackHole audio devices, a signed-in Chrome profile, voice models); see the Phase 6 section of [docs/COMMANDS.md](docs/COMMANDS.md).

## Consent, privacy and retention

- **Disclosure.** Senjin says at the start that it is an AI and that the conversation is transcribed.
- **Local only.** Transcripts, resumes, reports and the models stay on this machine. Nothing is uploaded. The one outside service is Google Meet itself, which carries the call.
- **Retention.** Interview data lives under `data/` (git-ignored, never committed): `data/sessions/` (transcripts), `data/reports/`, and the candidate folders. Nothing is deleted automatically; delete a candidate's files when your retention period ends.
- **Fairness.** Names, contact details and demographic cues are removed from the resume before any model reads it, and questions written from the job description use no candidate data. Paired-transcript checks for score differences are in `senjin.eval.fairness` (see the fairness section of `docs/COMMANDS.md`).
- **Advisory only.** The assessment is a recommendation. A person makes every hiring decision and should read the transcript.

## Data and attribution

Interviewer style was studied using transcripts from the [Anthropic Interviewer dataset](https://huggingface.co/datasets/Anthropic/AnthropicInterviewer) (MIT per the dataset card; the card also mentions CC-BY, so attribute it). The raw data is downloaded locally to `data/` and is never committed.

## Status

Phases 0 to 8 are built (tags `v0.1` to `v0.8`; the work since `v0.8` is not yet tagged). The bot has run live in Google Meet interviews with a human candidate. `v1.0` is not tagged yet. See [docs/ROADMAP.md](docs/ROADMAP.md), [docs/KNOWN_ISSUES.md](docs/KNOWN_ISSUES.md), [docs/ENHANCEMENTS.md](docs/ENHANCEMENTS.md) and [docs/adr/](docs/adr/).
