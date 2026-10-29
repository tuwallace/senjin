# Senjin: Incremental Build Roadmap

Source spec: [AI Technical Interviewer Blueprint](AI%20Technical%20Interviewer%20Blueprint.md)
Remote: https://github.com/tuwallace/senjin (branch `main` exists; local folder is not yet a git repo)
Dataset: https://huggingface.co/datasets/Anthropic/AnthropicInterviewer

## 0. Ground truth and where we depart from the blueprint

Local machine (checked 2026-10-29): Apple M2 Pro, 12 cores, 32 GB RAM, macOS, plenty of disk.
Already installed: `ollama`, `ollmcp`, `uv`, `ffmpeg`, `whisper.cpp`, Docker, Node 22, Python 3.14 (use 3.12 via `uv` for ML wheels).
Models already pulled: `qwen3:8b` / `qwen3:8b-16k`, `llama3.1:8b-instruct-q8_0`, `qwen2.5:32b-instruct-q4_K_M`, `glm-4.7-flash`, `phi3`, `nomic-embed-text`.
Not installed: Piper, a virtual audio driver (BlackHole), Playwright.

| Blueprint assumption | Reality here | Decision |
| :-- | :-- | :-- |
| Docker Compose with NVIDIA GPU | Mac: Docker containers get no Metal GPU | Run **Ollama natively**. Docker is optional and only for the MCP server and tests, added late. |
| PulseAudio null sinks, v4l2loopback | Linux-only | Use **BlackHole 2ch** (two instances: one for bot-hearing, one for bot-speaking) or Loopback. Headed Chromium on the Mac. |
| `--use-file-for-fake-audio-capture` to speak | Plays a fixed file, not a live stream | Route TTS to a virtual output device that Chromium uses as its microphone. The file flag is a fallback for tests only. |
| Headless Chromium joins Meet | Google often blocks or challenges headless and automated browsers | Run **headed** Chromium with a persistent profile signed in to a dedicated bot Google account. Treat Meet integration as the riskiest step. |
| Anthropic dataset as "methodology" | It is 1,250 qualitative *research* interviews (workforce 1,000, creatives 125, scientists 125; columns `transcript_id`, `text`; CC-BY). It contains no technical interviews and no scores. | Use it for **interviewer behavior** (open-ended questions, probing follow-ups, neutral tone, pacing, wrap-up), not for candidate scoring. See Phase 4. |
| `llama3.1:8b` | Fine, but tool-calling quality varies | Default to `qwen3:8b-16k` for the live loop (tool use, 16k context). Use `qwen2.5:32b` as an offline judge for assessment. Benchmark in Phase 2. |

## Working agreements

- Every phase ends with a runnable demo and a git tag (`v0.N`). Nothing starts until the previous exit criteria pass.
- Build text-first, then add audio, then add Meet. Each layer is testable without the next.
- Everything runs locally. No cloud LLM calls in the product path.
- Consent: the bot announces it is an AI and that the call is transcribed. Interview data stays under `data/` (git-ignored).
- Hiring decisions: the assessment is advisory. Phase 7 includes a bias and fairness check before any real use.

## Target layout

```
senjin/
  docs/                     blueprint, roadmap, ADRs
  src/senjin/
    llm/                    Ollama client, streaming, model config
    interview/              state machine, prompts, question selection
    mcp_server/             FastMCP server: resume, JD, question bank
    audio/                  vad, stt, tts, devices, barge-in
    meet/                   Playwright joiner
    assess/                 schema, scorer, report
    eval/                   dataset loaders, simulated candidates, metrics
  data/                     git-ignored: dataset cache, transcripts, resumes
  question_bank/            YAML questions with rubrics
  tests/
  pyproject.toml            uv-managed
```

## Phase 0: Repo and tooling (0.5 day)

- `git init`, add `origin` = tuwallace/senjin, `git pull origin main` (remote already has a commit; reconcile before pushing). Add `.gitignore` (`data/`, `.venv/`, `*.wav`, `.DS_Store`).
- `uv init`, Python 3.12, `pytest`, `ruff`, `pydantic`, `httpx`, `ollama`, `mcp`.
- `brew install --cask blackhole-2ch` (needs sudo, user must run it), `brew install piper` or pip `piper-tts`, `uv run playwright install chromium`.
- Write `CLAUDE.md` with run commands, and add GitHub Actions for `ruff` and `pytest` (no model calls in CI).
- Exit: `uv run pytest` passes on an empty smoke test, CI green on `main`.

## Phase 1: Text-only interviewer in the terminal (v0.1, 2 days)

Goal: a working interview loop with no audio, no MCP, no Meet.

- `llm/`: streaming chat against `http://localhost:11434`, model name from config.
- `interview/`: explicit state machine `INTRO -> RESUME_DIVE -> TECHNICAL -> CANDIDATE_QA -> SIGNOFF`, with per-state turn and time budgets. The state machine, not the LLM, decides when to advance. The LLM only writes the next utterance.
- System prompt v1 (senior engineer persona): one question at a time, short spoken-style sentences (no markdown or lists, since this will be spoken), never reveals the rubric, never gives answers.
- Transcript logger writes JSONL per session to `data/sessions/`.
- Exit: you can complete a 15-minute typed interview, the transcript is saved, and states advance correctly under a scripted test using a fake LLM.

## Phase 2: MCP context via ollmcp (v0.2, 2 days)

- `mcp_server/` (FastMCP, stdio): tools `get_candidate_resume`, `get_job_description`, `get_next_technical_question(topic, difficulty, asked_ids)`, `record_note(text)`. Data lives in `data/candidates/` and `question_bank/*.yaml` (each question has id, topic, difficulty, prompt, follow-ups, rubric).
- Smoke-test the server interactively with `ollmcp -s src/senjin/mcp_server/server.py -m qwen3:8b-16k` before wiring it into code.
- Wire the same tools into the orchestrator through the Ollama tool-calling API (ollmcp is the dev/debug client; the orchestrator calls the same MCP server programmatically).
- Model bake-off on 20 scripted turns: tool-call correctness, latency to first token, instruction adherence. Candidates: `qwen3:8b-16k`, `llama3.1:8b-instruct-q8_0`, `glm-4.7-flash`. Record results in `docs/adr/0001-model-choice.md`.
- Exit: the model reliably fetches the resume, opens with a resume-specific question, and never repeats a question id. Tool-call success is at least 95% on the bake-off set.

## Phase 3: Post-interview assessment (v0.3, 1.5 days)

- `assess/schema.py`: Pydantic model matching the blueprint JSON, with changes: add `evidence` (quoted transcript spans with turn indices) for every score and weakness, per-question rubric scores, and `confidence`.
- Constrained decoding with Ollama structured outputs (`format=<json schema>`), then Pydantic validation, then one repair retry, then fail loudly.
- Run the judge with `qwen2.5:32b-instruct-q4_K_M` (offline, so speed does not matter) and compare to the live model.
- Reject any claim whose quoted evidence is not a substring of the transcript (anti-hallucination check from the blueprint's mitigations).
- Exit: 10 synthetic transcripts (good, mediocre, bad, evasive) produce valid JSON 100% of the time, scores are ordered correctly, and every evidence quote verifies.

## Phase 4: Use the Anthropic dataset (v0.4, 2 days)

The dataset's value is interviewer craft, so use it for three things:

1. **Style corpus.** Load via `datasets` (`Anthropic/AnthropicInterviewer`, splits `workforce`, `creatives`, `scientists`) into `data/hf_cache/`. Parse each `text` into turns, extract the interviewer turns, and cluster by move type (opener, probe, clarify, reflect, transition, wrap-up). Pick 20 to 30 exemplar moves per type.
2. **Prompt and few-shot library.** Distil the move types into the interviewer system prompt, with a few retrieved exemplars injected per turn (embed with `nomic-embed-text`, retrieve by move type and context). Adapt the wording from "AI at work" to technical topics.
3. **Evaluation.** Build a style metric: for held-out dataset turns, check whether our interviewer's question is open-ended, single-barrelled, and a probe of the prior answer (LLM-as-judge on `qwen2.5:32b`, calibrated against dataset turns as positives).

Also produce **simulated candidates**: seed a candidate persona (strong, average, weak) and have a second local model play them, so later phases can be tested without a human. Check the dataset license (CC-BY) and attribute it in the README. Do not commit the raw data.

- Exit: a report comparing v0.3 and v0.4 interviewer turns on the style metric, and a simulated end-to-end interview that runs unattended.

## Phase 5: Voice loop on the laptop, no Meet (v0.5, 4 days)

Prove audio with the built-in mic and speakers first.

- `audio/vad.py`: Silero VAD on 16 kHz frames. End-of-turn silence is configurable (the blueprint's 1.5 s is too slow; start at 0.8 s and tune).
- `audio/stt.py`: `faster-whisper` (CPU, int8) or `whisper.cpp` with Metal (already installed and likely faster on M2). Pick by measured real-time factor with `small.en` and `medium.en`.
- `audio/tts.py`: Piper with a medium voice. Sentence-level streaming: split the LLM token stream on sentence boundaries, synthesize and play sentence 1 while sentence 2 is generated.
- Barge-in: keep VAD running during playback. On confirmed speech (over 300 ms), stop playback, truncate the assistant turn in history to what was actually spoken, and continue. Use headphones during dev to avoid echo, then add echo handling in Phase 6.
- Latency budget: end of candidate speech to first audio under 2.5 s on average. Log per-stage timings (VAD, STT, LLM first token, TTS first chunk).
- Exit: a spoken 10-minute interview with the laptop mic, working barge-in, and a latency report at or under budget.

## Phase 6: Google Meet bot (v0.6, 4 to 5 days; highest risk)

- Install BlackHole twice (or use an aggregate setup): `BH-Hear` (browser output goes here, we record it) and `BH-Speak` (we play TTS here, browser input reads it). Set via Chromium audio flags or a macOS audio routing helper.
- `meet/joiner.py` with Playwright: headed Chromium, persistent profile on a dedicated bot account, `--use-fake-ui-for-media-stream`, join with camera off and mic on, handle "Ask to join", lobby wait with timeout, "someone is in the meeting" detection, and leave on end state or when the candidate leaves.
- Replace the laptop mic and speakers in the Phase 5 pipeline with the BlackHole devices.
- Echo and double-talk: record only `BH-Hear`, which does not contain the bot's own voice, so echo is largely avoided by design.
- Run against a Meet you host from a second account or device. Do not point it at third parties' meetings.
- Fallbacks if Google blocks automation: (a) have a human admit and keep the browser in a real user profile, (b) a Chrome extension for tab audio capture, (c) swap the transport for a self-hosted meeting tool (Jitsi) and keep everything else.
- Exit: the bot joins your Meet, greets, hears you, answers, handles one interruption, and leaves cleanly. Ten consecutive successful joins.

## Phase 7: Hardening, evaluation, and ship (v1.0, 1 week)

- Robustness: reconnect on network drop, silence watchdog ("Are you still there?"), time-boxed interview with a graceful wrap-up, crash-safe incremental transcript writes.
- Report output: Markdown and JSON scorecard, written to `data/reports/`, optional upload to a Drive folder or Slack by an explicit user action.
- Fairness and safety checks: strip names and demographic cues from the assessment input, run paired transcripts that differ only in cues (accent-transcribed text, gendered names) and check the score variance. Document limits in the README: advisory only, humans decide.
- Optional containerization: Dockerfile for the MCP server, tests, and assessment job. The Ollama server and audio stay native on macOS. Provide a Linux compose file (with the blueprint's NVIDIA block) as a separate target only if you later move to a GPU box.
- Exit: a full dress rehearsal with a human candidate on Meet, a reviewed report, and a `v1.0` tag.

## Dependency order

```
P0 -> P1 -> P2 -> P3
              \-> P4 (needs P1 prompt, feeds P2/P3 quality)
P1+P2+P4 -> P5 -> P6 -> P7
```

P3 and P4 can run in parallel after P2. P5 only needs P1 and P4's prompt, so it can start once P1 is stable if you want audio earlier.

## Main risks

| Risk | Phase | Mitigation |
| :-- | :-- | :-- |
| Google blocks or challenges automated sign-in and headless browsers | 6 | Headed browser, persistent profile, Jitsi fallback. Spike this early (a 1-day proof of join) in parallel with Phase 1. |
| 8B model drifts off script or hallucinates scores | 1, 3 | State machine owns control flow, structured output, evidence-quote verification, 32B judge. |
| End-to-end latency feels robotic | 5 | Sentence streaming, short replies, small VAD tail, Metal whisper, measure every stage. |
| Dataset mismatch (research interviews, not technical) | 4 | Use it only for style and probing behavior, and build a technical question bank separately. |
| Local memory pressure (32 GB: LLM + Whisper + TTS + Chromium) | 5, 6 | Keep the live model at 8B, load the 32B judge only after the call, set `OLLAMA_KEEP_ALIVE` deliberately. |
| Recording and consent law | 6, 7 | Spoken disclosure at the start, no storage without consent, retention policy in the README. |

## Suggested first week

1. Day 1: Phase 0, plus a 2-hour spike to confirm a Playwright browser can join a Meet you host.
2. Days 2 to 3: Phase 1.
3. Days 4 to 5: Phase 2, including the model bake-off.
