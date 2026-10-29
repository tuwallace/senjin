# Senjin Commands

Run everything from the repo root (`/Users/twallace/source/senjin`) unless noted.
Phases marked **planned** have no code yet; their commands are the intended ones from [ROADMAP.md](ROADMAP.md) and will be updated as each phase lands.

| Phase | Status | Tag |
| :-- | :-- | :-- |
| 0 Repo and tooling | done | v0.0 (initial scaffold) |
| 1 Text interviewer | done | `v0.1` |
| 2 MCP context | done | `v0.2` |
| 3 Assessment | done | `v0.3` |
| 4 Anthropic dataset | planned | `v0.4` |
| 5 Voice loop | done | `v0.5` |
| 6 Google Meet bot | built, not run against real Meet | `v0.6` |
| 7 Hardening | built; `v1.0` waits for a live dress rehearsal | `v0.7` |

## Prerequisites (once)

```bash
ollama serve            # skip if the Ollama app is already running
ollama pull qwen3:8b-16k
uv sync                 # creates .venv with Python 3.12 and installs dependencies
```

Check Ollama is up: `curl -s http://localhost:11434/api/tags | head -c 200`

## Interview limits (environment variables)

By default Senjin asks 5 interview questions: 2 about the candidate's background and 3 technical (a question plus its follow-ups counts as turns). The greeting, the candidate's own Q&A and the goodbye are not counted. This keeps local model use small. Change the limits with environment variables:

| Variable | Default | Meaning |
| :-- | :-- | :-- |
| `SENJIN_MAX_QUESTIONS` | 5 | questions asked, background plus technical (2 to 30) |
| `SENJIN_RESUME_QUESTIONS` | 40% of the above, at least 1 | how many of those are about the resume |
| `SENJIN_MAX_MINUTES` | 45 | time box for the whole interview |
| `SENJIN_IDLE_SECONDS` | 25 | silence before "Are you still there?" (0 = off) |
| `SENJIN_SILENCE_MS` | 1000 | silence that ends the candidate's turn (200 to 3000); lower is snappier but may cut people off |
| `SENJIN_FILLER_MS` | 1100 | if no reply has started this long after the candidate stops, play a short "Okay, right." or "I see." (picked at random) (0 = off) |

```bash
export SENJIN_MAX_QUESTIONS=8            # for this terminal session
SENJIN_MAX_QUESTIONS=3 uv run senjin-voice run      # for one command
uv run senjin-meet run URL --max-questions 4        # a flag beats the variable
```

The candidate's own Q&A has no limit: Sen keeps answering until the candidate says they are done (for example "no, I'm good" or "that's all"), the time box (`SENJIN_MAX_MINUTES`) runs out, or the candidate goes silent.

The filler clip fills dead air: the clips are recorded with Piper at startup, play only when the reply is slow, never twice in a row, and are not part of what the model remembers (the transcript logs them as `filler` rows). Flags: `--silence-ms`, `--filler-ms` on `senjin-voice run`, `senjin-voice simulate` and `senjin-meet run`; `senjin-voice report` shows how often the filler played.

Keep your own settings between runs in a `.env` file (git-ignored) at the repo root:

```bash
cp .env.example .env        # then uncomment and edit the lines you want
```

Precedence, highest first: command-line flags, variables exported in your shell, `.env`, built-in defaults. Only `SENJIN_*` names are read from `.env`; `OLLAMA_HOST` must still be exported. The file is looked up in the current directory, then the repo root. Point at another file with `SENJIN_ENV_FILE=path`, or set `SENJIN_ENV_FILE=` (empty) to ignore `.env` files.

Bad values stop the program with a message naming the variable. Every `chat`, `voice run` and `meet run` prints the plan when it starts, for example `Interview plan: 5 questions (2 background + 3 technical)...`. Settings live in `src/senjin/config.py`.

## Phase 0: Repo and tooling

```bash
uv sync                                  # install dependencies
uv run pytest                            # all tests (no Ollama or audio needed)
uv run ruff check .                      # lint
uv run ruff format .                     # format
git remote -v                            # origin = github.com/tuwallace/senjin
git push origin main                     # CI (.github/workflows/ci.yml) runs lint + tests
```

Optional installs for later phases (the BlackHole install needs `sudo`, so run it yourself):

```bash
brew install --cask blackhole-2ch        # Phase 6: virtual audio device
uv add piper-tts                         # Phase 5: text to speech
uv add playwright && uv run playwright install chromium   # Phase 6
```

## Phase 1: Text-only interviewer

Start a typed interview with the default model and the MCP tools on:

```bash
uv run senjin-chat
```

Options:

```bash
uv run senjin-chat --model llama3.1:8b-instruct-q8_0   # pick another Ollama model
uv run senjin-chat --out data/sessions/mine.jsonl       # choose the transcript path
uv run senjin-chat --no-tools                           # Phase 1 mode: no MCP server
```

Type answers at the `You:` prompt. Stop with `quit`, `exit` or Ctrl-D.
Transcripts go to `data/sessions/<timestamp>.jsonl`, one JSON line per turn (`role`, `phase`, `text`); tool calls appear as `role: "tool"`.

Inspect the last transcript:

```bash
ls -t data/sessions/*.jsonl | head -1 | xargs python3 -c "import json,sys; [print(f\"{r['role']:>9} [{r['phase']}] {r['text'][:120]}\") for r in map(json.loads, open(sys.argv[1]))]"
```

Tune the phase turn and time budgets in `src/senjin/interview/state.py` (`DEFAULT_BUDGETS`) and the persona in `src/senjin/interview/prompts.py`.

## Phase 2: MCP server and tool calling

The server exposes `get_candidate_resume`, `get_job_description`, `get_next_technical_question` and `record_note`. `senjin-chat` starts it automatically as a child process.

Try the tools interactively with ollmcp:

```bash
ollmcp -j examples/ollmcp-servers.json -m qwen3:8b-16k
```

Then ask, for example, "Get the candidate's resume" or "Give me a hard database question".

Run the server alone (stdio; useful for debugging startup errors):

```bash
uv run python -m senjin.mcp_server.server
```

Use your own candidate and question bank:

```bash
export SENJIN_CANDIDATE_DIR=data/candidates/jane      # needs resume.md and job_description.md
export SENJIN_QUESTION_BANK=question_bank             # *.yaml files
export SENJIN_NOTES_PATH=data/sessions/notes.jsonl    # where record_note writes
uv run senjin-chat
```

Question format (`question_bank/*.yaml`): `id`, `topic`, `difficulty` (easy, medium, hard), `prompt`, `follow_ups`, `rubric`. The rubric is never shown to the live model.

Compare models on tool calling (about a minute per model):

```bash
uv run python -m senjin.eval.bakeoff qwen3:8b-16k llama3.1:8b-instruct-q8_0
```

Results are recorded in [adr/0001-model-choice.md](adr/0001-model-choice.md).

## Phase 3: Post-interview assessment

Score a saved transcript with the local judge (`qwen2.5:32b-instruct-q4_K_M`, about 19 GB; the first call loads it and takes a while):

```bash
uv run senjin-assess data/sessions/<file>.jsonl
```

Outputs `data/reports/<file>.json` (full report with verification stats) and `data/reports/<file>.md` (readable scorecard).

Options:

```bash
uv run senjin-assess data/sessions/<file>.jsonl --judge qwen3:8b-16k     # faster, weaker judge
uv run senjin-assess data/sessions/<file>.jsonl --out-dir /tmp/reports    # choose where reports go
```

Try it without running an interview, using the synthetic transcripts (`strong`, `mediocre`, `weak`, `evasive`):

```bash
uv run senjin-assess examples/transcripts/strong.jsonl
```

Check judge quality (4 transcripts x 3 runs; about 10 minutes with the 32B judge):

```bash
uv run python -m senjin.eval.assess_eval --runs 3
uv run python -m senjin.eval.assess_eval --runs 1 --judge qwen3:8b-16k
```

It prints one line per run, the valid-JSON rate, and PASS or FAIL for the score ordering (strong > mediocre > weak, evasive below mediocre).

How it stays honest: output is schema-constrained, retried once if invalid, and every cited quote is checked against the candidate's actual words. Claims with invented quotes are dropped and confidence is set to low. If nothing verifies, the command fails instead of writing a report.

Unit tests (fake judge, no Ollama): `uv run pytest tests/test_assess.py`

## Phase 4: Anthropic dataset

The dataset (1,250 research interviews, MIT/CC-BY) is used for interviewer craft only: follow-up style, not technical content or scoring.

Download (about 11 MB into `data/hf_cache/`, git-ignored) and inspect:

```bash
uv run senjin-data fetch
uv run senjin-data stats        # transcripts per split and interviewer move counts
```

Build the exemplar library (needs `ollama pull nomic-embed-text`; writes `data/exemplars/moves.jsonl`):

```bash
uv run senjin-data moves                 # 40 exemplars each of probe, clarify, reflect
uv run senjin-data moves --per-kind 80   # a bigger library
```

Use exemplars in an interview (retrieved per turn in the background and technical phases):

```bash
uv run senjin-chat --exemplars data/exemplars/moves.jsonl
```

Run an unattended interview with a simulated candidate (`strong`, `average` or `weak`):

```bash
uv run senjin-sim --persona average --quick                                    # about 9 turns, no exemplars
uv run senjin-sim --persona strong --quick --exemplars data/exemplars/moves.jsonl
uv run senjin-sim --persona weak                                               # full-length interview
uv run senjin-sim --persona average --quick --orchestrate                      # code-driven questions, as in voice and Meet
```

Each run prints the transcript path (under `data/sessions/`), which you can score with `senjin-assess`.

Compare interviewer style (single question, open-ended, ties to the candidate's answer, speakable, length):

```bash
uv run python -m senjin.eval.style                                  # dataset baseline
uv run python -m senjin.eval.style data/sessions/a.jsonl data/sessions/b.jsonl
```

Tests: `uv run pytest tests/test_dataset.py`

Attribution: Anthropic Interviewer transcripts, https://huggingface.co/datasets/Anthropic/AnthropicInterviewer. Raw data and exemplars stay in `data/` and are not committed.

## Phase 5: Voice loop on the laptop

Stack: Silero VAD (ONNX) -> whisper.cpp server (Metal) -> Ollama -> Piper TTS -> speakers, with barge-in.

One-time setup (downloads the Silero VAD and a Piper voice, about 65 MB, into `data/models/`; whisper models are read from `~/.cache/whisper-models/`):

```bash
uv run senjin-voice setup
uv run senjin-voice devices          # list microphones and speakers
```

Check the audio stack without a microphone (speaks a sentence, transcribes it back, runs the VAD):

```bash
uv run senjin-voice check
uv run senjin-voice check --stt-model ggml-large-v3-turbo-q5_0.bin   # slower, more accurate
uv run senjin-voice check --tts say                                  # macOS voice instead of Piper
```

Run the full pipeline with synthetic candidate speech and no microphone or speaker (real VAD, STT, LLM, TTS; prints a latency summary):

```bash
uv run senjin-voice simulate
uv run senjin-voice simulate --turns 3 --silence-ms 600
```

Hold a spoken interview. Use headphones, or the interviewer will hear itself; macOS asks for microphone permission the first time:

```bash
uv run senjin-voice run
uv run senjin-voice run --silence-ms 600        # end your turn after 0.6 s of silence (default 0.8 s)
uv run senjin-voice run --no-barge-in           # speakers instead of headphones: ignore the mic while Sen talks
uv run senjin-voice run --model llama3.1:8b-instruct-q8_0 --stt-model ggml-base.en.bin
```

Stop with Ctrl-C. Outputs go to `data/sessions/voice-<timestamp>.jsonl` (transcript) and `.latency.jsonl` (per-turn timings). Summarize a latency file:

```bash
uv run senjin-voice report data/sessions/voice-<timestamp>.latency.jsonl
```

Barge-in: if you speak for about 300 ms while Sen is talking, playback stops at once and the transcript keeps only the words Sen actually said (an `interrupt` row records them).

Tests (no audio hardware or models needed): `uv run pytest tests/test_voice.py`

## Phase 6: Google Meet bot

Status: built and tested against a mock Meet page and Chrome's fake audio devices. **Not yet run against a real Meet or real BlackHole devices**; do the steps below in order, and expect to adjust `src/senjin/meet/selectors.py` on first contact with real Meet.

How audio flows: Chrome's microphone is forced to `BlackHole 2ch`, and the voice loop plays Piper speech into it. Chrome's speaker is forced to `BlackHole 16ch`, and the voice loop records the candidate from it. The bot's own voice never reaches the recording, so there is no echo. You will not hear the call yourself; join it from another device or account to listen.

1. Install the virtual audio drivers (needs your password; reboot or log out and in if macOS does not list them):

```bash
brew install --cask blackhole-2ch blackhole-16ch
```

2. Confirm both devices are visible:

```bash
uv run senjin-meet devices
```

3. Prove the audio bridge works, with no Google account or meeting (opens a Chrome window, speaks, listens, prints PASS or FAIL in each direction):

```bash
uv run senjin-meet selftest
```

4. Sign in the Google account the bot should use. A Chrome window opens and you type the credentials yourself; the session is saved in `data/meet_profile/` (git-ignored). A dedicated account is recommended:

```bash
uv run senjin-meet login
```

Switch the bot to a different Google account (stop any running `senjin-meet` first):

```bash
uv run senjin-meet logout                          # delete the saved sign-in (data/meet_profile)
uv run senjin-meet keep-account                    # set the current sign-in aside, then sign in a new one
uv run senjin-meet keep-account --label work       # same, saved as data/meet_profile_work
```

`keep-account` moves the old profile to `data/meet_profile_<label>` instead of deleting it. To go back to it: `uv run senjin-meet logout`, then `mv data/meet_profile_work data/meet_profile`.

Find out why the bot cannot hear the call (joins, then prints the level recorded from the hear device and what Meet's audio elements are doing every 5 seconds; speak in the meeting while it runs):

```bash
uv run senjin-meet debug "https://meet.google.com/xxx-xxxx-xxx" --seconds 120
```

5. Create a Meet from another account or device (the host), then send the bot in. The host must admit the bot from the lobby:

```bash
uv run senjin-meet run "https://meet.google.com/xxx-xxxx-xxx"
```

Options:

```bash
uv run senjin-meet run URL --candidate-dir data/candidates/jane   # real resume.md + job_description.md (default: the sample Jane Doe)
uv run senjin-meet run URL --wait-candidate 120      # seconds to wait for a second participant before starting
uv run senjin-meet run URL --admit-timeout 600       # seconds to wait in the lobby
uv run senjin-meet run URL --silence-ms 600          # end-of-turn silence
uv run senjin-meet run URL --model llama3.1:8b-instruct-q8_0 --stt-model ggml-base.en.bin
uv run senjin-meet run URL --speak-device "BlackHole 2ch" --hear-device "BlackHole 16ch"
```

The bot says it is an AI and that the call is transcribed in its first sentence. Only join meetings you host or have permission to join. The run ends after the sign-off or when the call ends; the transcript is `data/sessions/meet-<timestamp>.jsonl` and can be scored with `senjin-assess`.

Tests (Chrome required; they are skipped when it is missing): `uv run pytest tests/test_meet.py`

If joining fails: check `src/senjin/meet/selectors.py` first (Meet changes its labels), then run with a visible window (the default) and watch where it stops.

## Phase 7: Hardening

Reliability, built into `senjin-voice run` and `senjin-meet run`:

```bash
uv run senjin-voice run --max-minutes 30 --idle-seconds 20
uv run senjin-meet run URL --max-minutes 30 --idle-seconds 20
```

- Time box: at 85% of `--max-minutes` the interviewer skips to the candidate's questions; at 100% it signs off. Default 45.
- Silence watchdog: after `--idle-seconds` of silence (default 25) it says "Are you still there?", then once more, then ends the call politely. Use `--idle-seconds 0` to turn it off.
- Errors: if a reply fails (for example Ollama stops), it apologizes and asks the candidate to repeat; after 3 failures in a row it ends the call and says someone will follow up.
- Meet drops: if the call UI disappears and the interview is not finished, the bot rejoins up to 2 times (waiting up to 2 minutes each time), then gives up.
- Transcripts are written line by line, so a crash keeps everything said so far.

Scorecard (Markdown and JSON in `data/reports/`):

```bash
uv run senjin-assess data/sessions/<file>.jsonl
uv run senjin-meet run URL --assess                  # score automatically when the call ends
```

Personal cues are removed before the judge reads the transcript: the candidate's name (taken from `resume.md`'s first heading, or `--candidate-name "Jane Doe"`), gendered pronouns, and whole sentences about family, age, origin or identity. The reports show the redacted text. To score without redaction (for comparison only):

```bash
uv run senjin-assess data/sessions/<file>.jsonl --no-redact
```

Fairness checks:

```bash
uv run pytest -m fairness                      # offline: redacted paired transcripts are identical
uv run python -m senjin.eval.fairness          # live: 6 name/cue variants x 2 transcripts, with and without redaction (20+ minutes)
uv run python -m senjin.eval.fairness --bases strong --judge qwen3:8b-16k
```

The live check prints each score and a technical-score spread per group. With redaction on, the spread must be 0.

Containers (tests, assessment and the MCP server only; Ollama, audio and the Meet bot stay native on macOS):

```bash
docker build -t senjin:dev .
docker run --rm senjin:dev                                           # runs the test suite
docker run --rm -v "$PWD/data:/app/data" senjin:dev senjin-assess data/sessions/<file>.jsonl
```

Inside Docker, Ollama is reached at `http://host.docker.internal:11434` (set `OLLAMA_HOST` to change it).

Not included: uploading reports to Drive or Slack. Reports stay local until a person shares them.

## Git and release

```bash
git status
git add -A && git commit -m "message"
git tag v0.N && git push origin main v0.N            # tag each finished phase
```

## Troubleshooting

| Symptom | Fix |
| :-- | :-- |
| `Connection refused` on 11434 | Start Ollama: `ollama serve` |
| First reply is slow | The model is loading; later turns are faster. Keep it warm with `OLLAMA_KEEP_ALIVE=30m` |
| Reply is off-topic on the first turn | Use a model with a standard chat template; see the opening-prompt note in `session.py` |
| Tool never called | Re-run the bake-off; try `llama3.1:8b-instruct-q8_0` |
| `uv: command not found` | `brew install uv` |
