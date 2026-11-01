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
| `SENJIN_MAX_MINUTES` | scaled | time box for the whole interview. Unset, it scales with the question count: about 2 minutes per interviewer turn (a question, or a follow-up) plus 10 for the greeting, the candidate's questions and the goodbye, never below the phase caps (5 minutes background, 15 technical). The default 5 questions give 30 minutes; 18 questions (6 background) give 70. |
| `SENJIN_IDLE_SECONDS` | 25 | silence before "Are you still there?" (0 = off) |
| `SENJIN_SILENCE_MS` | 1000 | silence that ends the candidate's turn (200 to 3000); lower is snappier but may cut people off |
| `SENJIN_FILLER_MS` | 1100 | if no reply has started this long after the candidate stops, play a short "Okay, right." or "I see." (picked at random) (0 = off) |
| `SENJIN_HOLD_MS` | 1500 | after a short (under 14 words) or trailing-off answer to a background or technical question, wait this much longer for the candidate to carry on before replying; if they do, both parts are answered as one (0 = off) |
| `SENJIN_FOLLOW_UPS` | 1 | follow-ups after each technical question (0 to 3). `SENJIN_MAX_QUESTIONS` counts main questions, so the default asks 2 background questions and 3 separate technical questions, each followed by one follow-up |

```bash
export SENJIN_MAX_QUESTIONS=8            # for this terminal session
SENJIN_MAX_QUESTIONS=3 uv run senjin-voice run --candidate-dir data/candidates/jane      # for one command
uv run senjin-meet run URL --max-questions 4 --candidate-dir data/candidates/jane        # a flag beats the variable
```

The candidate's own Q&A has no limit: Senjin keeps answering until the candidate says they are done (for example "no, I'm good" or "that's all"), the time box (`SENJIN_MAX_MINUTES`) runs out, or the candidate goes silent.

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
uv run senjin-chat --candidate-dir data/candidates/jane
```

Options:

```bash
uv run senjin-chat --model llama3.1:8b-instruct-q8_0 --candidate-dir data/candidates/jane   # pick another Ollama model
uv run senjin-chat --out data/sessions/mine.jsonl --candidate-dir data/candidates/jane       # choose the transcript path
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
uv run python -m senjin.mcp_server.server --candidate-dir examples/candidate --bank-dir question_bank
```

`--candidate-dir` is required. The server asks from `<candidate-dir>/questions.yaml`, or from every file in `--bank-dir` if you give one (the evaluation tools do). `--notes-path FILE` is where `record_note` writes.

Use your own candidate (see "Interview questions for one candidate" below; run `senjin-review` first):

```bash
uv run senjin-chat --candidate-dir data/candidates/jane
```

Question format (`question_bank/*.yaml`): `id`, `topic`, `difficulty` (easy, medium, hard), `prompt`, `follow_ups`, `rubric`. The rubric is never shown to the live model.

Compare models on tool calling (about a minute per model):

```bash
uv run python -m senjin.eval.bakeoff qwen3:8b-16k llama3.1:8b-instruct-q8_0
```

Results are recorded in [adr/0001-model-choice.md](adr/0001-model-choice.md).

## Interview questions for one candidate: `senjin-review`

Assembles the technical questions for one candidate from the shared bank (`question_bank/*.yaml`) and their resume. Run it before any interview.

```bash
uv run senjin-review --candidate-dir data/candidates/taft
uv run senjin-review --candidate-dir data/candidates/taft --dry-run   # show the questions, write nothing
uv run senjin-review --candidate-dir data/candidates/taft --force     # overwrite an existing questions.yaml
```

**The candidate folder is never guessed.** `senjin-review`, `senjin-assess`, `senjin-meet run`, `senjin-voice run`, `senjin-voice simulate` and `senjin-chat` all need `--candidate-dir`; without it, or if the folder does not exist, they stop with an `error:` line. `senjin-review` also needs `resume.md` and `job_description.md` in it (if `resume.md` is missing but another `.md` file is there, the error names it so you can rename it), and the interview commands need the `questions.yaml` that `senjin-review` writes.

**The interview plan.** The shape of each interview comes from the job, not from global settings. `senjin-review` first reads the job description, has the judge pick a profile from `profiles/` (`junior_generalist`, `senior_fullstack`, `backend`, `frontend`, `data_engineering`, `cloud_security`, `qa_program_lead`, `staff_architect`, or `default`), and saves it as `<candidate-dir>/interview.yaml`: background questions (asked by the model from `resume.md`), technical questions, follow-ups per question, minutes per turn and the resulting time box. A staff role gets fewer, deeper questions (4 technical, 2 follow-ups, 50 minutes) and a junior role a shorter one (31 minutes). The profile is chosen from the job description only, so every candidate for the same job gets the same shape. A saved plan is read back as it is, including your edits; `--profile NAME` switches it, `--refresh-plan` chooses again, `--plan-only` writes it and stops, `--list-profiles` shows the options.

`senjin-review` then writes the technical questions for the plan's count: half from the shared bank, half written for the job description (`--bank-questions` and `--job-questions` override the split). The interview commands (`senjin-meet run`, `senjin-voice run` and `simulate`, `senjin-chat`) stop with an error if `interview.yaml` is missing, or if `questions.yaml` has fewer questions than the plan asks for.

Precedence for the interview's shape: command-line flags, then `interview.yaml`, then `SENJIN_MAX_QUESTIONS`, `SENJIN_RESUME_QUESTIONS`, `SENJIN_FOLLOW_UPS` and `SENJIN_MAX_MINUTES`, which now only apply to tools that read no candidate (`senjin-chat --no-tools`, `senjin-sim`). Audio settings (`SENJIN_SILENCE_MS`, `HOLD_MS`, `FILLER_MS`, `IDLE_SECONDS`) are never part of the plan.

What it does, with one local model (default the judge model, `--model` to change it):

1. **Lists the job's requirements.** Must-have and nice-to-have items, skipping eligibility rules, benefits, years of experience, degrees and "prior experience" items. A preferred-qualifications section is found in code and read on its own, so its items are nice-to-have. The list is saved as `<candidate-dir>/requirements.yaml` and read back on later runs instead of asking the model again, so a run does not depend on what the model happened to list.
2. **Picks the shared files.** The judge reads the job description and the question files in `question_bank/` and chooses the ones that suit the job.
3. **Picks the best shared questions by skills coverage.** Every bank question has `skills:` tags saying what it tests. A requirement is covered by a question when all the words of one of its tags appear in the requirement's name (case, plurals and spellings like Next.js/NextJS are folded; broad tags such as `api` or `security` never match alone). The code picks the questions that cover the most weighted requirements; the judge's relevance order only breaks ties. No model is involved in the matching, so it is repeatable.
4. **Writes questions only for what the bank cannot cover.** New questions are written for requirements that no question in the chosen files covers (heaviest first), one per requirement. If the bank covers everything, or has fewer gaps than `--job-questions`, the unused places go back to the bank: hand-written questions come first. The count is enforced: unacceptable questions are dropped and the model is asked again for the missing number, up to two more times; if it still falls short, nothing is written. Nothing about the candidate is sent for this step, so every candidate for a role gets questions written from the same material.
5. **Orders and checks.** The judge orders all the questions by relevance, most relevant first (the interview asks them in that order), then the coverage report shows, per requirement, whether the interview tests it (`yes`) or nothing does (`GAP`), with metrics: weighted coverage, must-have coverage, questions per covered requirement, and questions that test no listed requirement. It is saved as `questions.coverage.md`.

**Steering.** Run `senjin-review --candidate-dir DIR --requirements-only` to write `requirements.yaml` and stop, then edit it: `skip: true` drops a requirement, `kind` and `weight` change how much it matters (must-have 2, nice-to-have 1). `--focus "FedRAMP" --focus testing` (repeatable) makes the matching requirements must-haves with triple weight for one run; a term that matches nothing is an error that lists the requirements. `--refresh-requirements` lists them again from the job description.

**Measuring.** `senjin-review --candidate-dir DIR --score` checks the current `questions.yaml` against `requirements.yaml` with no model at all: edit a question or add `skills:` to it, run it again, and compare the metrics. Questions without `skills:` count for nothing and are listed.

The resume is redacted before the model sees it (names, emails, phone numbers, links and demographic sentences). Written questions are checked in code: exactly one short spoken question, no markdown, no rubric leakage, no repeats, a requirement from the list, nothing about citizenship, clearance or personal topics.

Output: `<candidate-dir>/questions.yaml`, in the same format as `question_bank/*.yaml` plus `basis` (the judge's reason for a shared question, or the job requirement a written one tests) and `origin` (the bank file, or "generated") for the reviewer. Neither is shown to the interviewer. The folder is under `data/`, which git ignores. Edit the file freely; the command will not overwrite it without `--force`. `senjin-meet run` and `senjin-voice run` print how many questions they loaded, and `senjin-assess` reads rubrics from the same file.

## Phase 3: Post-interview assessment

Score a saved transcript with the local judge (`qwen2.5:32b-instruct-q4_K_M`, about 19 GB; the first call loads it and takes a while):

```bash
uv run senjin-assess data/sessions/<file>.jsonl --candidate-dir data/candidates/jane
```

Outputs `data/reports/<file>.json` (full report with verification stats) and `data/reports/<file>.md` (readable scorecard).

Options:

```bash
uv run senjin-assess data/sessions/<file>.jsonl --judge qwen3:8b-16k --candidate-dir data/candidates/jane     # faster, weaker judge
uv run senjin-assess data/sessions/<file>.jsonl --out-dir /tmp/reports --candidate-dir data/candidates/jane    # choose where reports go
```

Try it without running an interview, using the synthetic transcripts (`strong`, `mediocre`, `weak`, `evasive`):

```bash
uv run senjin-assess examples/transcripts/strong.jsonl --candidate-dir examples/candidate
```

Check judge quality (4 transcripts x 3 runs; about 10 minutes with the 32B judge):

```bash
uv run python -m senjin.eval.assess_eval --runs 3
uv run python -m senjin.eval.assess_eval --runs 1 --judge qwen3:8b-16k
```

It prints one line per run, the valid-JSON rate, and PASS or FAIL for the score ordering (strong > mediocre > weak, evasive below mediocre).

How it stays honest: output is schema-constrained, retried once if invalid, and every cited quote is checked against the candidate's actual words. Claims with invented quotes are dropped and confidence is set to low. If nothing verifies, the command fails instead of writing a report.

Unit tests (fake judge, no Ollama): `uv run pytest tests/test_assess.py`

The hiring bar is applied in code after the judge answers: **Hire** needs technical 7 or more, communication 5 or more and no question below 4 (constants at the top of the bar in `src/senjin/assess/judge.py`). The command prints each question's judge score, and the report says what the judge recommended and whether the bar changed it. The agent introduces itself as Senjin.

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
uv run senjin-chat --exemplars data/exemplars/moves.jsonl --candidate-dir data/candidates/jane
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
uv run senjin-voice simulate --candidate-dir data/candidates/jane
uv run senjin-voice simulate --turns 3 --silence-ms 600 --candidate-dir data/candidates/jane
```

Hold a spoken interview. Use headphones, or the interviewer will hear itself; macOS asks for microphone permission the first time:

```bash
uv run senjin-voice run --candidate-dir data/candidates/jane
uv run senjin-voice run --silence-ms 600 --candidate-dir data/candidates/jane        # end your turn after 0.6 s of silence (default 0.8 s)
uv run senjin-voice run --no-barge-in --candidate-dir data/candidates/jane           # speakers instead of headphones: ignore the mic while Senjin talks
uv run senjin-voice run --model llama3.1:8b-instruct-q8_0 --stt-model ggml-base.en.bin --candidate-dir data/candidates/jane
```

Stop with Ctrl-C. Outputs go to `data/sessions/voice-<timestamp>.jsonl` (transcript) and `.latency.jsonl` (per-turn timings). Summarize a latency file:

```bash
uv run senjin-voice report data/sessions/voice-<timestamp>.latency.jsonl
```

Barge-in: if you speak for about 300 ms while Senjin is talking, playback stops at once and the transcript keeps only the words Senjin actually said (an `interrupt` row records them).

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
uv run senjin-meet run "https://meet.google.com/xxx-xxxx-xxx" --candidate-dir data/candidates/jane
```

Options:

```bash
uv run senjin-meet run URL --candidate-dir data/candidates/jane   # resume.md, job_description.md and the questions.yaml from senjin-review
uv run senjin-meet run URL --wait-candidate 120 --candidate-dir data/candidates/jane      # seconds to wait for a second participant before starting
uv run senjin-meet run URL --admit-timeout 600 --candidate-dir data/candidates/jane       # seconds to wait in the lobby
uv run senjin-meet run URL --silence-ms 600 --candidate-dir data/candidates/jane          # end-of-turn silence
uv run senjin-meet run URL --model llama3.1:8b-instruct-q8_0 --stt-model ggml-base.en.bin --candidate-dir data/candidates/jane
uv run senjin-meet run URL --speak-device "BlackHole 2ch" --hear-device "BlackHole 16ch" --candidate-dir data/candidates/jane
```

The bot says it is an AI and that the call is transcribed in its first sentence. Only join meetings you host or have permission to join. The run ends after the sign-off or when the call ends; the transcript is `data/sessions/meet-<timestamp>.jsonl` and can be scored with `senjin-assess`.

Tests (Chrome required; they are skipped when it is missing): `uv run pytest tests/test_meet.py`

If joining fails: check `src/senjin/meet/selectors.py` first (Meet changes its labels), then run with a visible window (the default) and watch where it stops.

## Phase 7: Hardening

Reliability, built into `senjin-voice run` and `senjin-meet run`:

```bash
uv run senjin-voice run --max-minutes 30 --idle-seconds 20 --candidate-dir data/candidates/jane
uv run senjin-meet run URL --max-minutes 30 --idle-seconds 20 --candidate-dir data/candidates/jane
```

- Time box: at 85% of `--max-minutes` the interviewer skips to the candidate's questions; at 100% it signs off. Default: scaled to the question count (30 minutes for 5 questions).
- Silence watchdog: after `--idle-seconds` of silence (default 25) it says "Are you still there?", then once more, then ends the call politely. Use `--idle-seconds 0` to turn it off.
- Errors: if a reply fails (for example Ollama stops), it apologizes and asks the candidate to repeat; after 3 failures in a row it ends the call and says someone will follow up.
- Meet drops: if the call UI disappears and the interview is not finished, the bot rejoins up to 2 times (waiting up to 2 minutes each time), then gives up.
- Transcripts are written line by line, so a crash keeps everything said so far.

Scorecard (Markdown and JSON in `data/reports/`):

```bash
uv run senjin-assess data/sessions/<file>.jsonl --candidate-dir data/candidates/jane
uv run senjin-meet run URL --assess --candidate-dir data/candidates/jane                  # score automatically when the call ends
```

Personal cues are removed before the judge reads the transcript: the candidate's name (taken from `resume.md`'s first heading, or `--candidate-name "Jane Doe"`), gendered pronouns, and whole sentences about family, age, origin or identity. The reports show the redacted text. To score without redaction (for comparison only):

```bash
uv run senjin-assess data/sessions/<file>.jsonl --no-redact --candidate-dir data/candidates/jane
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
docker run --rm -v "$PWD/data:/app/data" senjin:dev senjin-assess data/sessions/<file>.jsonl --candidate-dir /app/data/candidates/jane
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
