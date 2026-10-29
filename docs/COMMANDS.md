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
| 5 Voice loop | planned | `v0.5` |
| 6 Google Meet bot | planned | `v0.6` |
| 7 Hardening | planned | `v1.0` |

## Prerequisites (once)

```bash
ollama serve            # skip if the Ollama app is already running
ollama pull qwen3:8b-16k
uv sync                 # creates .venv with Python 3.12 and installs dependencies
```

Check Ollama is up: `curl -s http://localhost:11434/api/tags | head -c 200`

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

## Phase 4: Anthropic dataset (planned)

```bash
uv run senjin-data fetch                # download Anthropic/AnthropicInterviewer into data/hf_cache/
uv run senjin-data moves                # extract and cluster interviewer moves
uv run senjin-eval style                # style metric: baseline vs. current interviewer
uv run senjin-sim --persona strong      # run an unattended interview with a simulated candidate
```

## Phase 5: Voice loop on the laptop (planned)

```bash
uv run senjin-voice --stt whisper.cpp --tts piper    # spoken interview with the mic and speakers (use headphones)
uv run senjin-voice --latency-report                 # per-stage timings
```

## Phase 6: Google Meet bot (planned)

```bash
uv run senjin-meet login                             # one-time sign in of the bot Google account, saves a browser profile
uv run senjin-meet join "https://meet.google.com/xxx-xxxx-xxx"
```

Audio devices (BlackHole) must be installed first; see Phase 0.

## Phase 7: Hardening (planned)

```bash
uv run senjin-report data/sessions/<file>.jsonl      # final Markdown and JSON scorecard
uv run pytest -m fairness                            # paired-transcript bias checks
```

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
