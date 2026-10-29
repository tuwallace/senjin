# ADR 0001: Live interviewer model

Date: 2026-10-29. Status: accepted for Phase 2, revisit after Phase 5 latency measurements.

## Context
The live loop needs reliable tool calling (resume, question bank, notes), spoken-style output, and low latency on an M2 Pro with 32 GB.

## Bake-off
`uv run python -m senjin.eval.bakeoff <models>`: 4 scenarios x 3 runs. A run passes when the model's first action is an acceptable tool call (or no tool when it should just speak). In the candidate Q&A scenario, looking up the job description before answering also counts as acceptable. First-event time is the time to the first text chunk or tool call; it includes model load for the first run.

| Model | Run 1 (strict) | Run 2 (Q&A lookup allowed) | Mean first event |
| :-- | :-- | :-- | :-- |
| qwen3:8b-16k | 12/12 | 12/12 | 1.4 to 1.6 s |
| llama3.1:8b-instruct-q8_0 | 12/12 | 11/12 | 1.1 to 1.4 s |
| glm-4.7-flash | 10/12 | 12/12 | 1.3 to 1.4 s |

glm-4.7-flash's two run-1 misses were get_job_description calls during candidate Q&A, which is reasonable behavior; the scenario now accepts it. llama3.1 missed one run the second time, so results vary by a run or two between sessions.

## Decision
Default to `qwen3:8b-16k` (16k context leaves room for long interviews; thinking is disabled with `think=False`); it was the only model with no misses in both runs. `llama3.1:8b-instruct-q8_0` and `glm-4.7-flash` are viable fallbacks; glm is a 19 GB model, so it uses much more memory than the 8B models.

## Caveats
The sample is small (12 runs per model) and noisy, so treat the scores as a smoke check, not a ranking. Re-run when the prompt or tools change.

## Deviation from the blueprint
`get_next_technical_question` takes `topic` and `difficulty` only. The server remembers which questions were asked, instead of the model passing `asked_ids`, because models track ids unreliably. The rubric is never returned to the live model; it is kept for the Phase 3 assessment.
