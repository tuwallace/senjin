# ADR 0001: Live interviewer model

Date: 2026-10-29. Status: accepted for Phase 2, revisit after Phase 5 latency measurements.

## Context
The live loop needs reliable tool calling (resume, question bank, notes), spoken-style output, and low latency on an M2 Pro with 32 GB.

## Bake-off
`uv run python -m senjin.eval.bakeoff <models>`: 4 scenarios x 3 runs. A run passes when the model calls the expected tool (or none when it should just speak). First-event time is the time to the first text chunk or tool call; it includes model load for the first run.

| Model | Tool accuracy | Mean first event |
| :-- | :-- | :-- |
| qwen3:8b-16k | 12/12 | 1.37 s |
| llama3.1:8b-instruct-q8_0 | 12/12 | 1.35 s |
| glm-4.7-flash | 10/12 | 1.41 s |

## Decision
Default to `qwen3:8b-16k` (16k context leaves room for long interviews; thinking is disabled with `think=False`). `llama3.1:8b-instruct-q8_0` is an equal fallback. Not chosen: `glm-4.7-flash` (misses tool calls).

## Caveats
The sample is small (12 runs per model), so treat the scores as a smoke check, not a ranking. Re-run when the prompt or tools change.

## Deviation from the blueprint
`get_next_technical_question` takes `topic` and `difficulty` only. The server remembers which questions were asked, instead of the model passing `asked_ids`, because models track ids unreliably. The rubric is never returned to the live model; it is kept for the Phase 3 assessment.
