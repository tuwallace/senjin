# ADR 0003: Voice loop design and latency

Date: 2026-10-29. Status: implemented; measured with synthetic speech, not yet with a live microphone.

## Choices
- **VAD:** Silero v5 ONNX run directly with onnxruntime (no PyTorch). A separate `TurnDetector` turns frame probabilities into start/end events: 96 ms to start, 800 ms of silence to end (the blueprint says 1.5 s; that alone would blow the latency budget), 320 ms pre-roll, bursts under 350 ms discarded.
- **STT:** whisper.cpp `whisper-server` (Metal, model stays loaded) instead of faster-whisper, which has no Apple GPU backend. On a 4.6 s spoken sentence: base.en 0.09 s, small 0.23 s, large-v3-turbo-q5 0.73 s. Default is `ggml-small.bin`: base.en misheard "Northwind Pay" as "Northwind Bay".
- **TTS:** Piper (`en_US-lessac-medium`), about 0.1 s for the first sentence. macOS `say` is a fallback.
- **Streaming:** LLM tokens are cut into sentences; each sentence is synthesized while the previous one plays.
- **Barge-in:** the VAD keeps running during playback. 300 ms of candidate speech stops the player, cancels the reply, and rewrites the interviewer's history to the sentences that had begun playing. Assumes headphones; `--no-barge-in` ignores the mic during playback for speakers.

## Latency work (synthetic 6-turn interview, `qwen3:8b-16k`)
Budget: end of candidate speech to first audio under 2.5 s on average. The 800 ms endpoint wait comes on top.

| Step | Mean to first audio |
| :-- | :-- |
| First run: model decides tool calls, transition note in system prompt | 4.5 s |
| + orchestrator prefetches resume/question on phase entry, hides `record_note` | 3.6 s |
| + stable system prompt within a phase (KV-cache reuse), `keep_alive=30m` | 2.45 s |

Per turn after the last change: 0.9 to 1.3 s inside a phase, 2.1 to 3.1 s on the first turn of a phase, and 5.9 s on the entry to the technical phase (full prompt prefill plus the question lookup).

Why: Ollama reuses its KV cache only for an identical prompt prefix. The old "Transition: ..." note made the system message differ on the first turn of each phase, forcing a full prefill on the second turn too. The mean is at the budget but the worst case is not; phase transitions are the slow turns.

## Trade-offs
- Prefetch picks the next unasked question in bank order (no topic or difficulty choice) on entering the technical phase; the model can still call the tool for later topics.
- `record_note` is hidden from the live model (it cost a full LLM round trip before speech). Notes can be derived from the transcript at assessment time.
- The transition note was dropped; the phase instruction alone changes the topic. Observed side effect: occasional abrupt phrasing such as "Let's try a different approach."
- The assessment reads `assistant` rows, which hold the full generated text even when a barge-in cut it short.

## Not verified
Live microphone and speaker paths (`mic_frames`, `Player`) have not been exercised on hardware, and echo is unsolved without headphones. STT accuracy was tested on clean Piper speech only; "Postgres" came back as "post-curs"/"Postcues" and "idempotency" as "ID potency".

## Next options for latency
Insert a short pre-synthesized acknowledgement ("Mm-hm.") when the model is slow; put the phase instruction in the last user message so the prefix stays stable across phases; try a smaller model for the first turn of a phase.
