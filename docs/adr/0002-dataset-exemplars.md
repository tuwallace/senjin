# ADR 0002: Anthropic Interviewer dataset as style exemplars

Date: 2026-10-29. Status: implemented, exemplars off by default.

## What the dataset is
1,250 interviews (workforce 1,000, creatives 125, scientists 125), turns labelled `Assistant:` (opener), `AI:` and `User:`. 14,495 AI/User exchanges. They are research interviews about AI at work, with no technical content and no scores.

## What we built
- `eval/dataset.py`: fetch, parse into turns, extract interviewer moves with a heuristic type (opener, probe, clarify, reflect, transition, wrap_up). Counts: probe 11,105, reflect 1,684, clarify 362, transition 77, opener 1,250, wrap_up 1,250.
- `interview/exemplars.py`: craft filter (one open question, 8 to 45 words, responding to a real answer), 40 exemplars each of probe, clarify and reflect, embedded with `nomic-embed-text` on the participant's answer. At each turn in the background and technical phases, the 2 nearest exemplars are added to the system prompt as style examples.
- `eval/style.py`: deterministic style metrics. `eval/sim.py`: simulated candidates (strong, average, weak) for unattended interviews.

The roadmap proposed an LLM judge for style. I used heuristic metrics instead so before/after numbers are reproducible and free. The cost: they only see surface features.

## Result
Dataset baseline versus 4 quick simulated interviews per arm (2 personas x 2 runs, 20 mid-interview turns per arm, `qwen3:8b-16k`):

| | single question | open-ended | ties to answer | speakable | words |
| :-- | :-- | :-- | :-- | :-- | :-- |
| Dataset interviewer | 29% | 60% | 78% | 100% | 54 |
| Ours, no exemplars | 95% | 90% | 100% | 100% | 30 |
| Ours, with exemplars | 75% | 80% | 100% | 100% | 33 |

Findings:
1. Our interviewer already beats the dataset on these metrics, because the dataset's interviewer often asks several questions per turn. The dataset is not a gold standard for "single open question" style.
2. Exemplars did not help and slightly hurt the single-question rate (95% to 75%). The sample is 20 turns per arm, so this is suggestive, not conclusive. Reading the transcripts, exemplar runs tended to add a second question ("What were the main challenges? What mechanisms did you use?").
3. Both arms open turns with praise ("That's impressive!", "That's a solid approach"), which our system prompt forbids. The metrics do not measure this.

## Decision
Keep the exemplar machinery (`--exemplars`) but leave it off by default. Do not claim a gain. Next steps if we revisit: add a praise/acknowledgement metric, filter exemplars to those with no praise, and try a stricter "one question mark" instruction before adding retrieved examples.

## Data handling
Raw CSVs and the embedded exemplars live under `data/` (git-ignored) and are never committed. README carries the attribution.
