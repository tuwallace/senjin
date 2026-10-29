# ADR 0005: Hardening, redaction and fairness checks

Date: 2026-10-29. Status: implemented. `v1.0` is not tagged: it needs a live dress rehearsal with a human candidate on a real Meet (see Not verified).

## Reliability
- **Time box:** `InterviewState.max_total_seconds` (default 45 minutes in the CLIs). At 85% the interview jumps to the candidate's questions, at 100% to sign-off, on top of the per-phase budgets.
- **Silence watchdog** in `VoiceLoop`: "Are you still there?", then a second nudge, then a polite goodbye that ends the call. Candidate speech resets it.
- **Error recovery** in `VoiceLoop`: a failed reply (for example Ollama stopped) gets a spoken apology; three failures in a row end the call with a goodbye. Canned lines are recorded in the history through `InterviewSession.say`.
- **Rejoin** (`meet/supervisor.py`): if the call UI disappears before the interview finished, the bot rejoins up to twice.
- **Crash-safe transcripts** were already line-by-line JSONL.
All four are covered by tests with fakes; the rejoin is also exercised against the mock Meet page that hides its call UI.

## Redaction
`assess/redact.py` runs before the judge by default: candidate names (from `resume.md`'s heading or `--candidate-name`) and "my name is X" introductions are masked; gendered pronouns and titles become neutral; whole sentences mentioning family, age, origin, identity, pregnancy or retirement are dropped. Reports show the redacted text, and quote verification runs against the redacted text the judge saw.

It is best effort and blunt: it can drop a sentence that also carried technical content ("my family's restaurant used Redis"), it only knows English patterns, and it cannot catch cues it has no pattern for (school names, employers, locations, writing style). Speech-to-text also hides accent, which is a benefit of the pipeline, but pronunciation errors can lower perceived quality for some speakers; that bias is not measured here.

## Fairness check
`senjin.eval.fairness` rewrites the first candidate turn with 6 variants (neutral control, male with spouse, female parent, 58 years old, name and origin cue, 24 and recent graduate) on the strong and mediocre transcripts and scores each with the 32B judge (temperature 0), with and without redaction.

| Check | Result |
| :-- | :-- |
| Offline (`pytest -m fairness`): redacted variants of all four example transcripts are text-identical | pass |
| Live, no redaction: technical spread across variants | 0 on both transcripts (strong 8, mediocre 4); every decision unchanged |
| Live, redaction on: technical spread across variants | 0 on both transcripts; every decision unchanged |

Honest reading: the live run did not detect bias even without redaction, so it shows no harm but also cannot show that redaction helps. Two transcripts, one run each, an obvious cue and an unambiguous quality gap give it little power. The offline test is the stronger guarantee: with redaction on, the judge receives identical text for these variants. A more sensitive test would use borderline transcripts, many more variants, and repeated runs.

## Containers
`Dockerfile` builds an image for tests, assessment and the MCP server. `docker run --rm senjin:dev` ran the suite in the container: 71 passed, 4 skipped (the Chrome tests). Ollama, audio and the Meet bot stay native on macOS because containers get no Metal GPU or Core Audio. The image reaches Ollama through `host.docker.internal`.

## Not verified
1. A dress rehearsal with a human candidate on a real Meet: Phase 6 itself has not run against real BlackHole devices or Google Meet (ADR 0004).
2. The watchdog, apology and time-box timings were tested with fast fakes, not with real speech and real silence.
3. Report upload to Drive or Slack was left out on purpose; reports stay local until a person shares them.
4. Assessment remains advisory. The judge was only checked on four synthetic transcripts; it should not make or gate hiring decisions on its own.
