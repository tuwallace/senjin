# Session logs

Hand-maintained notes on *why*: decisions, dead ends and what is next. The code and commands are in the graph and in [[COMMANDS]]; plans are in [[ROADMAP]] and [[ENHANCEMENTS]]; open problems are in [[KNOWN_ISSUES]]. No interview content belongs here.

## 2026-11-03/04: live-call fixes, voice, video

**Done**
- Interview behaviour: re-ask a lost part of a question, a punctuation-tolerant quote check, a late filler so slow turns are not silent, a fixed greeting, and coverage labels that say "cut" rather than "GAP" for questions dropped for time.
- Added the QA/program-lead profile and a 20-question QA bank.
- Voice: Senjin speaks 20% slower (`SENJIN_VOICE_PACE`); Piper lessac was picked after an eight-voice listening test.
- Published the YouTube video (see README). Tooling is in `docs/video/`; the video itself is not in git.

**Decisions and why**
- The greeting is fixed text, not model output. A prompt change made the model drop the AI notice and the "ready to begin?" question, which is a consent problem, so the model no longer writes it ([[ENHANCEMENTS]] has the voice notes).
- The decision model (`nimble`) stays **off by default**. It was more accurate on 70 labelled turns (43 right against 40 for the rules on "is the Q&A over?") but added about 600 ms per call, and no live call showed it firing. See [[ENHANCEMENTS]] section 1.
- The report stays local for the MVP; upload to Drive or Slack is an enhancement, not MVP.
- The time box stays a ceiling, not a target.
- The video narration uses Kokoro (`am_michael`) rather than a macOS voice because a macOS voice sounded robotic. The opening clip is labelled a re-creation with synthetic voices: it was built from a real transcript, not recorded live.

**Dead ends**
- `screencapture` produced an empty file when stopped by a signal; a fixed `-V` duration saves correctly.
- Joining video segments with the ffmpeg concat demuxer left silent gaps; the concat filter fixed it.
- graphify's semantic pass over the docs with `qwen2.5-coder:7b` returned nothing; the graph is code-only unless a larger model works (see below).

**Next**
- Tag `v1.0` or `v0.9` (undecided).
- Legal notice requirements for automated interviews are the owner's to check.
- Optional: profile the one slow Q&A turn seen in a live call.
- Enhancements still open: scheduling and hosting the Meet, more decision-model uses, more voices, report upload.
