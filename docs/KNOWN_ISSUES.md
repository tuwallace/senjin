# Known issues

## Minor

### Company and product names are sometimes misheard

Speech recognition (whisper `ggml-small.bin`) can misspell names that are not common English
words. In live Meet calls "Paratus Solutions" came out as "Parada Solutions" and "paratist".

- Effect: a misspelled name in the transcript the judge reads. The meaning of the answer is
  usually still clear, so scores are rarely affected.
- Already done: the resume and job description supply a vocabulary hint to the recognizer
  (`src/senjin/audio/vocab.py`). It helps with some terms (Databricks, Next.js) but did not fix
  this name.
- Possible fix, not done: use a larger model (`ggml-large-v3-turbo`), at the cost of slower
  recognition. Pass it with `--stt-model`.
- Status: accepted for the MVP. Review transcripts by ear for names before relying on them.

### Short answers can be garbled

Very short utterances ("Yes, I'm ready.") are sometimes transcribed oddly ("- I'm second.").
The interview continues normally. Not yet shown to be caused by the vocabulary hint.

### Review: shared-question reasons can be wrong or generic

For shared bank questions, `basis` is the judge's one-line reason. It can be generic, and in an
earlier version it cited an unrelated resume item. Questions written for the job description now
get a deterministic basis (the requirement they test), so this only affects shared ones. Watch it
during testing.

### Review: requirement lists and coverage vary between runs

The requirement list (10 to 13 items) and the coverage mapping change from run to run, the
mapping is lenient (a question that merely mentions compliance may be credited with covering it),
and the preferred section of a job description is sometimes skipped (the report warns when it
is). Treat the coverage report as a guide to gaps, not a measurement.
