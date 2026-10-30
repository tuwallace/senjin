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
