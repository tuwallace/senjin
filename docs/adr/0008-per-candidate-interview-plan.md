# 0008: A per-candidate interview plan

Status: accepted

## Context
The interview's shape (how many background and technical questions, how many follow-ups, how
long) came from global `SENJIN_*` settings, so a junior and a staff role got the same interview.
A fixed time box also cut off long interviews.

## Decision
`senjin-review` picks an interview profile for the job and saves it as
`<candidate-dir>/interview.yaml`. Profiles are small files in `profiles/`: question counts,
follow-ups, minutes per interviewer turn and extra minutes for the greeting, the candidate's
questions and the goodbye. The time box is computed from them (or pinned with `minutes`).

- The judge reads the job description only and names a profile; an unknown answer falls back to
  `default`, which reproduces the earlier 5-question, 30-minute interview. The choice is saved, so
  it does not change between runs, and a person can edit it or switch with `--profile`.
- Because the job description is the only input, every candidate for a job gets the same shape.
  The resume still shapes the background questions and the choice of shared questions.
- Precedence: flags, then `interview.yaml`, then `SENJIN_*`. Those settings now cover tools that
  read no candidate. Audio timing stays global because it depends on the machine, not the job.
- Interview commands stop with an error when `interview.yaml` is missing or when `questions.yaml`
  has fewer questions than the plan asks for, rather than falling back silently.
- `senjin-review` sizes the question set from the plan; `--score` uses the plan's technical count
  as the window an interview reaches.

## Consequences
- Review is required before an interview and must be re-run when the job description changes.
- Profile choice depends on a local model and can be wrong; the saved file, `--profile` and the
  printed plan are the safeguards. Minutes per turn (2.0) is an estimate to calibrate from real
  interviews.
