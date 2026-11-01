# 0007: Per-candidate question banks

Status: accepted

## Context
The shared question bank is generic. A live run against a full-stack Next.js role asked about
rate limiters, and nothing in the technical part depended on the job or the candidate. Questions
generated from the job description in one request were generic too, weaker than hand-written ones:
they copied the job description's bullets and repeated one another.

## Decision
The interview is sized by `SENJIN_MAX_QUESTIONS` and `SENJIN_RESUME_QUESTIONS` (for example 18:
6 background questions about the resume, asked by the model from `resume.md`, and 12 technical).
`senjin-review --candidate-dir DIR` writes the technical questions to `DIR/questions.yaml`:

1. The job's requirements are listed (must-have and nice-to-have; no eligibility items, years of
   experience, degrees or "prior experience" items) and saved as `requirements.yaml`, which a
   person can edit (skip, kind, weight) and which later runs read back. `--focus` steers one run.
2. The judge model picks the `question_bank/*.yaml` files that suit the job description.
3. The code picks the best shared questions by skills coverage: bank questions carry `skills`
   tags, matched to requirement names with no model; the judge's order only breaks ties.
4. New questions are written only for requirements no chosen-file question can cover, one
   requirement per question; places not needed go back to the bank, because hand-written questions
   are better. The count is enforced in code with up to two top-up requests, and the command
   fails without writing if it cannot be reached. These questions use the job description alone,
   so every candidate for a role gets the same material.
5. The judge orders all of them by relevance and the report shows which requirements the
   interview tests, with metrics (`--score` recomputes them without a model after hand edits).

Interviews, `senjin-assess` and the MCP server read only that file. There is no fallback to the
whole shared bank and no environment variable for the bank or the candidate folder:
`SENJIN_CANDIDATE_DIR`, `SENJIN_QUESTION_SET` and `SENJIN_QUESTION_BANK` were removed. Every
command that reads a candidate requires `--candidate-dir` and stops with an `error:` line if it
is missing or the folder does not exist. The evaluation tools (`senjin-sim`, bake-off, fairness)
use the built-in sample in `examples/candidate` and the whole shared bank.

Time boxes scale with the question count (about 2 minutes per interviewer turn, 10 extra for the
greeting, the candidate's questions and the goodbye) unless `SENJIN_MAX_MINUTES` is set (30 minutes for the default 5 questions, 70 for 18).

- The resume is redacted before the model sees it, and it only influences which shared questions
  are picked and their order. Written questions are checked in code (one short spoken question,
  no markdown, no rubric leakage, no repeats, a listed requirement, nothing about eligibility
  or personal traits).
- The file is for a person to review and edit. It is never overwritten without `--force` and lives
  under `data/`, which is not committed.

## Consequences
- Review is a required step before an interview, at the cost of several minutes per candidate.
- Quality depends on a local 32B model. Requirement lists vary from run to run, and the coverage
  mapping is lenient; both are tracked in `docs/KNOWN_ISSUES.md`.
- The resume shapes the interview through the background phase and the choice of shared questions,
  not through candidate-specific technical questions (an earlier version wrote those, and was
  dropped in favour of comparable questions across candidates).
