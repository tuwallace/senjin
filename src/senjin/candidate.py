"""The candidate folder: always named on the command line, never guessed from the environment."""

import argparse
from pathlib import Path

RESUME = "resume.md"
JOB_DESCRIPTION = "job_description.md"
QUESTIONS = "questions.yaml"  # written by senjin-review


class CandidateError(ValueError):
    pass


def add_candidate_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--candidate-dir",
        type=Path,
        help=f"folder with {RESUME} and {JOB_DESCRIPTION} (required)",
    )


def require_candidate_dir(
    flag: Path | None, files: tuple[str, ...] = (), questions: bool = False
) -> Path:
    """The folder from --candidate-dir, checked. Raises CandidateError with a one-line reason."""
    if flag is None:
        raise CandidateError(
            f"--candidate-dir is required: the folder with {RESUME} and {JOB_DESCRIPTION}"
        )
    if not flag.is_dir():
        raise CandidateError(f"candidate folder not found: {flag}")
    for name in files:
        if not (flag / name).is_file():
            others = sorted(p.name for p in flag.glob("*.md") if p.name not in files)
            hint = f" Found {', '.join(others)}; rename the right one to {name}." if others else ""
            raise CandidateError(f"{flag / name} is missing.{hint}")
        try:
            text = (flag / name).read_text(encoding="utf-8")
        except UnicodeDecodeError:
            raise CandidateError(
                f"{flag / name} is not a text file (a Word or PDF document renamed to .md?): "
                "save it as Markdown or plain text"
            ) from None
        if not text.strip():
            raise CandidateError(f"{flag / name} is empty")
    if questions and not (flag / QUESTIONS).is_file():
        raise CandidateError(
            f"{flag / QUESTIONS} not found: run `senjin-review --candidate-dir {flag}` first"
        )
    return flag


def candidate_from_args(
    flag: Path | None, files: tuple[str, ...] = (), questions: bool = False
) -> Path:
    """For command-line entry points: a bad folder ends the program with one `error:` line."""
    try:
        return require_candidate_dir(flag, files, questions)
    except CandidateError as exc:
        raise SystemExit(f"error: {exc}") from None
