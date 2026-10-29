"""`uv run senjin-assess data/sessions/x.jsonl [--judge MODEL] [--out-dir data/reports]`."""

import argparse
import asyncio
import os
from pathlib import Path

from senjin.assess.judge import (
    DEFAULT_JUDGE,
    OllamaJudge,
    asked_question_ids,
    assess,
    load_rows,
    to_markdown,
)
from senjin.assess.redact import names_from_dir
from senjin.mcp_server.store import load_questions

ROOT = Path(__file__).resolve().parents[3]


async def run(
    path: Path,
    judge_model: str,
    out_dir: Path,
    bank: Path,
    names: list[str] | None = None,
    redact: bool = True,
) -> Path:
    rows = load_rows(path)
    ids = asked_question_ids(rows)
    rubrics = {q.id: q.rubric for q in load_questions(bank) if q.id in ids}
    if names is None:
        candidate_dir = Path(os.environ.get("SENJIN_CANDIDATE_DIR", ROOT / "examples/candidate"))
        names = names_from_dir(candidate_dir)
    report = await assess(rows, OllamaJudge(judge_model), rubrics, names, redact)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{path.stem}.json").write_text(report.model_dump_json(indent=2))
    (out_dir / f"{path.stem}.md").write_text(to_markdown(report))
    a = report.assessment
    print(
        f"{a.hire_recommendation}  tech {a.technical_score}  comm {a.communication_score}  "
        f"confidence {a.confidence}  ({'personal cues redacted' if redact else 'NOT redacted'})"
    )
    print(f"Wrote {out_dir / (path.stem + '.md')}")
    return out_dir / f"{path.stem}.md"


def main() -> None:
    parser = argparse.ArgumentParser(description="Score a Senjin interview transcript")
    parser.add_argument("transcript", type=Path)
    parser.add_argument("--judge", default=DEFAULT_JUDGE)
    parser.add_argument("--out-dir", type=Path, default=Path("data/reports"))
    parser.add_argument("--question-bank", type=Path, default=ROOT / "question_bank")
    parser.add_argument(
        "--candidate-name", action="append", help="name to mask (default: from the resume)"
    )
    parser.add_argument(
        "--no-redact", action="store_true", help="show the judge names and personal details"
    )
    args = parser.parse_args()
    asyncio.run(
        run(
            args.transcript,
            args.judge,
            args.out_dir,
            args.question_bank,
            args.candidate_name,
            not args.no_redact,
        )
    )


if __name__ == "__main__":
    main()
