"""Live check of the judge on the synthetic transcripts in examples/transcripts.

Usage: uv run python -m senjin.eval.assess_eval [--judge MODEL] [--runs 3]
Pass criteria: every run returns valid JSON, and mean technical scores order
strong > mediocre > weak, with the evasive transcript scoring below mediocre.
"""

import argparse
import asyncio
from pathlib import Path
from statistics import mean

from senjin.assess.judge import DEFAULT_JUDGE, OllamaJudge, asked_question_ids, assess, load_rows
from senjin.mcp_server.store import load_questions

ROOT = Path(__file__).resolve().parents[3]
LEVELS = ["strong", "mediocre", "weak", "evasive"]


async def main(judge_model: str, runs: int) -> int:
    bank = load_questions(ROOT / "question_bank")
    judge = OllamaJudge(judge_model)
    scores: dict[str, list[int]] = {}
    ok = total = 0
    for level in LEVELS:
        rows = load_rows(ROOT / f"examples/transcripts/{level}.jsonl")
        ids = asked_question_ids(rows)
        rubrics = {q.id: q.rubric for q in bank if q.id in ids}
        for _ in range(runs):
            total += 1
            try:
                report = await assess(rows, judge, rubrics)
            except Exception as exc:
                print(f"{level}: FAILED {exc}")
                continue
            ok += 1
            a = report.assessment
            scores.setdefault(level, []).append(a.technical_score)
            print(
                f"{level:9} tech {a.technical_score} comm {a.communication_score} "
                f"{a.hire_recommendation:8} conf {a.confidence} "
                f"dropped {report.verification.dropped_findings}"
            )
    means = {k: mean(v) for k, v in scores.items()}
    print(f"\nvalid {ok}/{total}; mean technical: {means}")
    ordered = (
        all(k in means for k in LEVELS)
        and means["strong"] > means["mediocre"] > means["weak"]
        and means["evasive"] < means["mediocre"]
    )
    print("ordering:", "PASS" if ordered else "FAIL")
    return 0 if ok == total and ordered else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge", default=DEFAULT_JUDGE)
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()
    raise SystemExit(asyncio.run(main(args.judge, args.runs)))
