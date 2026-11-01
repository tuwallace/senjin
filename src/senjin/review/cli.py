"""`uv run senjin-review --candidate-dir DIR`: assemble the technical questions for one candidate.

Sized from SENJIN_MAX_QUESTIONS: the background phase asks about the resume, and the technical
questions in <candidate-dir>/questions.yaml are split between the shared bank and new questions
written for the job description's requirements that the bank leaves uncovered.

1. The judge picks the shared question_bank/ files that suit the job description.
2. The job's requirements are read from requirements.yaml, or listed by the judge and saved there
   (edit the file, or use --focus, to steer).
3. The judge orders the shared questions; the code then picks the ones whose skills tags cover
   the most weighted requirements, using the judge's order to break ties.
4. New questions are written for the requirements still uncovered, most heavily weighted first.
5. The judge orders all of them; the coverage report shows what the interview tests.

`--score` checks an existing questions.yaml against requirements.yaml with no model at all.
"""

import argparse
import asyncio
import math
from collections.abc import Callable
from pathlib import Path

from senjin.assess.judge import DEFAULT_JUDGE, JudgeLLM, OllamaJudge
from senjin.assess.redact import candidate_names
from senjin.candidate import (
    JOB_DESCRIPTION,
    QUESTIONS,
    RESUME,
    CandidateError,
    add_candidate_arg,
    require_candidate_dir,
)
from senjin.config import load_env_file
from senjin.mcp_server.store import ROOT, load_questions
from senjin.plan import PLAN, PlanError, load_plan, load_profiles
from senjin.review.coverage import (
    COVERED_ASKED,
    build_report,
    coverage_mapping,
    extract_requirements,
    preferred_note,
    uncovered,
)
from senjin.review.generate import ReviewError, generate_questions, render_yaml, scrub_resume
from senjin.review.plan import apply_focus, load_requirements, save_requirements
from senjin.review.profile import describe_profiles, ensure_plan
from senjin.review.schema import Requirement
from senjin.review.select import bank_files, choose_files, from_dicts, rank_pool, to_dicts
from senjin.review.skills import covered_by, pick_by_coverage

REQUIREMENTS = "requirements.yaml"


def split_technical(technical: int) -> tuple[int, int]:
    """(from the shared bank, written for the job description): an even split, bank first."""
    bank = math.ceil(technical / 2)
    return bank, technical - bank


async def get_requirements(
    judge: JudgeLLM,
    candidate_dir: Path,
    job_description: str,
    refresh: bool,
    show: Callable[[str], None],
    save: bool = True,
) -> tuple[list[Requirement], dict[str, float]]:
    path = candidate_dir / REQUIREMENTS
    if path.is_file() and not refresh:
        found, weights = load_requirements(path)
        show(f"   {len(found)} requirements from {path} (edit it to steer; --refresh-requirements)")
        return found, weights
    found = await extract_requirements(judge, job_description)
    if save:
        save_requirements(path, found, {})
        show(f"   {len(found)} requirements, saved to {path}")
    else:
        show(f"   {len(found)} requirements")
    return found, {}


async def review(
    candidate_dir: Path,
    judge: JudgeLLM,
    bank_questions: int,
    job_questions: int,
    bank_dir: Path = ROOT / "question_bank",
    out: Path | None = None,
    force: bool = False,
    dry_run: bool = False,
    focus: list[str] | None = None,
    refresh_requirements: bool = False,
    requirements_only: bool = False,
    show: Callable[[str], None] = print,
) -> Path | None:
    """Assemble, show and (unless dry_run) write the questions. Returns the file written."""
    candidate_dir = require_candidate_dir(candidate_dir, files=(RESUME, JOB_DESCRIPTION))
    target = out or candidate_dir / QUESTIONS
    if target.exists() and not (force or dry_run or requirements_only):
        raise ReviewError(f"{target} already exists; pass --force to overwrite it (or edit it)")
    if bank_questions < 1 or job_questions < 0:
        raise ReviewError("--bank-questions must be at least 1 and --job-questions at least 0")
    total = bank_questions + job_questions
    resume = (candidate_dir / RESUME).read_text(encoding="utf-8")
    job_description = (candidate_dir / JOB_DESCRIPTION).read_text(encoding="utf-8")
    redacted = scrub_resume(resume, candidate_names(resume))

    show(f"Reading {candidate_dir / RESUME} and {candidate_dir / JOB_DESCRIPTION}")
    show(
        f"Model: {judge.model}. {bank_questions} bank + {job_questions} job-description questions."
    )

    show("\n1. Listing the job's requirements...")
    requirements, weights = await get_requirements(
        judge, candidate_dir, job_description, refresh_requirements, show, save=not dry_run
    )
    if focus:
        requirements, weights, matched = apply_focus(requirements, weights, focus)
        for term, names in matched.items():
            show(f"   focus {term!r}: {'; '.join(names)}")
    if requirements_only:
        show(f"\nEdit {candidate_dir / REQUIREMENTS}, then run again without --requirements-only.")
        return None

    show("\n2. Choosing shared question files for this job...")
    files = bank_files(bank_dir)
    chosen, notes = await choose_files(judge, job_description, files)
    for note in notes:
        show(f"   {note}")
    show(f"   using: {', '.join(chosen)}")
    shared = []
    for name in chosen:
        for question in files[name]:
            question.origin = f"question_bank/{name}.yaml"
            shared.append(question)

    # Hand-written questions come first: new ones are written only for requirements that no
    # question in the chosen files can cover, and unused places go back to the bank.
    pool_map, pool_notes = await coverage_mapping(requirements, shared, judge)
    bank_gaps = uncovered(requirements, pool_map, weights)
    writable = min(job_questions, len(bank_gaps)) if bank_gaps else 0
    slots_bank = min(total - writable, len(shared))
    slots_job = total - slots_bank
    show(
        f"\n3. The bank covers {len(requirements) - len(bank_gaps)} of {len(requirements)} "
        f"requirements; picking {slots_bank} of {len(shared)} by skills coverage..."
    )
    for note in pool_notes:
        show(f"   {note}")
    ordered, rank_notes = await rank_pool(judge, job_description, redacted, shared, len(shared))
    for note in rank_notes:
        show(f"   {note}")
    picked = pick_by_coverage(ordered, requirements, weights, slots_bank)
    for question in picked:
        covers = [r.name for r in covered_by(question, requirements)]
        if covers:
            question.basis = f"Covers: {'; '.join(covers)}. {question.basis}".strip()
    picked_map, map_notes = await coverage_mapping(requirements, picked, judge)
    before = build_report(requirements, picked_map, picked, total, weights)
    gaps = uncovered(requirements, picked_map, weights)
    show(f"   those cover {len(requirements) - len(gaps)} of {len(requirements)} requirements")
    for note in map_notes:
        show(f"   {note}")

    new: list[dict] = []
    if slots_job:
        show(f"\n4. Writing {slots_job} questions for requirements the bank cannot cover...")
        targets = bank_gaps or [r for r in requirements if r.kind == "must_have"] or requirements
        if not bank_gaps:
            show("   the bank covers everything, so these go deeper on the must-haves")
        written = await generate_questions(judge, job_description, targets, slots_job)
        new = written.questions
        for line in written.dropped:
            show(f"   rejected: {line}")

    show(f"\n5. Ordering all {total} questions by relevance and checking coverage...")
    pool = [*picked, *from_dicts(new, "generated")]
    final, order_notes = await rank_pool(
        judge, job_description, redacted, pool, total, update_basis=False
    )
    for note in order_notes:
        show(f"   {note}")
    mapping, final_notes = await coverage_mapping(requirements, final, judge)
    report = build_report(requirements, mapping, final, total, weights)
    report.notes += final_notes
    if note := preferred_note(job_description, requirements):
        report.notes.append(note)
    report.notes.append(
        f"before the new questions, the {bank_questions} shared ones covered "
        f"{before.count(COVERED_ASKED)} of {before.total()} must-have requirements"
    )

    for position, item in enumerate(final, 1):
        show(f"\n{position:>2}. {item.id}  [{item.difficulty}]  from {item.origin}")
        show(f"    {item.prompt}")
        show(f"    why: {item.basis}")
    show("\n" + report.to_text())

    if dry_run:
        show("\nDry run: nothing was written.")
        return None
    source = f"{candidate_dir / RESUME} and {candidate_dir / JOB_DESCRIPTION}"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_yaml(to_dicts(final), judge.model, source), encoding="utf-8")
    sidecar = target.with_suffix(".coverage.md")
    sidecar.write_text(report.to_markdown(), encoding="utf-8")
    show(f"\nWrote {target} and {sidecar}.")
    show("Review and edit the questions; interviews for this candidate will use them in order.")
    return target


async def score(
    candidate_dir: Path,
    window: int | None = None,
    focus: list[str] | None = None,
    show: Callable[[str], None] = print,
):
    """Check questions.yaml against requirements.yaml. No model; edit, run, repeat."""
    candidate_dir = require_candidate_dir(candidate_dir, questions=True)
    path = candidate_dir / REQUIREMENTS
    if not path.is_file():
        raise ReviewError(
            f"{path} not found: run `senjin-review --candidate-dir {candidate_dir}` "
            "(or with --requirements-only) first"
        )
    requirements, weights = load_requirements(path)
    if focus:
        requirements, weights, _ = apply_focus(requirements, weights, focus)
    questions = load_questions(candidate_dir, [Path(QUESTIONS).stem])
    mapping, notes = await coverage_mapping(requirements, questions, judge=None)
    if window is None and (candidate_dir / PLAN).is_file():
        window = load_plan(candidate_dir / PLAN).technical_questions  # what an interview asks
    report = build_report(requirements, mapping, questions, window or len(questions), weights)
    report.notes += notes
    show(report.to_text())
    return report


async def run_command(
    args: argparse.Namespace, candidate_dir: Path, judge: JudgeLLM | None = None
) -> None:
    judge = judge or OllamaJudge(args.model)
    profiles = load_profiles()
    job_description = (candidate_dir / JOB_DESCRIPTION).read_text(encoding="utf-8")
    print("Interview plan:")
    plan = await ensure_plan(
        candidate_dir,
        judge,
        job_description,
        profiles,
        profile=args.profile,
        refresh=args.refresh_plan,
        save=not args.dry_run,
    )
    if args.plan_only:
        print(f"\nEdit {candidate_dir / PLAN} if you like, then run senjin-review again.")
        return
    bank_n, job_n = split_technical(plan.technical_questions)
    await review(
        candidate_dir,
        judge,
        args.bank_questions or bank_n,
        args.job_questions if args.job_questions is not None else job_n,
        out=args.out,
        force=args.force,
        dry_run=args.dry_run,
        focus=args.focus,
        refresh_requirements=args.refresh_requirements,
        requirements_only=args.requirements_only,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Assemble the interview plan and technical questions for one candidate from "
        "the shared bank and the job description's requirements"
    )
    add_candidate_arg(parser)
    parser.add_argument("--out", type=Path, help=f"default: <candidate-dir>/{QUESTIONS}")
    parser.add_argument("--profile", help="use this interview profile instead of choosing one")
    parser.add_argument(
        "--list-profiles", action="store_true", help="show the interview profiles and stop"
    )
    parser.add_argument(
        "--refresh-plan", action="store_true", help=f"choose the profile again, replacing {PLAN}"
    )
    parser.add_argument(
        "--plan-only", action="store_true", help=f"write {PLAN} and stop, so you can edit it"
    )
    parser.add_argument(
        "--bank-questions", type=int, help="from the shared bank (default: half the technical)"
    )
    parser.add_argument(
        "--job-questions", type=int, help="written for the job description (default: the rest)"
    )
    parser.add_argument(
        "--focus",
        action="append",
        help="a requirement to stress (repeatable), for example --focus FedRAMP --focus testing",
    )
    parser.add_argument(
        "--requirements-only",
        action="store_true",
        help=f"list the requirements into {REQUIREMENTS} and stop, so you can edit it",
    )
    parser.add_argument(
        "--refresh-requirements",
        action="store_true",
        help=f"list the requirements again instead of reading {REQUIREMENTS}",
    )
    parser.add_argument(
        "--score",
        action="store_true",
        help=f"check {QUESTIONS} against {REQUIREMENTS} with no model, and stop",
    )
    parser.add_argument("--model", default=DEFAULT_JUDGE, help="local model that picks and writes")
    parser.add_argument("--dry-run", action="store_true", help="show the questions, write nothing")
    parser.add_argument("--force", action="store_true", help="overwrite an existing file")
    args = parser.parse_args()
    load_env_file()
    try:
        if args.list_profiles:
            print(describe_profiles(load_profiles()))
            return
        if args.score:
            candidate_dir = require_candidate_dir(args.candidate_dir, questions=True)
            asyncio.run(score(candidate_dir, focus=args.focus))
            return
        candidate_dir = require_candidate_dir(args.candidate_dir, files=(RESUME, JOB_DESCRIPTION))
        asyncio.run(run_command(args, candidate_dir))
    except (ReviewError, CandidateError, PlanError) as exc:
        raise SystemExit(f"error: {exc}") from None


if __name__ == "__main__":
    main()
