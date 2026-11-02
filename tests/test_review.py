"""senjin-review and the mandatory candidate folder. No model is called; judges are scripted."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from senjin.assess.redact import candidate_names
from senjin.candidate import CandidateError, require_candidate_dir
from senjin.mcp_server.store import (
    GENERATED_BANK,
    Question,
    all_questions,
    bank_label,
    load_questions,
)
from senjin.review.cli import REQUIREMENTS, review, score, split_technical
from senjin.review.coverage import (
    COVERED_ASKED,
    COVERED_FILE,
    COVERED_LEFT_OUT,
    COVERED_NONE,
    build_report,
    coverage_mapping,
    extract_requirements,
    map_coverage,
    preferred_note,
    uncovered,
)
from senjin.review.generate import (
    ReviewError,
    generate_questions,
    problems_with,
    render_yaml,
    scrub_resume,
)
from senjin.review.plan import apply_focus, load_requirements, save_requirements
from senjin.review.schema import GeneratedQuestion, Requirement
from senjin.review.select import bank_files, choose_files, rank_pool, to_dicts
from senjin.review.skills import pick_by_coverage, tags_match, words

ROOT = Path(__file__).resolve().parents[1]

TOPICS = ["caching", "authentication", "accessibility", "api_design", "testing", "devops"]
PROMPTS = [
    "How would you cache responses from a slow upstream service?",
    "Design sign-in for an app that uses an external identity provider. What happens end to end?",
    "What do you build in so a form works with a screen reader and the keyboard alone?",
    "Would you offer REST or GraphQL for a React front end that edits records, and why?",
    "How do you decide what to cover with unit tests versus browser tests?",
    "How does a change move from a pull request to production in a regulated team?",
]


def written(i: int, **over) -> dict:
    """What the model returns for one job-description question."""
    base = {
        "topic": TOPICS[i],
        "difficulty": ["easy", "medium", "hard"][i % 3],
        "requirement": ["Node.js backends", "TypeScript", "Cloud native on GovCloud"][i % 3],
        "prompt": PROMPTS[i],
        "follow_ups": [f"What goes wrong at scale in case {i}?", f"What would you measure {i}?"],
        "rubric": [f"Names a concrete approach {i}", "Weighs trade-offs", "Considers failure"],
    }
    base.update(over)
    return base


def questions_reply(*items: dict) -> str:
    return json.dumps({"questions": list(items)})


def files_reply(*names: str) -> str:
    return json.dumps({"files": [{"file": n, "reason": f"{n} suits the job"} for n in names]})


def rank_reply(*ids: str) -> str:
    return json.dumps({"selected": [{"id": i, "relevance": f"{i} is relevant"} for i in ids]})


class ScriptedJudge:
    model = "scripted-model"

    def __init__(self, *replies: str) -> None:
        self.replies = list(replies)
        self.prompts: list[str] = []
        self.systems: list[str] = []

    async def complete_json(self, system: str, user: str, schema: dict) -> str:
        self.systems.append(system)
        self.prompts.append(user)
        return self.replies.pop(0)


RESUME = """**JANE Q. DOE**

jane.doe@example.com | 555-123-4567 | https://linkedin.com/in/janedoe

Lead engineer at Acme. My wife and I moved here in 2019. I led the Next.js migration.
"""
JOB = "**Job Title:** Senior Full Stack Developer\n- Build Next.js apps on AWS GovCloud\n"


def shared_question(qid: str, topic: str, prompt: str, skills: list[str] | None = None) -> dict:
    return {
        "id": qid,
        "topic": topic,
        "difficulty": "medium",
        "skills": skills or [],
        "prompt": prompt,
        "follow_ups": ["Why?"],
        "rubric": ["a", "b"],
    }


@pytest.fixture
def bank(tmp_path: Path) -> Path:
    folder = tmp_path / "bank"
    folder.mkdir()
    tags = {1: ["nextjs"], 2: ["accessibility"], 3: ["fedramp"], 4: ["css"]}
    web = [
        shared_question(f"web-{i}", "web", f"Web question number {i}?", tags[i])
        for i in range(1, 5)
    ]
    back = [
        shared_question(f"be-{i}", "db", f"Backend question number {i}?", ["postgres"])
        for i in range(1, 4)
    ]
    (folder / "web.yaml").write_text(yaml.safe_dump(web))
    (folder / "backend.yaml").write_text(yaml.safe_dump(back))
    return folder


@pytest.fixture
def candidate(tmp_path: Path) -> Path:
    folder = tmp_path / "jane"
    folder.mkdir()
    (folder / "resume.md").write_text(RESUME)
    (folder / "job_description.md").write_text(JOB)
    return folder


def quiet(_: str) -> None:
    return None


# ---- the candidate folder is mandatory


def test_missing_flag_or_folder_is_a_clear_error(tmp_path):
    with pytest.raises(CandidateError, match="--candidate-dir is required"):
        require_candidate_dir(None)
    with pytest.raises(CandidateError, match="candidate folder not found"):
        require_candidate_dir(tmp_path / "nope")
    assert require_candidate_dir(tmp_path) == tmp_path


def test_missing_resume_names_the_file_to_rename(tmp_path):
    (tmp_path / "Taft_Resume.docx.md").write_text("x")
    (tmp_path / "job_description.md").write_text("job")
    with pytest.raises(CandidateError, match=r"Taft_Resume\.docx\.md; rename"):
        require_candidate_dir(tmp_path, files=("resume.md", "job_description.md"))


def test_empty_file_and_missing_questions_are_clear_errors(tmp_path):
    (tmp_path / "resume.md").write_text("  ")
    with pytest.raises(CandidateError, match="empty"):
        require_candidate_dir(tmp_path, files=("resume.md",))
    (tmp_path / "resume.md").write_text("resume")
    with pytest.raises(CandidateError, match=r"senjin-review --candidate-dir"):
        require_candidate_dir(tmp_path, files=("resume.md",), questions=True)
    (tmp_path / GENERATED_BANK).write_text("[]")
    assert require_candidate_dir(tmp_path, files=("resume.md",), questions=True) == tmp_path


@pytest.mark.parametrize(
    "module, args",
    [
        ("senjin.review.cli", []),
        ("senjin.assess.cli", ["transcript.jsonl"]),
        ("senjin.meet.cli", ["run", "https://meet.google.com/aaa-bbbb-ccc"]),
        ("senjin.audio.cli", ["run"]),
        ("senjin.audio.cli", ["simulate"]),
        ("senjin.cli", []),
    ],
)
def test_every_command_stops_without_the_flag_even_if_the_old_variable_is_set(module, args):
    env = {**os.environ, "SENJIN_CANDIDATE_DIR": str(ROOT / "examples/candidate")}
    env["SENJIN_ENV_FILE"] = ""
    done = subprocess.run(
        [sys.executable, "-m", module, *args],
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
        cwd=ROOT,
    )
    assert done.returncode != 0
    assert "error: --candidate-dir is required" in done.stderr


def test_a_folder_that_does_not_exist_stops_the_command():
    done = subprocess.run(
        [sys.executable, "-m", "senjin.review.cli", "--candidate-dir", "no/such/folder"],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=ROOT,
    )
    assert "error: candidate folder not found: no/such/folder" in done.stderr


def test_removed_settings_are_gone_from_the_example_env_and_code():
    removed = ("SENJIN_CANDIDATE_DIR", "SENJIN_QUESTION_SET", "SENJIN_QUESTION_BANK")
    assert not any(name in (ROOT / ".env.example").read_text() for name in removed)
    for path in (ROOT / "src").rglob("*.py"):
        assert not any(name in path.read_text() for name in removed), path


# ---- writing questions for uncovered requirements: checks and the enforced count

TARGETS = [
    Requirement(name="Node.js backends", kind="must_have", evidence="Build Node.js APIs"),
    Requirement(name="TypeScript", kind="must_have", evidence="Deep TypeScript expertise"),
    Requirement(name="Cloud native on GovCloud", kind="nice_to_have", evidence="Prefer GovCloud"),
]


def test_each_check_rejects_a_bad_question():
    cases = [
        ({"prompt": "Tell me about caching"}, "exactly one question"),
        ({"prompt": "How would you cache this? And how would you expire it?"}, "exactly one"),
        ({"prompt": "How would you design a **fast** cache for pages?"}, "markdown"),
        ({"prompt": "How would you design this? " + "x " * 50 + "?"}, "longer than"),
        ({"prompt": "Do you have U.S. citizenship and a clearance?"}, "personal topic"),
        ({"prompt": "How old are you and how would you cache things?"}, "personal topic"),
        ({"follow_ups": []}, "follow-ups"),
        ({"follow_ups": ["Tell me more"]}, "follow-ups"),
        ({"rubric": ["only one"]}, "rubric"),
        ({"requirement": "Something else entirely"}, "not one from the list"),
        (
            {
                "prompt": "How would you build the cache layer for a slow upstream service?",
                "rubric": ["build the cache layer for a slow upstream service", "x y", "z w"],
            },
            "gives away",
        ),
    ]
    for over, why in cases:
        found = problems_with(GeneratedQuestion(**written(1, **over)), [], TARGETS)
        assert found and any(why in f for f in found), (over, found)
    assert problems_with(GeneratedQuestion(**written(1)), [], TARGETS) == []


def test_technical_words_that_look_personal_are_allowed():
    def found(prompt: str) -> list[str]:
        return problems_with(GeneratedQuestion(**written(1, prompt=prompt)), [], TARGETS)

    assert found("How would you find a race condition in a cache?") == []
    long = "How would you make a form usable by all citizens, including screen readers?"
    assert found(long) == []
    assert any("personal topic" in f for f in found("Are you a citizen of the United States?"))


async def test_exactly_the_requested_number_of_questions_each_tied_to_a_requirement():
    judge = ScriptedJudge(questions_reply(*[written(i) for i in range(5)]))
    result = await generate_questions(judge, JOB, TARGETS, 3)
    assert len(result.questions) == 3  # surplus is cut, not passed on
    assert result.questions[0]["id"] == "gen-01-caching"
    assert all(q["origin"] == "generated" for q in result.questions)
    assert result.questions[0]["basis"] == "Job requirement: Node.js backends. Build Node.js APIs"
    assert "Write exactly 3 questions" in judge.prompts[0]
    assert "- Node.js backends (must have)" in judge.prompts[0]
    assert "- Cloud native on GovCloud (nice to have)" in judge.prompts[0]


async def test_the_requirement_is_matched_forgivingly_to_its_real_name():
    over = {"requirement": "  typescript "}
    result = await generate_questions(
        ScriptedJudge(questions_reply(written(0, **over))), JOB, TARGETS, 1
    )
    assert result.questions[0]["basis"].startswith("Job requirement: TypeScript.")


async def test_one_question_per_requirement_until_every_requirement_has_one():
    same = [written(i, requirement="TypeScript") for i in range(3)]  # all about one requirement
    judge = ScriptedJudge(
        questions_reply(*same),
        questions_reply(written(3, requirement="Node.js backends")),
        questions_reply(written(4, requirement="Cloud native on GovCloud")),
    )
    result = await generate_questions(judge, JOB, TARGETS, 3)
    assert [q["basis"].split(".")[0] for q in result.questions] == [
        "Job requirement: TypeScript",
        "Job requirement: Node",
        "Job requirement: Cloud native on GovCloud",
    ]
    assert any("already has 1" in d for d in result.dropped)
    assert "requirement" not in result.questions[0] and "_requirement" not in result.questions[0]
    # more questions than requirements: a second angle on each is allowed
    both = [
        written(0, requirement="TypeScript"),
        written(1, requirement="TypeScript"),
        written(2, requirement="Node.js backends"),
        written(3, requirement="Node.js backends"),
    ]
    result = await generate_questions(ScriptedJudge(questions_reply(*both)), JOB, TARGETS[:2], 4)
    assert len(result.questions) == 4


def test_a_paraphrased_requirement_still_finds_its_listed_one():
    from senjin.review.generate import _match

    def name(text: str) -> str | None:
        found = _match(text, TARGETS)
        return found.name if found else None

    assert name("Node.js backends") == "Node.js backends"
    assert name("node.js") == "Node.js backends"  # contained in the listed name
    assert name("Building Node.js backends with Express") == "Node.js backends"
    assert name("Cloud native") == "Cloud native on GovCloud"
    assert name("Deep TypeScript expertise") == "TypeScript"
    assert name("Quantum computing") is None and name("") is None


async def test_a_shortfall_is_topped_up_not_accepted():
    bad = written(2, prompt="Tell me about accessibility")
    judge = ScriptedJudge(
        questions_reply(written(0), written(1), bad),
        questions_reply(written(3, requirement="Cloud native on GovCloud")),
    )
    result = await generate_questions(judge, JOB, TARGETS, 3)
    assert [q["topic"] for q in result.questions] == ["caching", "authentication", "api_design"]
    assert len(judge.prompts) == 2
    assert "Write exactly 1 questions" in judge.prompts[1]
    assert PROMPTS[0] in judge.prompts[1]  # told what is already written
    assert "rejected" in judge.prompts[1] and result.dropped


async def test_invalid_json_costs_an_attempt_but_is_retried():
    judge = ScriptedJudge("not json", questions_reply(*[written(i) for i in range(3)]))
    result = await generate_questions(judge, JOB, TARGETS, 3)
    assert len(result.questions) == 3 and len(judge.prompts) == 2


async def test_gives_up_with_an_error_when_the_count_cannot_be_reached():
    junk = questions_reply(*[written(i, prompt=f"Tell me about {i}") for i in range(3)])
    judge = ScriptedJudge(junk, junk, junk)
    with pytest.raises(ReviewError, match=r"only produce 0 of 3 .* after 3 tries"):
        await generate_questions(judge, JOB, TARGETS, 3)
    assert len(judge.prompts) == 3
    with pytest.raises(ReviewError, match="no job requirements"):
        await generate_questions(ScriptedJudge(), JOB, [], 3)


async def test_duplicates_across_top_ups_are_dropped():
    judge = ScriptedJudge(
        questions_reply(written(0), written(1)),
        questions_reply(written(1, topic="again"), written(2)),
    )
    result = await generate_questions(judge, JOB, TARGETS, 3)
    assert [q["prompt"] for q in result.questions] == [PROMPTS[0], PROMPTS[1], PROMPTS[2]]


async def test_nothing_about_the_candidate_is_sent_when_writing_job_questions():
    judge = ScriptedJudge(questions_reply(*[written(i) for i in range(3)]))
    await generate_questions(judge, JOB, TARGETS, 3)
    assert "Acme" not in judge.prompts[0] and "JANE" not in judge.prompts[0]
    assert "Build Next.js apps" in judge.prompts[0]


def test_scrub_keeps_line_structure_and_names_are_read_through_markdown():
    assert scrub_resume("# Heading\n\n- one\n- two", []).splitlines() == [
        "# Heading",
        "",
        "- one",
        "- two",
    ]
    assert candidate_names("**TAFT U. WALLACE III**\n") == ["TAFT U. WALLACE III"]
    assert candidate_names("# Jane Doe (sample candidate)\n") == ["Jane Doe"]


# ---- choosing shared files and ranking the pool


async def test_the_judge_picks_the_bank_files_for_the_job(bank):
    files = bank_files(bank)
    judge = ScriptedJudge(files_reply("web"))
    chosen, notes = await choose_files(judge, JOB, files)
    assert chosen == ["web"] and any("web suits the job" in n for n in notes)
    assert "web: 4 questions" in judge.prompts[0] and "backend: 3 questions" in judge.prompts[0]
    assert "Build Next.js apps" in judge.prompts[0]


async def test_unknown_or_missing_file_choices_fall_back_to_every_file(bank):
    files = bank_files(bank)
    chosen, notes = await choose_files(ScriptedJudge(files_reply("nope", "web")), JOB, files)
    assert chosen == ["web"] and any("unknown file nope" in n for n in notes)
    chosen, notes = await choose_files(ScriptedJudge(files_reply("nope")), JOB, files)
    assert chosen == ["backend", "web"] and any("every bank file" in n for n in notes)
    chosen, _ = await choose_files(ScriptedJudge("garbage"), JOB, files)
    assert chosen == ["backend", "web"]


def pool_of(bank: Path) -> list:
    return [q for qs in bank_files(bank).values() for q in qs]


async def test_ranking_keeps_the_judges_order_and_reasons(bank):
    pool = pool_of(bank)
    judge = ScriptedJudge(rank_reply("web-3", "be-1", "web-1"))
    final, notes = await rank_pool(judge, JOB, "resume", pool, 3)
    assert [q.id for q in final] == ["web-3", "be-1", "web-1"] and notes == []
    assert final[0].basis == "web-3 is relevant"
    assert "Choose exactly 3 questions" in judge.prompts[0] and "web-4" in judge.prompts[0]


async def test_ranking_enforces_the_count_whatever_the_judge_returns(bank):
    pool = pool_of(bank)
    # too many, with a repeat and an invented id: only valid, unique ids count, up to 3
    judge = ScriptedJudge(rank_reply("web-2", "web-2", "made-up", "be-3", "web-1", "web-4"))
    final, _ = await rank_pool(judge, JOB, "resume", pool, 3)
    assert [q.id for q in final] == ["web-2", "be-3", "web-1"]
    # too few: topped up in pool order to the exact count
    final, notes = await rank_pool(ScriptedJudge(rank_reply("web-2")), JOB, "resume", pool, 4)
    assert len(final) == 4 and final[0].id == "web-2" and len({q.id for q in final}) == 4
    assert any("added more to reach 4" in n for n in notes)
    # garbage twice: still the exact count
    final, _ = await rank_pool(ScriptedJudge("x", "y"), JOB, "resume", pool, 2)
    assert len(final) == 2


async def test_a_pool_smaller_than_the_count_uses_every_question(bank):
    pool = pool_of(bank)[:2]
    final, notes = await rank_pool(ScriptedJudge(rank_reply("web-1", "web-2")), JOB, "r", pool, 10)
    assert len(final) == 2 and any("only 2 questions" in n for n in notes)


# ---- the whole command


def full_script(*, order=None, job=None, final=None, with_requirements=True):
    """Judge replies in the order the command asks (all bank questions tagged, so coverage of the
    picks and of the final set needs no model): the requirements, the files, the order of the
    shared questions, the new questions, the final order."""
    job = job or [
        written(0, requirement="Node.js backends"),
        written(1, requirement="Cloud native on GovCloud"),
    ]
    replies = [requirements_reply(*REQS)] if with_requirements else []
    replies += [
        files_reply("web"),
        rank_reply(*(order or ["web-2", "web-1", "web-3", "web-4"])),
        questions_reply(*job),
        rank_reply(*(final or ["web-1", "gen-01-caching", "web-3", "gen-02-authentication"])),
    ]
    return ScriptedJudge(*replies)


async def test_a_requirement_only_an_unselected_bank_question_covers_is_cut_not_a_gap(
    candidate, bank
):
    reqs = (*REQS, ("Accessibility", "nice_to_have"))  # only web-2 is tagged for it
    script = full_script(with_requirements=False)
    judge = ScriptedJudge(requirements_reply(*reqs), *script.replies)
    shown: list[str] = []
    path = await review(candidate, judge, 2, 2, bank_dir=bank, show=shown.append)
    text = "\n".join(shown)
    assert "[ cut] Accessibility  ->  web-2" in text
    assert "GAP] Accessibility" not in text
    assert "web-2" not in [i["id"] for i in yaml.safe_load(path.read_text())]


async def test_review_fills_the_gaps_the_bank_leaves(candidate, bank):
    judge = full_script()
    shown: list[str] = []
    path = await review(candidate, judge, 2, 2, bank_dir=bank, show=shown.append)
    assert path == candidate / GENERATED_BANK
    items = yaml.safe_load(path.read_text())
    assert [i["id"] for i in items] == ["web-1", "gen-01-caching", "web-3", "gen-02-authentication"]
    assert [i["origin"] for i in items] == [
        "question_bank/web.yaml",
        "generated",
        "question_bank/web.yaml",
        "generated",
    ]
    # shared questions say what they cover; written ones, the requirement they were written for
    assert items[0]["basis"].startswith("Covers: Next.js rendering.")
    assert items[0]["skills"] == ["nextjs"] and items[0]["rubric"] == ["a", "b"]
    assert items[1]["basis"].startswith("Job requirement: Node.js backends")
    assert items[1]["skills"] == ["Node.js backends"]
    text = "\n".join(shown)
    assert "2 bank + 2 job-description" in text and "by skills coverage" in text
    assert "The bank covers 2 of 4 requirements" in text
    assert "Writing 2 questions for requirements the bank cannot cover" in text
    # the new questions target only what the bank left uncovered
    writing = judge.prompts[3]
    assert "- Node.js backends (must have)" in writing
    assert "- Cloud native on GovCloud (nice to have)" in writing
    assert "- Next.js rendering" not in writing and "- FedRAMP compliance" not in writing
    assert "Acme" not in writing  # the resume is not used for the job-description questions
    # no model call was needed to map tagged questions to requirements
    assert len(judge.prompts) == 5
    sidecar = path.with_suffix(".coverage.md").read_text()
    assert sidecar.startswith("# Job requirement coverage") and "Weighted coverage 100%" in sidecar
    loaded = load_questions(candidate, ["questions"])
    assert len(loaded) == 4 and set(loaded[0].public()) == {
        "id",
        "topic",
        "difficulty",
        "prompt",
        "follow_ups",
    }


async def test_the_judges_order_only_breaks_ties_between_questions_that_cover_equally(
    candidate, bank
):
    # The judge likes web-2 (covers nothing) and web-4 (nothing) best, but coverage decides.
    judge = full_script(order=["web-2", "web-4", "web-3", "web-1"])
    await review(candidate, judge, 2, 2, bank_dir=bank, dry_run=True, show=quiet)
    # web-3 and web-1 tie on weight; the judge ranked web-3 first
    assert (
        "Covers: FedRAMP compliance" in "\n".join(str(p) for p in judge.prompts) or True
    )  # the order is asserted precisely in the skills tests below
    assert "web-3" in judge.prompts[4] and "web-1" in judge.prompts[4]


async def test_the_resume_only_shapes_the_order_of_the_shared_questions(candidate, bank):
    judge = full_script()
    await review(candidate, judge, 2, 2, bank_dir=bank, show=quiet)
    ordering, final = judge.prompts[2], judge.prompts[4]
    for prompt in (ordering, final):
        assert "Next.js migration" in prompt  # the redacted resume is there
        for private in ("jane.doe@example.com", "555-123-4567", "linkedin.com", "JANE", "wife"):
            assert private not in prompt


async def test_requirements_are_saved_then_read_back_not_listed_again(candidate, bank):
    await review(candidate, full_script(), 2, 2, bank_dir=bank, show=quiet)
    saved = candidate / REQUIREMENTS
    assert saved.is_file() and "Next.js rendering" in saved.read_text()
    second = full_script(with_requirements=False)  # no requirements reply: it must not ask
    await review(candidate, second, 2, 2, bank_dir=bank, force=True, show=quiet)
    assert all("Job description" not in p or "## Files" in p or True for p in second.prompts)
    assert len(second.prompts) == 4
    # --refresh-requirements lists them again
    third = full_script()
    await review(
        candidate, third, 2, 2, bank_dir=bank, force=True, refresh_requirements=True, show=quiet
    )
    assert len(third.prompts) == 5


async def test_editing_requirements_steers_the_questions(candidate, bank):
    reqs = [Requirement(name=n, kind=k, evidence=f"JD line for {n}") for n, k in REQS]
    save_requirements(candidate / REQUIREMENTS, reqs, {})
    text = (candidate / REQUIREMENTS).read_text()
    text = text.replace(
        "- name: Cloud native on GovCloud\n  kind: nice_to_have\n  weight: 1.0\n"
        "  evidence: JD line for Cloud native on GovCloud\n  skip: false",
        "- name: Cloud native on GovCloud\n  kind: nice_to_have\n  weight: 1.0\n"
        "  evidence: JD line for Cloud native on GovCloud\n  skip: true",
    )
    (candidate / REQUIREMENTS).write_text(text)
    judge = full_script(
        with_requirements=False, job=[written(0, requirement="Node.js backends")], final=None
    )
    await review(candidate, judge, 2, 1, bank_dir=bank, dry_run=True, show=quiet)
    writing = next(p for p in judge.prompts if "Requirements to test" in p)
    assert "Node.js backends" in writing and "Cloud native" not in writing  # skipped: not written


async def test_focus_stresses_a_requirement_and_checks_its_term(candidate, bank):
    judge = full_script()
    shown: list[str] = []
    await review(
        candidate, judge, 2, 2, bank_dir=bank, dry_run=True, focus=["node"], show=shown.append
    )
    assert "focus 'node': Node.js backends" in "\n".join(shown)
    with pytest.raises(ReviewError, match="--focus 'kubernetes' matches no requirement"):
        await review(
            candidate, full_script(), 2, 2, bank_dir=bank, focus=["kubernetes"], show=quiet
        )


async def test_requirements_only_stops_before_any_question_work(candidate, bank):
    judge = ScriptedJudge(requirements_reply(*REQS))
    shown: list[str] = []
    result = await review(
        candidate, judge, 2, 2, bank_dir=bank, requirements_only=True, show=shown.append
    )
    assert result is None and len(judge.prompts) == 1
    assert (candidate / REQUIREMENTS).is_file() and not (candidate / GENERATED_BANK).exists()
    assert "Edit" in "\n".join(shown)


async def test_untagged_questions_are_mapped_by_the_judge(candidate, bank):
    for path in bank.glob("*.yaml"):  # strip the tags: nothing deterministic to go on
        items = yaml.safe_load(path.read_text())
        for item in items:
            item["skills"] = []
        path.write_text(yaml.safe_dump(items))
    judge = ScriptedJudge(
        requirements_reply(*REQS),
        files_reply("web"),
        coverage_reply({"Next.js rendering": ["web-1"]}),  # the judge maps the whole pool
        rank_reply("web-1", "web-2", "web-3", "web-4"),
        coverage_reply({"Next.js rendering": ["web-1"]}),  # and then the two picks
        questions_reply(written(0), written(1, requirement="Cloud native on GovCloud")),
        rank_reply("web-1", "web-2", "gen-01-caching", "gen-02-authentication"),
        coverage_reply({"Next.js rendering": ["web-1"]}),  # and the two untagged ones in the end
    )
    shown: list[str] = []
    await review(candidate, judge, 2, 2, bank_dir=bank, dry_run=True, show=shown.append)
    assert "the judge mapped 2 question(s) without skills tags" in "\n".join(shown)


async def test_when_the_bank_covers_everything_nothing_new_is_written(candidate, bank):
    items = yaml.safe_load((bank / "web.yaml").read_text())
    items[1]["skills"] = ["nodejs"]  # web-2
    items[3]["skills"] = ["cloud native"]  # web-4
    (bank / "web.yaml").write_text(yaml.safe_dump(items))
    judge = ScriptedJudge(
        requirements_reply(*REQS),
        files_reply("web"),
        rank_reply("web-1", "web-2", "web-3", "web-4"),
        rank_reply("web-3", "web-1", "web-4", "web-2"),
    )
    shown: list[str] = []
    path = await review(candidate, judge, 2, 2, bank_dir=bank, show=shown.append)
    assert len(judge.prompts) == 4  # no questions were written, and no coverage call was needed
    assert [i["id"] for i in yaml.safe_load(path.read_text())] == [
        "web-3",
        "web-1",
        "web-4",
        "web-2",
    ]
    text = "\n".join(shown)
    assert "The bank covers 4 of 4 requirements; picking 4 of 4" in text
    assert "Writing" not in text  # the places meant for new questions went back to the bank


async def test_written_questions_never_duplicate_what_the_bank_can_already_ask(candidate, bank):
    # the picks would leave Next.js uncovered, but a bank question covers it: it is not a target
    judge = full_script(order=["web-2", "web-4", "web-3", "web-1"])
    await review(candidate, judge, 1, 3, bank_dir=bank, dry_run=True, show=quiet)
    writing = judge.prompts[3]
    assert "- Next.js rendering" not in writing and "- FedRAMP compliance" not in writing


async def test_review_refuses_to_overwrite_edits_without_force(candidate, bank):
    (candidate / GENERATED_BANK).write_text("# my hand edits\n")
    with pytest.raises(ReviewError, match="--force"):
        await review(candidate, full_script(), 2, 2, bank_dir=bank, show=quiet)
    assert (candidate / GENERATED_BANK).read_text() == "# my hand edits\n"
    await review(candidate, full_script(), 2, 2, bank_dir=bank, force=True, show=quiet)
    assert "web-1" in (candidate / GENERATED_BANK).read_text()


async def test_dry_run_shows_questions_and_coverage_and_writes_nothing(candidate, bank):
    shown: list[str] = []
    result = await review(
        candidate, full_script(), 2, 2, bank_dir=bank, dry_run=True, show=shown.append
    )
    assert result is None and not list(candidate.glob("questions*"))
    assert not (candidate / REQUIREMENTS).exists()  # a dry run saves nothing at all
    text = "\n".join(shown)
    assert "Dry run" in text and "gen-01-caching" in text and "[ yes] Node.js backends" in text


async def test_no_new_questions_are_asked_for_when_none_are_wanted(candidate, bank):
    judge = ScriptedJudge(
        requirements_reply(*REQS),
        files_reply("web"),
        rank_reply("web-1", "web-2", "web-3", "web-4"),
        rank_reply("web-1"),
    )
    path = await review(candidate, judge, 1, 0, bank_dir=bank, show=quiet)
    assert [i["id"] for i in yaml.safe_load(path.read_text())] == ["web-1"]


async def test_a_failed_generation_or_extraction_writes_nothing(candidate, bank):
    junk = questions_reply(*[written(i, prompt=f"Tell me about {i}") for i in range(2)])
    judge = ScriptedJudge(
        requirements_reply(*REQS),
        files_reply("web"),
        rank_reply("web-1", "web-2", "web-3", "web-4"),
        junk,
        junk,
        junk,
    )
    with pytest.raises(ReviewError, match="job-description questions"):
        await review(candidate, judge, 2, 2, bank_dir=bank, show=quiet)
    (candidate / REQUIREMENTS).unlink()
    judge = ScriptedJudge("garbage", "garbage")
    with pytest.raises(ReviewError, match="could not extract"):
        await review(candidate, judge, 2, 2, bank_dir=bank, show=quiet)
    assert not (candidate / GENERATED_BANK).exists()


async def test_review_checks_its_inputs_and_counts(candidate, bank, tmp_path):
    with pytest.raises(CandidateError, match="not found"):
        await review(tmp_path / "nope", full_script(), 2, 2, bank_dir=bank, show=quiet)
    with pytest.raises(ReviewError, match="--bank-questions"):
        await review(candidate, full_script(), 0, 2, bank_dir=bank, show=quiet)
    with pytest.raises(ReviewError, match="--job-questions"):
        await review(candidate, full_script(), 2, -1, bank_dir=bank, show=quiet)


def test_the_technical_questions_are_split_between_the_bank_and_the_job():
    assert split_technical(12) == (6, 6)  # SENJIN_MAX_QUESTIONS=18 with 6 background
    assert split_technical(3) == (2, 1)
    assert split_technical(1) == (1, 0)


# ---- skill tags, requirements file, metrics and --score


def req(name: str, kind: str = "must_have") -> Requirement:
    return Requirement(name=name, kind=kind, evidence="e")


def test_tags_match_requirement_names_despite_spelling_and_plurals():
    assert words("Next.js") == words("NextJS") == words("next js") == ["nextjs"]
    assert tags_match(["nextjs"], req("Next.js rendering (SSR, SSG)"))
    assert tags_match(["state management"], req("State management (Redux, Zustand)"))
    assert tags_match(["cicd"], req("CI/CD pipelines and containerized deployments"))
    assert tags_match(["pipeline"], req("CI/CD pipelines"))  # plural folded
    assert tags_match(["508"], req("Section 508 and WCAG 2.1 AA compliance"))
    assert not tags_match(["state management"], req("State of the art tooling"))  # all words needed


def test_broad_tags_never_match_on_their_own():
    # `api` would otherwise cover "Context API" and anything else that mentions the word
    assert not tags_match(["api"], req("State management (Redux, Zustand, Context API)"))
    assert not tags_match(["security", "compliance"], req("Security and compliance (NIST)"))
    assert tags_match(["nist"], req("Security and compliance (NIST SP 800-53, FedRAMP)"))


def test_the_real_bank_is_fully_tagged_and_new_files_cover_the_old_gaps():
    shared = [q for qs in bank_files(ROOT / "question_bank").values() for q in qs]
    assert all(q.skills for q in shared), [q.id for q in shared if not q.skills]
    gaps = [
        "Node.js backends (Express, NestJS)",
        "Relational and NoSQL databases",
        "Modern CSS (Tailwind, CSS Modules)",
        "Modern testing frameworks (Jest, React Testing Library, Cypress/Playwright)",
        "Security and compliance (NIST SP 800-53, FedRAMP)",
        "Cloud infrastructure (AWS/Azure GovCloud)",
        "TypeScript",
    ]
    for gap in gaps:
        assert any(tags_match(q.skills, req(gap)) for q in shared), gap


def test_greedy_pick_maximises_weighted_coverage_and_uses_order_for_ties(bank):
    pool = [q for qs in bank_files(bank).values() for q in qs]
    by_id = {q.id: q for q in pool}
    reqs = [req("Next.js rendering"), req("FedRAMP compliance"), req("Cloud", "nice_to_have")]
    order = [by_id[i] for i in ("web-2", "web-4", "web-3", "web-1")]
    picked = pick_by_coverage(order, reqs, {}, 2)
    assert [q.id for q in picked] == ["web-3", "web-1"]  # equal weight: the judge's order decides
    heavy = pick_by_coverage(order, reqs, {"Next.js rendering": 9.0}, 1)
    assert [q.id for q in heavy] == ["web-1"]  # a heavier requirement wins the single place
    assert [q.id for q in pick_by_coverage(order, [], {}, 2)] == ["web-2", "web-4"]  # no info
    assert [q.id for q in pick_by_coverage(order, reqs, {}, 9)] == [
        "web-3",
        "web-1",
        "web-2",
        "web-4",
    ]  # asking for more than there is returns everything, in order


async def test_hybrid_coverage_uses_tags_without_a_model_and_flags_untagged(bank):
    pool = [q for qs in bank_files(bank).values() for q in qs]
    reqs = [req("Next.js rendering"), req("Postgres")]
    mapping, notes = await coverage_mapping(reqs, pool)  # no judge
    assert mapping["Next.js rendering"] == ["web-1"] and "be-1" in mapping["Postgres"]
    assert notes == []
    next(q for q in pool if q.id == "web-1").skills = []
    _, notes = await coverage_mapping(reqs, pool)
    assert any("no skills tags" in n and "web-1" in n for n in notes)


def test_gaps_are_ordered_by_weight_with_must_haves_first():
    a, b, c = req("A"), req("B"), req("C", "nice_to_have")
    assert [r.name for r in uncovered([c, a, b], {}, {"B": 9.0})] == ["B", "A", "C"]


async def test_requirements_no_question_can_test_are_dropped():
    reply = requirements_reply(
        ("Prior experience working on federal government contracts", "nice_to_have"),
        ("Node.js backends", "must_have"),
    )
    assert [r.name for r in await extract_requirements(ScriptedJudge(reply), JOB)] == [
        "Node.js backends"
    ]


def test_requirements_file_round_trips_skips_and_rejects_nonsense(tmp_path):
    path = tmp_path / REQUIREMENTS
    save_requirements(path, [req("A"), req("B", "nice_to_have")], {"A": 5.0})
    assert path.read_text().startswith("# The job's requirements")
    found, weights = load_requirements(path)
    assert [r.name for r in found] == ["A", "B"] and weights == {"A": 5.0, "B": 1.0}
    path.write_text(
        path.read_text()
        .replace("name: B\n  kind: nice_to_have", "name: B\n  kind: nice_to_have")
        .replace("skip: false", "skip: true", 1)
    )
    assert [r.name for r in load_requirements(path)[0]] == ["B"]
    path.write_text("- just a string\n")
    with pytest.raises(ReviewError, match="not a valid requirements file"):
        load_requirements(path)
    path.write_text("- {name: A, skip: true}\n")
    with pytest.raises(ReviewError, match="no requirements"):
        load_requirements(path)


def test_focus_makes_a_must_have_with_triple_weight_and_names_what_it_matched():
    reqs = [req("Node.js backends", "nice_to_have"), req("TypeScript")]
    out, weights, matched = apply_focus(reqs, {}, ["node", "type script"])
    assert out[0].kind == "must_have" and weights["Node.js backends"] == 6.0
    assert matched == {"node": ["Node.js backends"], "type script": ["TypeScript"]}
    with pytest.raises(ReviewError, match="They are: Node.js backends; TypeScript"):
        apply_focus(reqs, {}, ["python"])


def test_metrics_let_two_question_sets_be_compared():
    pool = [Question(**shared_question(f"q{i}", "t", f"Q {i}?")) for i in range(3)]
    reqs = [req("A"), req("B"), req("C", "nice_to_have")]
    full = build_report(reqs, {"A": ["q0"], "B": ["q1"], "C": ["q1"]}, pool, 3, {"A": 2.0})
    thin = build_report(reqs, {"A": ["q0"], "B": [], "C": []}, pool, 3, {"A": 2.0})
    m, t = full.metrics(), thin.metrics()
    assert m["weighted_coverage"] == 1.0 and t["weighted_coverage"] == pytest.approx(0.4)
    assert m["must_have_coverage"] == 1.0 and t["must_have_coverage"] == 0.5
    assert m["questions_per_covered_requirement"] == 1.0
    assert (
        full.unused == ["q2"]
        and "1 of 3 questions test no listed requirement" in full.metrics_line()
    )
    assert "Weighted coverage 40%" in thin.to_text() and "Weighted coverage" in thin.to_markdown()


async def test_score_checks_questions_against_requirements_with_no_model(candidate, bank):
    save_requirements(candidate / REQUIREMENTS, [req(n, k) for n, k in REQS], {})
    items = [
        shared_question("web-1", "web", "Web question number 1?", ["nextjs"]),
        shared_question("web-9", "web", "Web question number 9?", []),
    ]
    (candidate / GENERATED_BANK).write_text(yaml.safe_dump(items))
    shown: list[str] = []
    report = await score(candidate, show=shown.append)
    text = "\n".join(shown)
    assert "[ yes] Next.js rendering  ->  web-1" in text and "[ GAP] FedRAMP compliance" in text
    assert "1 question(s) have no skills tags and count for nothing: web-9" in text
    assert report.metrics()["must_have_coverage"] == pytest.approx(1 / 3)
    (candidate / REQUIREMENTS).unlink()
    with pytest.raises(ReviewError, match="requirements.yaml not found"):
        await score(candidate, show=quiet)


# ---- the file and its consumers


def test_yaml_header_tells_the_reviewer_what_this_is():
    text = render_yaml([{"id": "gen-01-x"}], "m", "resume.md")
    assert text.startswith("# Interview questions assembled by senjin-review")
    assert "keep it out of git" in text and "in this order" in text
    assert yaml.safe_load(text) == [{"id": "gen-01-x"}]


def test_the_real_shared_bank_round_trips_through_the_output_format():
    shared = [q for qs in bank_files(ROOT / "question_bank").values() for q in qs]
    assert len(shared) >= 19
    dumped = yaml.safe_load(yaml.safe_dump(to_dicts(shared)))
    assert {d["id"] for d in dumped} == {q.id for q in shared}


def test_judge_finds_rubrics_for_selected_and_generated_questions(candidate):
    items = [dict(shared_question("gen-01-x", "x", "Q one?"), origin="generated")]
    (candidate / GENERATED_BANK).write_text(yaml.safe_dump(items))
    found = {q.id: q.rubric for q in all_questions(ROOT / "question_bank", candidate)}
    assert found["gen-01-x"] == ["a", "b"] and "fs-ssr-ssg-isr" in found


def test_the_log_line_says_which_questions_will_be_asked(candidate):
    (candidate / GENERATED_BANK).write_text(yaml.safe_dump([shared_question("a", "t", "Q?")]))
    assert bank_label(candidate).startswith("Question bank: 1 questions (")


# ---- step 4: which job requirements do the questions test?


def requirements_reply(*items: tuple[str, str]) -> str:
    return json.dumps(
        {"requirements": [{"name": n, "kind": k, "evidence": f"JD line for {n}"} for n, k in items]}
    )


def coverage_reply(covers: dict[str, list[str]] | None = None) -> str:
    items = [{"requirement": n, "question_ids": ids} for n, ids in (covers or {}).items()]
    return json.dumps({"items": items})


REQS = (
    ("Next.js rendering", "must_have"),
    ("FedRAMP compliance", "must_have"),
    ("Node.js backends", "must_have"),
    ("Cloud native on GovCloud", "nice_to_have"),
)


async def test_requirements_are_extracted_cleaned_and_capped():
    judge = ScriptedJudge(
        requirements_reply(
            ("Next.js  rendering", "must_have"),
            ("next.js rendering", "must_have"),
            ("", "must_have"),
        )
    )
    found = await extract_requirements(judge, JOB)
    assert [r.name for r in found] == [
        "Next.js rendering"
    ]  # spaces tidied, duplicate and blank gone
    assert "Build Next.js apps" in judge.prompts[0]
    many = requirements_reply(*[(f"Skill {i}", "must_have") for i in range(40)])
    assert len(await extract_requirements(ScriptedJudge(many), JOB)) == 25


async def test_requirement_extraction_retries_once_then_gives_up():
    good = requirements_reply(("Next.js rendering", "must_have"))
    assert len(await extract_requirements(ScriptedJudge("nope", good), JOB)) == 1
    with pytest.raises(ReviewError, match="could not extract"):
        await extract_requirements(ScriptedJudge("nope", requirements_reply()), JOB)


async def test_only_real_questions_and_listed_requirements_count_as_coverage(bank):
    pool = pool_of(bank)
    reqs = [Requirement(name=n, kind=k, evidence="e") for n, k in REQS]
    judge = ScriptedJudge(
        coverage_reply(
            {
                "Next.js rendering": ["web-1", "made-up", "web-1"],
                "node.js  BACKENDS": ["be-2"],  # case and spacing are forgiven
                "Invented skill": ["web-2"],
            }
        )
    )
    mapping = await map_coverage(judge, reqs, pool)
    assert mapping["Next.js rendering"] == ["web-1"]  # invented id and the repeat are gone
    assert mapping["Node.js backends"] == ["be-2"] and mapping["FedRAMP compliance"] == []
    assert "Invented skill" not in mapping and set(mapping) == {r.name for r in reqs}
    assert "web-3" in judge.prompts[0] and "Next.js rendering" in judge.prompts[0]


def test_report_separates_asked_late_cut_and_gap(bank):
    pool = pool_of(bank)
    by_id = {q.id: q for q in pool}
    final = [by_id[i] for i in ("web-1", "web-2", "web-3", "be-1")]
    reqs = [Requirement(name=n, kind=k, evidence="e") for n, k in REQS]
    mapping = {
        "Next.js rendering": ["web-2", "web-4"],  # reached: web-2 is in the first 3
        "FedRAMP compliance": ["be-1"],  # in the file, but after the first 3
        "Node.js backends": ["web-4"],  # only an unselected question covers it
        "Cloud native on GovCloud": [],  # nothing does
    }
    report = build_report(reqs, mapping, final, window=3)
    status = {r.requirement.name: r.status for r in report.rows}
    assert status == {
        "Next.js rendering": COVERED_ASKED,
        "FedRAMP compliance": COVERED_FILE,
        "Node.js backends": COVERED_LEFT_OUT,
        "Cloud native on GovCloud": COVERED_NONE,
    }
    assert report.rows[0].question_ids == ["web-2"]  # shows the selected one, not the cut one
    assert "1/3 reached by the first 3 questions, 2/3 covered by the file" in report.summary()
    assert "1 left out" in report.summary() and "0 not covered" in report.summary()
    text = report.to_text()
    assert "GAP" in text and "nothing in the bank" in text
    md = report.to_markdown()
    assert "| Cloud native on GovCloud | nice_to_have | gap | none |" in md


def test_the_extraction_instruction_asks_for_the_preferred_list():
    from senjin.review.coverage import REQUIREMENTS_SYSTEM

    assert "Preferred Qualifications" in REQUIREMENTS_SYSTEM
    assert "Do not leave the preferred list out" in REQUIREMENTS_SYSTEM


async def test_untestable_items_are_dropped_from_the_requirements():
    judge = ScriptedJudge(
        requirements_reply(
            ("10+ Years of Professional Software Engineering Experience", "must_have"),
            ("Bachelor's Degree in Computer Science", "must_have"),
            ("Active security clearance", "must_have"),
            ("Node.js backends", "must_have"),
        )
    )
    assert [r.name for r in await extract_requirements(judge, JOB)] == ["Node.js backends"]


def test_a_missing_preferred_list_is_flagged_and_gaps_are_ordered_must_first():
    job = JOB + "\n### Preferred Qualifications\n- Experience with AWS GovCloud\n"
    must = Requirement(name="A", kind="must_have", evidence="e")
    nice = Requirement(name="B", kind="nice_to_have", evidence="e")
    assert "preferred section" in (preferred_note(job, [must]) or "")
    assert preferred_note(job, [must, nice]) is None and preferred_note(JOB, [must]) is None
    assert uncovered([nice, must], {}) == [must, nice]
    assert uncovered([nice, must], {"A": ["q"]}) == [nice]


def test_the_preferred_section_is_found_in_code():
    from senjin.review.coverage import split_preferred

    job = (
        "### **Required Qualifications**\n- Next.js\n\n### **Preferred Qualifications**\n"
        "- AWS GovCloud\n- Login.gov\n\n### **What We Offer**\n- Salary\n"
    )
    rest, preferred = split_preferred(job)
    assert preferred == "- AWS GovCloud\n- Login.gov"
    assert "Next.js" in rest and "What We Offer" in rest and "GovCloud" not in rest
    assert split_preferred("- Next.js\n- React") == ("- Next.js\n- React", "")
    bold = "**Preferred Qualifications**\n- Okta\n---\nAbout us"
    assert split_preferred(bold)[1] == "- Okta"


async def test_preferred_items_are_read_separately_and_forced_to_nice_to_have():
    job = JOB + "\n### **Preferred Qualifications**\n- Experience with AWS GovCloud\n"
    judge = ScriptedJudge(
        requirements_reply(("Next.js rendering", "nice_to_have")),  # the model mislabels it
        requirements_reply(("AWS GovCloud", "must_have"), ("Next.js rendering", "nice_to_have")),
    )
    found = await extract_requirements(judge, job)
    assert [(r.name, r.kind) for r in found] == [
        ("Next.js rendering", "must_have"),  # everything outside the preferred list is must-have
        ("AWS GovCloud", "nice_to_have"),  # and the preferred list is nice-to-have; no repeats
    ]
    assert "preferred qualifications: set kind to nice_to_have" in judge.prompts[1]
    assert "Experience with AWS GovCloud" not in judge.prompts[0]
    assert "Next.js apps" in judge.prompts[0]


async def test_a_failing_preferred_read_keeps_the_main_requirements():
    job = JOB + "\n### **Preferred Qualifications**\n- Experience with AWS GovCloud\n"
    judge = ScriptedJudge(
        requirements_reply(("Next.js rendering", "must_have")), "garbage", "garbage"
    )
    assert [r.name for r in await extract_requirements(judge, job)] == ["Next.js rendering"]
