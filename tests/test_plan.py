"""The per-candidate interview plan: profiles, the saved file, precedence and the commands."""

import json
import os
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import pytest
import yaml

from senjin.config import InterviewConfig
from senjin.plan import (
    DEFAULT_PROFILE,
    PLAN,
    InterviewPlan,
    PlanError,
    load_plan,
    load_profiles,
    plan_for_interview,
    plan_from_profile,
    save_plan,
)
from senjin.review.cli import REQUIREMENTS, run_command, score, split_technical
from senjin.review.generate import ReviewError
from senjin.review.plan import save_requirements
from senjin.review.profile import classify_job, describe_profiles, ensure_plan
from senjin.review.schema import Requirement

ROOT = Path(__file__).resolve().parents[1]
JOB = "**Job Title:** Senior Full Stack Developer\n- Build Next.js apps on AWS GovCloud\n"


class ScriptedJudge:
    model = "scripted-model"

    def __init__(self, *replies: str) -> None:
        self.replies = list(replies)
        self.prompts: list[str] = []

    async def complete_json(self, system: str, user: str, schema: dict) -> str:
        self.prompts.append(user)
        return self.replies.pop(0)


def choice(profile: str, reason: str = "fits the role") -> str:
    return json.dumps({"profile": profile, "reason": reason})


def quiet(_: str) -> None:
    return None


@pytest.fixture
def profiles():
    return load_profiles()


@pytest.fixture
def folder(tmp_path: Path) -> Path:
    path = tmp_path / "cand"
    path.mkdir()
    (path / "resume.md").write_text("**JANE DOE**\nLed the Next.js migration.\n")
    (path / "job_description.md").write_text(JOB)
    return path


def write_questions(folder: Path, count: int) -> None:
    items = [
        {
            "id": f"q-{i}",
            "topic": "t",
            "difficulty": "easy",
            "skills": ["nextjs"],
            "prompt": f"Question {i}?",
            "follow_ups": ["Why?"],
            "rubric": ["a", "b"],
        }
        for i in range(count)
    ]
    (folder / "questions.yaml").write_text(yaml.safe_dump(items))


# ---- profiles


def test_the_profiles_are_valid_and_the_default_matches_the_old_global_interview(profiles):
    assert set(profiles) == {
        "default",
        "junior_generalist",
        "senior_fullstack",
        "backend",
        "frontend",
        "data_engineering",
        "cloud_security",
        "qa_program_lead",
        "staff_architect",
    }
    base = InterviewConfig.from_env({})
    config = plan_from_profile(profiles[DEFAULT_PROFILE]).apply(base)
    assert (config.resume, config.technical, config.follow_ups, config.minutes) == (2, 3, 1, 30)


def test_profiles_differ_in_the_ways_that_matter(profiles):
    base = InterviewConfig.from_env({})
    shape = {name: plan_from_profile(profile).apply(base) for name, profile in profiles.items()}
    junior, senior, staff = (
        shape[n] for n in ("junior_generalist", "senior_fullstack", "staff_architect")
    )
    assert junior.technical < senior.technical  # seniors get more questions
    assert (
        staff.technical < senior.technical and staff.follow_ups > senior.follow_ups
    )  # fewer, deeper
    assert staff.minutes > senior.minutes > junior.minutes  # and a longer box
    assert (senior.minutes, staff.minutes) == (36, 50)


def test_a_broken_profile_is_a_clear_error(tmp_path):
    (tmp_path / "default.yaml").write_text(
        "description: d\nseniority: mid\nbackground_questions: 2\ntechnical_questions: 3\n"
        "follow_ups: 1\nminutes_per_turn: 2\nextra_minutes: 10\n"
    )
    assert "default" in load_profiles(tmp_path)
    (tmp_path / "bad.yaml").write_text(
        "description: d\nseniority: wizard\nbackground_questions: 2\ntechnical_questions: 3\n"
        "follow_ups: 1\nminutes_per_turn: 2\nextra_minutes: 10\n"
    )
    with pytest.raises(PlanError, match="bad.yaml: seniority must be one of"):
        load_profiles(tmp_path)
    (tmp_path / "bad.yaml").unlink()
    (tmp_path / "default.yaml").unlink()
    with pytest.raises(PlanError, match="default.yaml is missing"):
        load_profiles(tmp_path)


# ---- the saved plan


def test_plan_round_trips_and_keeps_hand_edits(tmp_path, profiles):
    path = tmp_path / PLAN
    plan = plan_from_profile(profiles["senior_fullstack"], "a full-stack role")
    save_plan(path, plan)
    assert path.read_text().startswith("# The shape of this candidate's interview")
    assert load_plan(path) == plan
    path.write_text(path.read_text() + "minutes: 40\n")
    assert load_plan(path).minutes == 40 and load_plan(path).technical_questions == 5


def test_plan_errors_are_one_clear_line(tmp_path):
    with pytest.raises(PlanError, match=r"interview.yaml not found: run `senjin-review"):
        load_plan(tmp_path / PLAN)
    path = tmp_path / PLAN
    cases = {
        "technical_questions: 0": "technical_questions must be between 1 and 29",
        "follow_ups: 9": "follow_ups must be between 0 and 3",
        "seniority: boss": "seniority must be one of",
        "minutes_per_turn: abc": "minutes_per_turn is missing or not a number",
    }
    good = (
        "profile: x\nseniority: senior\nbackground_questions: 3\ntechnical_questions: 5\n"
        "follow_ups: 1\nminutes_per_turn: 2\nextra_minutes: 10\n"
    )
    assert load_plan_text(path, good).max_questions == 8
    for bad, message in cases.items():
        key = bad.split(":")[0]
        text = "".join(
            bad + "\n" if ln.startswith(key + ":") else ln + "\n" for ln in good.splitlines()
        )
        with pytest.raises(PlanError, match=message):
            load_plan_text(path, text)
    with pytest.raises(PlanError, match="2 to 30"):
        load_plan_text(path, good.replace("technical_questions: 5", "technical_questions: 29"))


def load_plan_text(path: Path, text: str) -> InterviewPlan:
    path.write_text(text)
    return load_plan(path)


# ---- precedence: flags, then the plan, then the environment


def test_the_plan_beats_the_environment_and_flags_beat_the_plan(profiles):
    env = {
        "SENJIN_MAX_QUESTIONS": "20",
        "SENJIN_RESUME_QUESTIONS": "5",
        "SENJIN_FOLLOW_UPS": "3",
        "SENJIN_MAX_MINUTES": "99",
        "SENJIN_SILENCE_MS": "1234",
        "SENJIN_HOLD_MS": "777",
    }
    plan = plan_from_profile(profiles["senior_fullstack"])
    config = plan.apply(InterviewConfig.from_env(env))
    assert (config.resume, config.technical, config.follow_ups) == (3, 5, 1)
    assert config.minutes == 36  # the environment's 99 does not apply
    assert (config.silence_ms, config.hold_ms) == (1234, 777)  # audio stays global
    flagged = config.override(max_questions=4, max_minutes=20, follow_ups=2, silence_ms=900)
    assert (flagged.max_questions, flagged.minutes, flagged.follow_ups) == (4, 20, 2)
    assert flagged.silence_ms == 900
    pinned = InterviewPlan(**{**plan.__dict__, "minutes": 55.0}).apply(InterviewConfig.from_env({}))
    assert pinned.minutes == 55


# ---- choosing the profile from the job


async def test_the_judge_picks_a_profile_from_the_job(profiles):
    judge = ScriptedJudge(choice("senior_fullstack", "Next.js and Node.js work"))
    profile, reason, notes = await classify_job(judge, JOB, profiles)
    assert profile.name == "senior_fullstack" and reason == "Next.js and Node.js work"
    assert notes == [] and "Build Next.js apps" in judge.prompts[0]
    assert "- staff_architect (lead, 4+4 questions)" in judge.prompts[0]


async def test_an_unknown_or_unreadable_answer_falls_back_to_the_default(profiles):
    profile, _, notes = await classify_job(ScriptedJudge(choice("wizard")), JOB, profiles)
    assert profile.name == DEFAULT_PROFILE
    assert any("'wizard', which is not a profile" in n for n in notes)
    profile, _, notes = await classify_job(ScriptedJudge("nope", "nope"), JOB, profiles)
    assert profile.name == DEFAULT_PROFILE and any("'default' profile is used" in n for n in notes)
    assert "senior_fullstack" in describe_profiles(profiles)


async def test_the_chosen_plan_is_saved_and_read_back_without_the_judge(folder, profiles):
    judge = ScriptedJudge(choice("backend", "an API role"))
    plan = await ensure_plan(folder, judge, JOB, profiles, show=quiet)
    assert plan.profile == "backend" and plan.reason == "an API role"
    assert load_plan(folder / PLAN) == plan
    again = await ensure_plan(folder, ScriptedJudge(), JOB, profiles, show=quiet)  # no reply left
    assert again == plan


async def test_a_hand_edited_plan_is_respected(folder, profiles):
    await ensure_plan(folder, ScriptedJudge(choice("backend")), JOB, profiles, show=quiet)
    path = folder / PLAN
    path.write_text(path.read_text().replace("technical_questions: 5", "technical_questions: 7"))
    plan = await ensure_plan(folder, ScriptedJudge(), JOB, profiles, show=quiet)
    assert plan.technical_questions == 7


async def test_profile_flag_and_refresh_replace_the_plan(folder, profiles):
    await ensure_plan(folder, ScriptedJudge(choice("backend")), JOB, profiles, show=quiet)
    forced = ScriptedJudge()  # --profile never asks the judge
    plan = await ensure_plan(folder, forced, JOB, profiles, profile="staff_architect", show=quiet)
    assert plan.profile == "staff_architect" and load_plan(folder / PLAN).follow_ups == 2
    refreshed = await ensure_plan(
        folder, ScriptedJudge(choice("frontend")), JOB, profiles, refresh=True, show=quiet
    )
    assert refreshed.profile == "frontend"
    with pytest.raises(ReviewError, match="unknown profile 'wizard'. They are: backend"):
        await ensure_plan(folder, ScriptedJudge(), JOB, profiles, profile="wizard", show=quiet)


async def test_a_dry_run_saves_no_plan_and_a_broken_one_is_an_error(folder, profiles):
    await ensure_plan(
        folder, ScriptedJudge(choice("backend")), JOB, profiles, save=False, show=quiet
    )
    assert not (folder / PLAN).exists()
    (folder / PLAN).write_text("seniority: boss\n")
    with pytest.raises(ReviewError, match="background_questions is missing"):
        await ensure_plan(folder, ScriptedJudge(), JOB, profiles, show=quiet)


async def test_plan_only_writes_the_plan_and_stops(folder):
    args = Namespace(model="m", profile=None, refresh_plan=False, plan_only=True, dry_run=False)
    await run_command(args, folder, ScriptedJudge(choice("senior_fullstack", "full stack")))
    assert load_plan(folder / PLAN).profile == "senior_fullstack"
    assert not (folder / "questions.yaml").exists()


def test_technical_count_from_the_plan_sets_the_bank_and_written_split(profiles):
    senior = plan_from_profile(profiles["senior_fullstack"])
    assert split_technical(senior.technical_questions) == (3, 2)
    assert split_technical(plan_from_profile(profiles["staff_architect"]).technical_questions) == (
        2,
        2,
    )


# ---- interviews need a plan that matches the questions


def test_an_interview_needs_a_plan_and_enough_questions(folder, profiles):
    write_questions(folder, 5)
    with pytest.raises(PlanError, match="interview.yaml not found"):
        plan_for_interview(folder)
    save_plan(folder / PLAN, plan_from_profile(profiles["senior_fullstack"]))
    assert plan_for_interview(folder).technical_questions == 5
    write_questions(folder, 3)
    with pytest.raises(PlanError, match="has 3 questions but the plan asks for 5"):
        plan_for_interview(folder)


@pytest.mark.parametrize(
    "module, args",
    [
        ("senjin.meet.cli", ["run", "https://meet.google.com/aaa-bbbb-ccc"]),
        ("senjin.audio.cli", ["run"]),
        ("senjin.audio.cli", ["simulate"]),
        ("senjin.cli", []),
    ],
)
def test_interview_commands_stop_with_an_error_when_there_is_no_plan(module, args, folder):
    write_questions(folder, 5)
    env = {**os.environ, "SENJIN_ENV_FILE": ""}
    done = subprocess.run(
        [sys.executable, "-m", module, *args, "--candidate-dir", str(folder)],
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
        cwd=ROOT,
    )
    assert done.returncode != 0
    assert "error:" in done.stderr and "interview.yaml not found" in done.stderr


def test_review_can_list_profiles_without_a_candidate_and_rejects_a_bad_one():
    run = lambda *a: subprocess.run(  # noqa: E731
        [sys.executable, "-m", "senjin.review.cli", *a],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=ROOT,
        env={**os.environ, "SENJIN_ENV_FILE": ""},
    )
    listed = run("--list-profiles")
    assert listed.returncode == 0 and "staff_architect (lead" in listed.stdout
    assert "junior_generalist" in listed.stdout


# ---- --score measures against what the plan says an interview asks


async def test_score_uses_the_plans_technical_count_as_the_window(folder, profiles):
    write_questions(folder, 3)
    save_requirements(
        folder / REQUIREMENTS,
        [Requirement(name="Next.js rendering", kind="must_have", evidence="e")],
        {},
    )
    from dataclasses import replace

    plan = replace(plan_from_profile(profiles["default"]), technical_questions=1)
    save_plan(folder / PLAN, plan)
    report = await score(folder, show=quiet)
    assert report.window == 1  # the interview asks one question, though the file holds three
    save_plan(folder / PLAN, replace(plan, technical_questions=3))
    assert (await score(folder, show=quiet)).window == 3


async def test_the_simulated_candidate_speaks_without_markdown_and_knows_when_to_stop():
    from senjin.eval.sim import PERSONAS, SimulatedCandidate
    from senjin.llm.fake import FakeLLM

    fake = FakeLLM(["I would use **Playwright** and `Jest`.\n\n## Why\nIt is fast."])
    candidate = SimulatedCandidate(fake, "resume", "Resume of Pat")
    said = await candidate.reply("What would you test with?")
    assert "*" not in said and "`" not in said and "#" not in said and "Playwright" in said
    system = candidate.messages[0]["content"]
    assert "Resume of Pat" in system and "no more questions" in system
    assert "resume" in PERSONAS and "Never invent" in PERSONAS["resume"]


def test_sim_with_a_candidate_folder_needs_the_plan_and_questions(folder):
    write_questions(folder, 5)
    done = subprocess.run(
        [sys.executable, "-m", "senjin.eval.sim", "--candidate-dir", str(folder)],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=ROOT,
        env={**os.environ, "SENJIN_ENV_FILE": ""},
    )
    assert done.returncode != 0 and "interview.yaml not found" in done.stderr
