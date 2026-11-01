"""Interview context behind the MCP tools. Plain Python so it is testable without MCP."""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
GENERATED_BANK = "questions.yaml"  # written by senjin-review into the candidate folder


@dataclass
class Question:
    id: str
    topic: str
    difficulty: str
    prompt: str
    follow_ups: list[str] = field(default_factory=list)
    rubric: list[str] = field(default_factory=list)
    skills: list[str] = field(default_factory=list)  # what it tests, for matching job requirements
    basis: str = ""  # why this question was chosen; for the human reviewer only
    origin: str = ""  # where it came from: a question_bank file or "generated"; reviewer only

    def public(self) -> dict:
        """What the interviewer may see; rubric, skills, basis and origin are for reviewers only."""
        return {
            "id": self.id,
            "topic": self.topic,
            "difficulty": self.difficulty,
            "prompt": self.prompt,
            "follow_ups": self.follow_ups,
        }


def _check_text_lists(item: dict, path: Path) -> None:
    """follow_ups and rubric must be lists of text. An unquoted `Name: value` line is parsed by
    YAML as a mapping, which would reach the judge and the interviewer as {'Name': 'value'}."""
    for key in ("follow_ups", "rubric", "skills"):
        bad = [v for v in item.get(key) or [] if not isinstance(v, str)]
        if bad:
            raise ValueError(
                f"{path.name}: {item.get('id', '?')} {key} has a non-text item {bad[0]!r}; "
                "put the whole line in quotes"
            )


def load_questions(bank_dir: Path, sets: list[str] | None = None) -> list[Question]:
    """Every question in the bank, or only the files named in `sets` (file names without .yaml)."""
    questions: list[Question] = []
    for path in sorted(bank_dir.glob("*.yaml")):
        if sets and path.stem not in sets:
            continue
        for item in yaml.safe_load(path.read_text()) or []:
            _check_text_lists(item, path)
            questions.append(Question(**item))
    ids = [q.id for q in questions]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate question ids in question bank")
    return questions


def bank_label(candidate_dir: Path) -> str:
    """One line for the log: which questions this interview will ask."""
    path = candidate_dir / GENERATED_BANK
    return f"Question bank: {len(load_questions(path.parent, [path.stem]))} questions ({path})"


def all_questions(bank_dir: Path, candidate_dir: Path) -> list[Question]:
    """The shared bank plus the candidate's generated questions, for looking up rubrics."""
    questions = load_questions(bank_dir)
    generated = candidate_dir / GENERATED_BANK
    if generated.is_file():
        known = {q.id for q in questions}
        questions += [
            q for q in load_questions(generated.parent, [generated.stem]) if q.id not in known
        ]
    return questions


class InterviewStore:
    """One store per interview: it remembers which questions were already asked."""

    def __init__(
        self,
        candidate_dir: Path,
        bank_dir: Path,
        notes_path: Path | None = None,
        question_sets: list[str] | None = None,
    ) -> None:
        self.candidate_dir = candidate_dir
        self.questions = load_questions(bank_dir, question_sets)
        if not self.questions:
            raise ValueError(f"no questions in {bank_dir} for sets {question_sets}")
        self.notes_path = notes_path
        self.asked: list[str] = []

    def resume(self) -> str:
        return self._read("resume.md")

    def job_description(self) -> str:
        return self._read("job_description.md")

    def next_question(self, topic: str | None = None, difficulty: str | None = None) -> dict:
        """First unasked question matching the filters; marks it asked."""
        for q in self.questions:
            if q.id in self.asked:
                continue
            if topic and q.topic != topic:
                continue
            if difficulty and q.difficulty != difficulty:
                continue
            self.asked.append(q.id)
            return q.public()
        remaining = [q for q in self.questions if q.id not in self.asked]
        return {
            "exhausted": not remaining,
            "message": "No unasked question matches those filters.",
            "available_topics": sorted({q.topic for q in remaining}),
        }

    def rubric(self, question_id: str) -> list[str]:
        return next(q.rubric for q in self.questions if q.id == question_id)

    def record_note(self, text: str) -> str:
        if self.notes_path:
            self.notes_path.parent.mkdir(parents=True, exist_ok=True)
            with self.notes_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"ts": time.time(), "note": text}) + "\n")
        return "noted"

    def _read(self, name: str) -> str:
        path = self.candidate_dir / name
        return path.read_text(encoding="utf-8") if path.exists() else f"({name} not provided)"
