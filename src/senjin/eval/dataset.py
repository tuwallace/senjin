"""Anthropic/AnthropicInterviewer: loading, turn parsing and interviewer-move extraction.

The dataset is 1,250 qualitative research interviews (not technical interviews). We use it
for interviewer craft only. Data lands in data/hf_cache (git-ignored); license and attribution
are in the README.
"""

import csv
import re
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

SPLITS = ("workforce", "creatives", "scientists")
URL = (
    "https://huggingface.co/datasets/Anthropic/AnthropicInterviewer/resolve/main/"
    "interview_transcripts/{split}_transcripts.csv"
)
_LABEL = re.compile(r"^(Assistant|AI|User):[ \t]*", re.M)

MOVE_KINDS = ("opener", "probe", "clarify", "reflect", "transition", "wrap_up")


@dataclass(frozen=True)
class Turn:
    role: str  # "interviewer" or "participant"
    text: str


@dataclass(frozen=True)
class Move:
    transcript_id: str
    split: str
    kind: str
    text: str
    context: str  # the participant turn this move responds to ("" for openers)


def fetch(cache_dir: Path) -> list[Path]:
    """Download the three CSVs if missing."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for split in SPLITS:
        path = cache_dir / f"{split}_transcripts.csv"
        if not path.exists():
            urllib.request.urlretrieve(URL.format(split=split), path)
        paths.append(path)
    return paths


def parse_turns(text: str) -> list[Turn]:
    parts = _LABEL.split(text)
    return [
        Turn("participant" if label == "User" else "interviewer", body.strip())
        for label, body in zip(parts[1::2], parts[2::2], strict=True)
        if body.strip()
    ]


def load_split(cache_dir: Path, split: str) -> Iterator[tuple[str, list[Turn]]]:
    csv.field_size_limit(10**9)
    with (cache_dir / f"{split}_transcripts.csv").open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            yield row["transcript_id"], parse_turns(row["text"])


_CLARIFY = re.compile(
    r"\b(what do you mean|do you mean|when you say|could you clarify|can you clarify|"
    r"just to (make sure|clarify)|help me understand)\b",
    re.I,
)
_REFLECT = re.compile(
    r"^(it sounds like|it seems|so you|if i understand|i hear|that makes sense|"
    r"that sounds|it's interesting|it is interesting|interesting)",
    re.I,
)
_TRANSITION = re.compile(
    r"\b(let's (move|turn|shift|talk about)|shifting|turning to|moving on|now,? i'd like|"
    r"switching gears|i'd like to (ask|explore|turn)|on a different note)\b",
    re.I,
)


def classify_move(text: str, ordinal: int, is_last: bool) -> str:
    """Heuristic move type for one interviewer turn."""
    if ordinal == 0:
        return "opener"
    if is_last:
        return "wrap_up"
    if _CLARIFY.search(text):
        return "clarify"
    if _TRANSITION.search(text):
        return "transition"
    if _REFLECT.match(text.strip()):
        return "reflect"
    return "probe"


def extract_moves(transcript_id: str, split: str, turns: list[Turn]) -> list[Move]:
    interviewer = [i for i, t in enumerate(turns) if t.role == "interviewer"]
    moves = []
    for ordinal, i in enumerate(interviewer):
        context = ""
        if i > 0 and turns[i - 1].role == "participant":
            context = turns[i - 1].text
        kind = classify_move(turns[i].text, ordinal, ordinal == len(interviewer) - 1)
        moves.append(Move(transcript_id, split, kind, turns[i].text, context))
    return moves
