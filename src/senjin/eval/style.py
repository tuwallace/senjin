"""Interviewer-style metrics: are the questions single, open-ended and tied to the last answer?

Usage:
  uv run python -m senjin.eval.style                      # dataset baseline
  uv run python -m senjin.eval.style data/sessions/a.jsonl data/sessions/b.jsonl
Heuristic and deterministic on purpose, so before/after comparisons are reproducible.
"""

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

from senjin.eval.dataset import SPLITS, extract_moves, load_split

_OPEN = re.compile(
    r"\b(what|how|why|tell me|walk me|describe|explain|in what ways?|talk (me )?through)\b", re.I
)
_STOP = {
    "about", "which", "their", "there", "would", "could", "should", "these", "those", "being",
    "other", "where", "while", "think", "really", "something", "things", "thing", "going",
    "using", "after", "before", "because", "again", "every", "might", "still", "first",
}  # fmt: skip
_MARKUP = re.compile(r"(\*\*|^#+ |^\s*[-*] |^\s*\d+\. |`)", re.M)


@dataclass
class Style:
    n: int
    single_question: float
    open_ended: float
    ties_to_answer: float
    speakable: float
    mean_words: float

    def row(self, label: str) -> str:
        return (
            f"{label:34} n={self.n:5} single-q {self.single_question:5.0%}  "
            f"open {self.open_ended:5.0%}  ties-to-answer {self.ties_to_answer:5.0%}  "
            f"speakable {self.speakable:5.0%}  words {self.mean_words:5.1f}"
        )


def _last_question(text: str) -> str:
    sentences = re.findall(r"[^.?!]*\?", text)
    return sentences[-1] if sentences else ""


def _content_words(text: str) -> set[str]:
    # 5-letter prefixes act as a crude stem (migration/migrated)
    return {w[:5] for w in re.findall(r"[a-z]{5,}", text.lower()) if w not in _STOP}


def measure(pairs: list[tuple[str, str]]) -> Style:
    """pairs: (interviewer_text, preceding_participant_text)."""
    if not pairs:
        return Style(0, 0, 0, 0, 0, 0)
    single = [t.count("?") == 1 for t, _ in pairs]
    open_ = [bool(_OPEN.search(_last_question(t))) for t, _ in pairs]
    ties = [bool(_content_words(t) & _content_words(c)) for t, c in pairs]
    speak = [not _MARKUP.search(t) for t, _ in pairs]
    return Style(
        len(pairs), mean(single), mean(open_), mean(ties), mean(speak),
        mean(len(t.split()) for t, _ in pairs),
    )  # fmt: skip


def dataset_pairs(cache_dir: Path, kinds=("probe", "clarify", "reflect")) -> list[tuple[str, str]]:
    pairs = []
    for split in SPLITS:
        for tid, turns in load_split(cache_dir, split):
            pairs += [
                (m.text, m.context) for m in extract_moves(tid, split, turns) if m.kind in kinds
            ]
    return pairs


def session_pairs(path: Path, phases=("resume_dive", "technical")) -> list[tuple[str, str]]:
    from senjin.assess.judge import load_rows

    pairs, last_user = [], ""
    for row in load_rows(path):
        if row["role"] == "user":
            last_user = row["text"]
        elif row["role"] == "assistant" and row["phase"] in phases and last_user:
            pairs.append((row["text"], last_user))
    return pairs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sessions", nargs="*", type=Path)
    parser.add_argument("--cache", type=Path, default=Path("data/hf_cache"))
    args = parser.parse_args()
    print(measure(dataset_pairs(args.cache)).row("dataset (mid-interview moves)"))
    for path in args.sessions:
        print(measure(session_pairs(path)).row(path.name))
    return 0


if __name__ == "__main__":
    sys.exit(main())
