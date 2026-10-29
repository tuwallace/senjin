"""Style exemplars from the Anthropic Interviewer dataset, retrieved per turn.

Built once with `senjin-data moves`, stored as JSONL in data/exemplars (git-ignored).
Exemplars show manner (acknowledge, probe the specific thing said, one open question),
never the research topic.
"""

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import ollama

from senjin.eval.dataset import SPLITS, Move, extract_moves, load_split
from senjin.eval.style import _OPEN, _last_question
from senjin.interview.state import Phase

EMBED_MODEL = "nomic-embed-text"
PROBING_KINDS = {"probe", "clarify", "reflect"}
PHASE_KINDS: dict[Phase, set[str]] = {
    Phase.RESUME_DIVE: PROBING_KINDS,
    Phase.TECHNICAL: PROBING_KINDS,
}


class ExemplarProvider(Protocol):
    async def examples(self, candidate_text: str, phase: Phase) -> list[str]: ...


@dataclass
class Exemplar:
    kind: str
    context: str
    text: str
    embedding: list[float]

    def render(self) -> str:
        return f'Participant: "{self.context}"\nInterviewer: "{self.text}"'


def good_exemplar(m: Move) -> bool:
    """Craft filter: one open question, concise, responding to a real answer."""
    words = len(m.text.split())
    return (
        m.kind in PROBING_KINDS
        and m.text.count("?") == 1
        and 8 <= words <= 45
        and 6 <= len(m.context.split()) <= 120
        and bool(_OPEN.search(_last_question(m.text)))
    )


def select_moves(cache_dir: Path, per_kind: int, seed: int = 7) -> list[Move]:
    pools: dict[str, list[Move]] = {k: [] for k in PROBING_KINDS}
    for split in SPLITS:
        for tid, turns in load_split(cache_dir, split):
            for m in extract_moves(tid, split, turns):
                if good_exemplar(m):
                    pools[m.kind].append(m)
    rng = random.Random(seed)
    chosen = []
    for kind in sorted(pools):
        pool = pools[kind]
        rng.shuffle(pool)
        chosen += pool[:per_kind]
    return chosen


def embed_texts(texts: list[str], host: str = "http://localhost:11434") -> list[list[float]]:
    client = ollama.Client(host=host)
    out: list[list[float]] = []
    for i in range(0, len(texts), 64):
        out += [
            list(v) for v in client.embed(model=EMBED_MODEL, input=texts[i : i + 64]).embeddings
        ]
    return out


def build_index(cache_dir: Path, out_path: Path, per_kind: int = 40) -> int:
    moves = select_moves(cache_dir, per_kind)
    vectors = embed_texts([m.context for m in moves])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for m, v in zip(moves, vectors, strict=True):
            row = {"kind": m.kind, "context": m.context, "text": m.text, "embedding": v}
            f.write(json.dumps(row) + "\n")
    return len(moves)


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


class ExemplarIndex:
    """Nearest exemplars by similarity between the candidate's answer and the exemplar context."""

    def __init__(self, exemplars: list[Exemplar], k: int = 2, embed=embed_texts) -> None:
        self.exemplars = exemplars
        self.k = k
        self._embed = embed

    @classmethod
    def load(cls, path: Path, k: int = 2) -> "ExemplarIndex":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        return cls([Exemplar(**r) for r in rows], k)

    async def examples(self, candidate_text: str, phase: Phase) -> list[str]:
        kinds = PHASE_KINDS.get(phase)
        if not kinds or not self.exemplars:
            return []
        pool = [e for e in self.exemplars if e.kind in kinds]
        query = self._embed([candidate_text])[0]
        pool.sort(key=lambda e: _cosine(query, e.embedding), reverse=True)
        return [e.render() for e in pool[: self.k]]
