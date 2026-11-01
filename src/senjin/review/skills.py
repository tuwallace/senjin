"""Match job requirements to questions by skill tags, with no model involved.

Every bank question lists what it tests in `skills` (for example `nextjs`, `state management`).
A requirement is covered by a question when all the words of one of its tags appear in the
requirement's name. That is cheap, repeatable and easy to correct by editing a tag, which
a judge's reading of the question text is not. Words are normalised so that "Next.js",
"NextJS" and "next js" meet, and plurals and a few common aliases are folded together.
"""

import re
from collections.abc import Iterable

from senjin.mcp_server.store import Question
from senjin.review.schema import Requirement

_PHRASES = [  # applied to lower-cased text before splitting into words
    (r"next\.?\s?js", "nextjs"),
    (r"node\.?\s?js", "nodejs"),
    (r"react\.?\s?js", "react"),
    (r"ci\s?[/-]?\s?cd", "cicd"),
    (r"no[- ]sql", "nosql"),
    (r"login\.gov", "logingov"),
    (r"type\s?script", "typescript"),
    (r"java\s?script", "javascript"),
    (r"graph\s?ql", "graphql"),
]
_ALIASES = {
    "k8s": "kubernetes",
    "postgresql": "postgres",
    "restful": "rest",
    "a11y": "accessibility",
    "ts": "typescript",
    "js": "javascript",
    "section": "",  # "Section 508" and "508" mean the same
}
_STOP = frozenset("a an and the of for to in on with using use via or as at by".split())
_NO_STEM = frozenset(
    {"nextjs", "nodejs", "logingov", "graphql", "kubernetes", "aws", "wcag", "saas"}
)
# Tags this broad would "cover" almost any requirement that mentions the word (for example
# `api` in "Context API"), so they never match on their own. Say what is specific instead.
GENERIC = frozenset(
    "api compliance security cloud database backend frontend architecture performance testing "
    "deployment devops operations reliability monitoring standards documentation scalability "
    "algorithm data structure system design react".split()
)


def words(text: str) -> list[str]:
    """Normalised words: lower case, phrases and aliases folded, plurals stripped."""
    text = text.lower()
    for pattern, replacement in _PHRASES:
        text = re.sub(pattern, replacement, text)
    out = []
    for raw in re.findall(r"[a-z0-9]+", text):
        word = _ALIASES.get(raw, raw)
        if word.endswith("ies") and len(word) > 4:
            word = word[:-3] + "y"
        elif (
            word.endswith("s")
            and len(word) > 3
            and not word.endswith("ss")
            and word not in _NO_STEM
        ):
            word = word[:-1]
        if word and word not in _STOP:
            out.append(word)
    return out


def tags_match(tags: Iterable[str], requirement: Requirement) -> bool:
    """True if every word of at least one tag appears in the requirement's name."""
    have = set(words(requirement.name))
    for tag in tags:
        w = words(tag)
        if w and not set(w) <= GENERIC and set(w) <= have:
            return True
    return False


def covered_by(question: Question, requirements: list[Requirement]) -> list[Requirement]:
    return [r for r in requirements if tags_match(question.skills, r)]


def weight_of(requirement: Requirement, weights: dict[str, float]) -> float:
    default = 2.0 if requirement.kind == "must_have" else 1.0
    return weights.get(requirement.name, default)


def pick_by_coverage(
    pool: list[Question],
    requirements: list[Requirement],
    weights: dict[str, float],
    count: int,
) -> list[Question]:
    """Greedy choice of `count` questions that cover the most weight not yet covered.

    `pool` is in the judge's order of preference, which breaks ties and, when nothing is left to
    gain (or no question has tags), decides alone. The result is repeatable for the same inputs.
    """
    chosen: list[Question] = []
    covered: set[str] = set()
    remaining = list(pool)
    while remaining and len(chosen) < count:
        scored = []
        for position, question in enumerate(remaining):
            new = [r for r in covered_by(question, requirements) if r.name not in covered]
            scored.append((sum(weight_of(r, weights) for r in new), -position, question))
        best = max(scored, key=lambda item: item[:2])[2]
        chosen.append(best)
        remaining.remove(best)
        covered |= {r.name for r in covered_by(best, requirements)}
    return chosen
