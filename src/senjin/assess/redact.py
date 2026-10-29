"""Best-effort removal of personal and demographic cues before the judge reads a transcript.

Names and gendered pronouns are masked; sentences that mention family, age, origin, identity
or similar are dropped whole. This reduces, but does not eliminate, the chance that the judge
scores on who the candidate is rather than what they said. The lists below are not complete;
`senjin.eval.fairness` measures what is left.
"""

import re
from collections.abc import Iterable
from pathlib import Path

_I = re.IGNORECASE

_DROP_SENTENCE = [
    re.compile(
        r"\b(my|our) (wife|husband|spouse|partner|girlfriend|boyfriend|kids?|children|son|"
        r"daughter|mother|father|mom|dad|parents|family|baby)\b",
        _I,
    ),
    re.compile(r"\b(mother|father|mom|dad|parent)s? of\b", _I),
    re.compile(r"\b\d{2}[- ]years?[- ]old\b", _I),
    re.compile(r"\bI(?:'m| am) (?:in my )?\d{2}s?\b(?!\s*(?:%|percent|years? of|years? in))", _I),
    re.compile(
        r"\b(originally from|I(?:'m| am) from|born in|grew up in|my accent|"
        r"native (?:speaker|language)|immigrant|visa|citizen(?:ship)?)\b",
        _I,
    ),
    re.compile(
        r"\bI(?:'m| am) (?:a |an )?(?:woman|man|female|male|black|white|asian|latino|latina|"
        r"hispanic|gay|lesbian|trans\w*|christian|muslim|jewish|hindu|buddhist|disabled|"
        r"veteran|pregnant|single)\b",
        _I,
    ),
    re.compile(r"\b(pregnan\w*|maternity|paternity|parental leave|retire\w*|disabilit\w*)\b", _I),
]
_NAME_INTRO = re.compile(
    r"\b(my name is|my name's|call me|this is)\s+[A-Z][\w'-]+(?:\s+[A-Z][\w'-]+)?", re.UNICODE
)
_PRONOUNS = {
    "he": "they", "she": "they", "him": "them", "her": "them", "his": "their",
    "hers": "theirs", "himself": "themself", "herself": "themself",
    "mr": "", "mrs": "", "ms": "", "miss": "", "sir": "", "madam": "",
}  # fmt: skip
_PRONOUN_RE = re.compile(r"\b(" + "|".join(_PRONOUNS) + r")\b\.?", _I)
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def _pronoun(match: re.Match) -> str:
    word = match.group(1)
    new = _PRONOUNS[word.lower()]
    if new and word[0].isupper():
        new = new.capitalize()
    return new


def redact(text: str, names: Iterable[str] = ()) -> str:
    kept = [s for s in _SENTENCE_END.split(text) if not any(p.search(s) for p in _DROP_SENTENCE)]
    out = " ".join(kept)
    out = _NAME_INTRO.sub(lambda m: m.group(1) + " [name]", out)
    for name in sorted(
        {n for n in names for n in [n, *n.split()] if len(n) >= 3}, key=len, reverse=True
    ):
        out = re.sub(rf"\b{re.escape(name)}\b", "[name]", out, flags=_I)
    out = _PRONOUN_RE.sub(_pronoun, out)
    return re.sub(r"\s+", " ", out).strip()


def redact_rows(rows: list[dict], names: Iterable[str] = ()) -> list[dict]:
    names = list(names)
    result = []
    for row in rows:
        if row["role"] in {"user", "assistant"}:
            row = {**row, "text": redact(row["text"], names)}
        result.append(row)
    return result


def candidate_names(resume_text: str) -> list[str]:
    """Name from the resume's first heading, e.g. '# Jane Doe (sample candidate)'."""
    first = resume_text.strip().splitlines()[0] if resume_text.strip() else ""
    name = re.sub(r"\(.*?\)", "", first.lstrip("# ")).strip()
    return [name] if name else []


def names_from_dir(candidate_dir: Path) -> list[str]:
    resume = candidate_dir / "resume.md"
    return candidate_names(resume.read_text(encoding="utf-8")) if resume.exists() else []
