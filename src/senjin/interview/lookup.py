"""Find the job description lines that bear on a candidate's question.

A small model reading a long description often says "not specified" about a fact that is
written in it. Handing it the matching lines in the turn note fixes that without another call.
"""

import re

_STOPWORDS = frozenset(
    "a an and are as at be been but by can could did do does for from get give had has have how i "
    "if in is it its me my of on or our please should so tell that the their them there these this "
    "to us was we were what whats when where which who why will with would you your about any also "
    "based role job position company team".split()
)


def _words(text: str) -> set[str]:
    return {
        w[:-1] if len(w) > 3 and w.endswith("s") else w
        for w in re.findall(r"[a-z0-9]+", text.lower())
        if len(w) > 1 and w not in _STOPWORDS
    }


def relevant_lines(document: str, question: str, limit: int = 3, max_chars: int = 280) -> list[str]:
    """Up to `limit` lines of `document` sharing the most distinctive words with `question`."""
    wanted = _words(question)
    scored: list[tuple[int, int, str]] = []
    for index, raw in enumerate(document.splitlines()):
        line = re.sub(r"[*_#>`]|^\s*[-•]\s*", "", raw).strip()
        if len(line) < 8:
            continue
        hits = len(wanted & _words(line))
        if hits:
            scored.append((-hits, index, line[:max_chars]))
    return [line for _, _, line in sorted(scored)[:limit]]
