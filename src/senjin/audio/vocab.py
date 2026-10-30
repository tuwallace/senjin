"""A short list of the names and terms in the resume and job description.

Speech recognition mishears names ("Paratus" became "Parada"). Giving it the words to expect
as a prompt makes it spell them the way the documents do.
"""

import re

_WORD = re.compile(r"\b[A-Z][A-Za-z0-9.+#]*[A-Za-z0-9+#]\b")
_STARTS = ".!?:\n-•#|>("  # a capital after these may just start a sentence or bullet


def _distinctive(word: str) -> bool:
    """Acronyms, CamelCase and dotted names are terms wherever they appear."""
    return word.isupper() or bool(re.search(r"[a-z][A-Z]|[.0-9+#]", word))


def vocabulary_hint(text: str, limit: int = 60) -> str:
    """Names, products and acronyms: capitalised mid-sentence and never in lower case."""
    lowercase = set(re.findall(r"\b[a-z][a-z0-9]*\b", text))
    counts: dict[str, int] = {}
    for match in _WORD.finditer(text):
        word = match.group()
        if word.lower() in lowercase or len(word) < 3:
            continue
        before = text[: match.start()].rstrip(" \t*_`")
        starts_sentence = not before or before[-1] in _STARTS
        names_a_place_or_firm = re.match(r"[\s*_`]*[A-Z][a-z]", text[match.end() :]) is not None
        if starts_sentence and not (_distinctive(word) or names_a_place_or_firm):
            continue
        counts[word] = counts.get(word, 0) + 1
    first_seen = {w: i for i, w in enumerate(counts)}  # the resume comes first: names before skills
    words = sorted(counts, key=lambda w: (-counts[w], first_seen[w]))[:limit]
    return ("Interview about " + ", ".join(words) + ".") if words else ""
