"""Remove praise from the start of a reply.

The prompt forbids praise but a small model still opens with "Great!" or "Thank you for your
detailed explanation". Interviewers must stay neutral, so the first sentence is checked in code.
"""

import re

_PRAISE = (
    r"great|excellent|awesome|perfect|wonderful|fantastic|brilliant|terrific|fascinating|"
    r"impressive|insightful|thoughtful|thorough|detailed|comprehensive|clear|helpful|"
    r"informative|nice|good|solid|interesting"
)
# A whole opening sentence that is praise: "Great insights.", "That's a really good point."
_PRAISE_SENTENCE = re.compile(
    rf"^\s*(?:(?:that'?s|that is|what|this is)\s+)?(?:an?\s+)?(?:really\s+|very\s+|truly\s+)?"
    rf"(?:{_PRAISE})\b(?!\s+(?:morning|afternoon|evening|to\s+(?:meet|hear|know)))[^.!?]*[.!?]+\s*",
    re.IGNORECASE,
)
_THANKS_FOR_PRAISE = re.compile(
    rf"^\s*(?:thank you|thanks|i appreciate)\s+for\s+(?:your|the|sharing)\s+(?:[a-z]+\s+)?"
    rf"(?:{_PRAISE})\b[^.!?]*[.!?]+\s*",
    re.IGNORECASE,
)
# A praise word that only prefaces the real sentence: "Great, let's start..."
_PRAISE_WORD = re.compile(
    r"^\s*(?:great|excellent|awesome|perfect|wonderful|fantastic|brilliant)\s*[,!.]\s+",
    re.IGNORECASE,
)


def _remove(text: str, *, first: bool) -> str:
    patterns = (
        (_THANKS_FOR_PRAISE, _PRAISE_WORD, _PRAISE_SENTENCE)
        if first
        else (
            _THANKS_FOR_PRAISE,
            _PRAISE_SENTENCE,
        )
    )
    cleaned = text
    for pattern in patterns:
        cleaned = pattern.sub("", cleaned, count=1)
    cleaned = cleaned.lstrip()
    return cleaned[:1].upper() + cleaned[1:] if cleaned != text.lstrip() else text


def strip_praise(text: str) -> str:
    """`text` without praise in its opening sentence; unchanged if nothing else would remain."""
    cleaned = _remove(text, first=True)
    return cleaned if cleaned.strip() else text


def first_sentence_end(text: str) -> int | None:
    """Index just past the first sentence end that is followed by more text, else None."""
    match = re.search(r"[.!?]+\s+\S", text)
    return match.end() - 1 if match else None


def plain_speech(text: str) -> str:
    """Drop markdown marks (**bold**, `code`, # headings) that a voice would read out loud."""
    text = re.sub(r"[*`]+", "", text)
    return re.sub(r"(?m)^[ \t]*#+[ \t]*", "", text)


class PraiseFilter:
    """Streams a reply sentence by sentence with praise sentences and markdown removed.

    Never leaves a reply silent: if everything would be removed, the original is spoken.
    """

    MAX_HELD = 240  # a sentence this long without an end is released as it is

    def __init__(self) -> None:
        self._held = ""
        self._first = True
        self._raw = ""
        self._emitted = False

    def feed(self, chunk: str) -> str:
        self._held += chunk
        self._raw += chunk
        out = []
        while (end := first_sentence_end(self._held)) is not None:
            sentence, self._held = self._held[:end], self._held[end:]
            out.append(self._clean(sentence))
        if len(self._held) > self.MAX_HELD:
            out.append(self._clean(self._held))
            self._held = ""
        return "".join(out)

    def flush(self) -> str:
        rest, self._held = self._held, ""
        out = self._clean(rest) if rest else ""
        if not self._emitted and not out.strip():
            out = plain_speech(self._raw) if self._raw.strip() else ""
            self._emitted = bool(out)
        return out

    def _clean(self, sentence: str) -> str:
        cleaned = _remove(plain_speech(sentence), first=self._first)
        self._first = False
        if cleaned.strip():
            self._emitted = True
            return cleaned
        return ""
