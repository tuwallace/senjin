"""A second opinion from a local decision model (Ollama `/v1/systemone`, e.g. `nimble`).

Used only where a rule can be wrong and a wrong call is cheap to check. Any failure (model not
installed, slow, bad reply) returns None and the caller keeps the rule's answer.
"""

import asyncio
import json
import urllib.request
from typing import Protocol

from senjin.llm.client import DEFAULT_HOST

TIMEOUT_S = 3.0


class Decider(Protocol):
    async def probability(
        self, name: str, state: dict[str, str], instructions: str
    ) -> float | None:
        """Chance (0 to 1) that the yes/no question is true, or None. The model reads `name` too,
        so it should say what is asked ("done"), not "q"."""
        ...


class OllamaDecider:
    def __init__(self, model: str, host: str = DEFAULT_HOST, timeout: float = TIMEOUT_S) -> None:
        self.model = model
        self.url = host.rstrip("/") + "/v1/systemone"
        self.timeout = timeout

    def _call(self, name: str, state: dict[str, str], instructions: str) -> float | None:
        body = json.dumps(
            {
                "model": self.model,
                "state": state,
                "questions": {name: {"type": "noul", "instructions": instructions}},
            }
        ).encode()
        try:
            with urllib.request.urlopen(
                urllib.request.Request(self.url, body), timeout=self.timeout
            ) as response:
                value = json.load(response)["answers"][name]["noul"]
            return float(value) if 0.0 <= float(value) <= 1.0 else None
        except (OSError, ValueError, KeyError, TypeError):
            return None

    async def probability(
        self, name: str, state: dict[str, str], instructions: str
    ) -> float | None:
        return await asyncio.to_thread(self._call, name, state, instructions)


def decider_for(model: str) -> Decider | None:
    """None when no decision model is configured: the rules decide alone."""
    return OllamaDecider(model) if model.strip() else None
