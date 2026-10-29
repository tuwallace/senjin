"""Crash-safe JSONL transcript: one flushed line per turn."""

import json
import time
from pathlib import Path


class TranscriptLog:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, role: str, text: str, phase: str) -> None:
        entry = {"ts": time.time(), "role": role, "phase": phase, "text": text}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    @staticmethod
    def load(path: Path) -> list[dict]:
        with path.open(encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
