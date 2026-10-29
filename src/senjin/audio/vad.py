"""Voice activity detection: Silero VAD (ONNX) plus a turn detector that finds utterances."""

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

RATE = 16000
FRAME = 512  # samples per VAD frame (32 ms at 16 kHz)
FRAME_MS = FRAME * 1000 / RATE
_CONTEXT = 64  # Silero v5 expects 64 samples of left context


class SileroVAD:
    """Per-frame speech probability. Stateful: call reset() between streams."""

    def __init__(self, model_path: Path) -> None:
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.inter_op_num_threads = opts.intra_op_num_threads = 1
        self._session = ort.InferenceSession(
            str(model_path), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self._sr = np.array(RATE, dtype=np.int64)
        self.reset()

    def reset(self) -> None:
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, _CONTEXT), dtype=np.float32)

    def __call__(self, frame: np.ndarray) -> float:
        x = np.concatenate([self._context, frame.astype(np.float32)[None]], axis=1)
        out, self._state = self._session.run(
            None, {"input": x, "state": self._state, "sr": self._sr}
        )
        self._context = x[:, -_CONTEXT:]
        return float(out[0, 0])


@dataclass
class TurnConfig:
    threshold: float = 0.5
    start_ms: int = 96  # speech needed before an utterance counts as started
    end_silence_ms: int = 800  # silence that ends an utterance (the blueprint suggests 1500)
    preroll_ms: int = 320  # audio kept from before the start so word onsets are not clipped
    min_utterance_ms: int = 350  # shorter bursts (coughs, clicks) are discarded
    max_utterance_s: float = 60.0


@dataclass
class TurnEvent:
    kind: str  # "start" | "end" | "discard"
    audio: np.ndarray | None = None  # float32 16 kHz, set on "end"


@dataclass
class TurnDetector:
    """Turns a stream of (frame, speech probability) into start/end/discard events."""

    config: TurnConfig = field(default_factory=TurnConfig)

    def __post_init__(self) -> None:
        c = self.config
        self._start_frames = max(1, round(c.start_ms / FRAME_MS))
        self._end_frames = max(1, round(c.end_silence_ms / FRAME_MS))
        self._min_frames = max(1, round(c.min_utterance_ms / FRAME_MS))
        self._max_frames = round(c.max_utterance_s * 1000 / FRAME_MS)
        self._preroll: deque[np.ndarray] = deque(maxlen=max(1, round(c.preroll_ms / FRAME_MS)))
        self.reset()

    def reset(self) -> None:
        self._preroll.clear()
        self._frames: list[np.ndarray] = []
        self._speech_run = 0
        self._silence_run = 0
        self._speech_frames = 0
        self.in_speech = False

    @property
    def speech_ms(self) -> float:
        """Speech accumulated in the current utterance (or the run that may start one)."""
        return (self._speech_frames if self.in_speech else self._speech_run) * FRAME_MS

    def feed(self, frame: np.ndarray, prob: float) -> list[TurnEvent]:
        is_speech = prob >= self.config.threshold
        events: list[TurnEvent] = []
        if not self.in_speech:
            self._preroll.append(frame)
            self._speech_run = self._speech_run + 1 if is_speech else 0
            if self._speech_run >= self._start_frames:
                self.in_speech = True
                self._frames = list(self._preroll)
                self._speech_frames = self._speech_run
                self._silence_run = 0
                events.append(TurnEvent("start"))
            return events

        self._frames.append(frame)
        if is_speech:
            self._speech_frames += 1
            self._silence_run = 0
        else:
            self._silence_run += 1
        if self._silence_run >= self._end_frames or len(self._frames) >= self._max_frames:
            keep = len(self._frames) - max(0, self._silence_run - 6)  # leave ~200 ms of tail
            audio = np.concatenate(self._frames[:keep])
            long_enough = self._speech_frames >= self._min_frames
            self.reset()
            events.append(TurnEvent("end", audio) if long_enough else TurnEvent("discard"))
        return events
