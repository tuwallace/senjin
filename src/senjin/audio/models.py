"""Model files for the voice loop. Downloaded once into data/models (git-ignored)."""

import urllib.request
from pathlib import Path

MODELS_DIR = Path("data/models")
WHISPER_DIR = Path.home() / ".cache/whisper-models"  # where whisper.cpp models already live

SILERO_URL = "https://github.com/snakers4/silero-vad/raw/v5.1.2/src/silero_vad/data/silero_vad.onnx"
PIPER_BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/"
PIPER_VOICE = "en_US-lessac-medium"

SILERO_PATH = MODELS_DIR / "silero_vad.onnx"
PIPER_PATH = MODELS_DIR / f"{PIPER_VOICE}.onnx"


def _download(url: str, dest: Path) -> None:
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(dest)


def ensure_models() -> list[Path]:
    _download(SILERO_URL, SILERO_PATH)
    _download(PIPER_BASE + f"{PIPER_VOICE}.onnx", PIPER_PATH)
    _download(PIPER_BASE + f"{PIPER_VOICE}.onnx.json", Path(str(PIPER_PATH) + ".json"))
    return [SILERO_PATH, PIPER_PATH]
