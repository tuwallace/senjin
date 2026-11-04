"""Build a mock video call from a real Senjin transcript.

Senjin speaks with its own Piper voice (same pace as the live bot). The candidate speaks with a
macOS voice. Each caption chunk becomes one frame (a call-style layout drawn in HTML and
screenshotted with Playwright) joined with the audio by ffmpeg. Nothing is uploaded.

    uv run python docs/video/mock/make_mock_meeting.py data/sessions/meet-20261103-132204.jsonl
"""

import argparse
import html
import json
import re
import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np

CANDIDATE_VOICE = "Samantha"
CHUNK_WORDS = 24
TEMPLATE = """<!doctype html><meta charset=utf-8><style>
*{{box-sizing:border-box;margin:0}} body{{width:1920px;height:1080px;background:#202124;color:#e8eaed;
font-family:-apple-system,Helvetica,Arial,sans-serif;position:relative;overflow:hidden}}
.top{{position:absolute;top:28px;left:40px;font-size:26px;color:#bdc1c6}}
.mock{{position:absolute;top:22px;right:40px;font-size:22px;color:#fdd663;background:#3c3a2a;
padding:8px 18px;border-radius:20px}}
.tiles{{position:absolute;top:100px;left:40px;right:40px;height:600px;display:flex;gap:24px}}
.tile{{flex:1;background:#3c4043;border-radius:20px;position:relative;display:flex;
align-items:center;justify-content:center;border:6px solid transparent}}
.tile.on{{border-color:#8ab4f8;box-shadow:0 0 40px #8ab4f866}}
.av{{width:260px;height:260px;border-radius:50%;display:flex;align-items:center;justify-content:center;
font-size:120px;font-weight:500;color:#fff}}
.name{{position:absolute;left:24px;bottom:20px;font-size:28px;background:#0008;padding:6px 16px;border-radius:10px}}
.bars{{position:absolute;right:24px;top:24px;display:flex;gap:5px;align-items:flex-end;height:44px;opacity:0}}
.on .bars{{opacity:1}} .bars i{{width:8px;background:#8ab4f8;border-radius:4px;display:block}}
.cap{{position:absolute;left:160px;right:160px;bottom:150px;text-align:center;font-size:40px;line-height:1.35;
background:#000b;padding:18px 30px;border-radius:14px;min-height:120px}}
.cap b{{color:#8ab4f8;font-size:28px;display:block;margin-bottom:4px}}
.bar{{position:absolute;bottom:28px;left:0;right:0;display:flex;justify-content:center;gap:22px}}
.btn{{width:76px;height:76px;border-radius:50%;background:#3c4043;display:flex;align-items:center;
justify-content:center;font-size:34px}} .end{{background:#ea4335;width:110px;border-radius:40px}}
</style>
<div class=top>Technical interview · {clock}</div>
<div class=mock>Mock recreation · synthetic voices · from a real transcript</div>
<div class=tiles>
 <div class="tile {s_on}"><div class=av style="background:#1a73e8">S</div>
  <div class=name>Senjin (AI interviewer)</div><div class=bars><i style="height:18px"></i><i style="height:40px"></i><i style="height:26px"></i></div></div>
 <div class="tile {c_on}"><div class=av style="background:#188038">C</div>
  <div class=name>Candidate</div><div class=bars><i style="height:30px"></i><i style="height:16px"></i><i style="height:40px"></i></div></div>
</div>
<div class=cap><b>{who}</b>{text}</div>
<div class=bar><div class=btn>🎤</div><div class=btn>📷</div><div class=btn>💬</div><div class="btn end">📞</div></div>"""


def chunks(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.?!])\s+", text.strip())
    out, cur = [], []
    for s in sentences:
        if cur and len(" ".join(cur + [s]).split()) > CHUNK_WORDS:
            out.append(" ".join(cur))
            cur = []
        cur.append(s)
    if cur:
        out.append(" ".join(cur))
    return out


def write_wav(path: Path, samples: np.ndarray, rate: int) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(samples.astype(np.int16).tobytes())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("transcript", type=Path)
    ap.add_argument("--out", type=Path, default=Path("data/video/mock-meeting.mp4"))
    ap.add_argument("--max-turns", type=int, default=0, help="0 = whole interview")
    ap.add_argument("--candidate-voice", default=CANDIDATE_VOICE)
    args = ap.parse_args()

    from playwright.sync_api import sync_playwright

    from senjin.audio.tts import PiperTTS
    from senjin.config import InterviewConfig

    pace = InterviewConfig.from_env().voice_pace
    tts = PiperTTS(pace=pace)
    rows = [json.loads(line) for line in args.transcript.read_text().splitlines()]
    turns = [(r["role"], r["text"]) for r in rows if r["role"] in ("assistant", "user")]
    if args.max_turns:
        turns = turns[: args.max_turns]

    work = Path(tempfile.mkdtemp(prefix="mockmeet-"))
    segs: list[Path] = []
    elapsed = 0.0
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome")
        page = browser.new_page(viewport={"width": 1920, "height": 1080})
        for ti, (role, text) in enumerate(turns):
            senjin = role == "assistant"
            for ci, chunk in enumerate(chunks(text)):
                n = f"{ti:03d}-{ci:02d}"
                wav = work / f"{n}.wav"
                if senjin:
                    audio = tts._synth(chunk)  # sync; Playwright's sync API owns the event loop
                    pad = np.zeros(int(tts.sample_rate * 0.25), dtype=np.int16)
                    write_wav(wav, np.concatenate([audio, pad]), tts.sample_rate)
                else:
                    aiff = work / f"{n}.aiff"
                    subprocess.run(
                        ["say", "-v", args.candidate_voice, "-r", "175", "-o", str(aiff), chunk],
                        check=True,
                    )
                    subprocess.run(
                        [
                            "ffmpeg",
                            "-y",
                            "-loglevel",
                            "error",
                            "-i",
                            str(aiff),
                            "-ar",
                            "22050",
                            "-ac",
                            "1",
                            "-af",
                            "apad=pad_dur=0.25",
                            str(wav),
                        ],
                        check=True,
                    )
                clock = f"{int(elapsed // 60):02d}:{int(elapsed % 60):02d}"
                page.set_content(
                    TEMPLATE.format(
                        clock=clock,
                        s_on="on" if senjin else "",
                        c_on="" if senjin else "on",
                        who="Senjin" if senjin else "Candidate",
                        text=html.escape(chunk),
                    )
                )
                png = work / f"{n}.png"
                page.screenshot(path=str(png))
                seg = work / f"{n}.mp4"
                subprocess.run(
                    [
                        "ffmpeg",
                        "-y",
                        "-loglevel",
                        "error",
                        "-loop",
                        "1",
                        "-framerate",
                        "10",
                        "-i",
                        str(png),
                        "-i",
                        str(wav),
                        "-c:v",
                        "libx264",
                        "-tune",
                        "stillimage",
                        "-pix_fmt",
                        "yuv420p",
                        "-c:a",
                        "aac",
                        "-ar",
                        "44100",
                        "-shortest",
                        str(seg),
                    ],
                    check=True,
                )
                with wave.open(str(wav)) as w:
                    elapsed += w.getnframes() / w.getframerate()
                segs.append(seg)
            elapsed += 0.5
        browser.close()

    listing = work / "list.txt"
    listing.write_text("".join(f"file '{s}'\n" for s in segs))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(listing),
            "-c",
            "copy",
            str(args.out),
        ],
        check=True,
    )
    print(f"wrote {args.out} ({len(segs)} clips, about {elapsed / 60:.1f} min)")


if __name__ == "__main__":
    main()
