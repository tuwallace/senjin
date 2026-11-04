"""Assemble the Senjin YouTube video: narrated slides plus the mock call.

Slides are HTML screenshotted with Playwright; narration is the Kokoro-82M neural voice
(`am_michael`, via docs/video/kokoro_say.py) or, with --engine say, a macOS voice. Either differs from Senjin's Piper voice and the candidate's Samantha voice in the mock call. The call
segment plays the mock meeting's own audio. Output goes under data/video/ (git-ignored).

    uv run python docs/video/make_video.py --call data/video/mock-meeting-short.mp4
"""

import argparse
import html
import re
import subprocess
import tempfile
from pathlib import Path

NARRATOR = "am_michael"  # Kokoro voice id; with --engine say, a macOS voice name such as Daniel
RATE = 170  # say only
KOKORO_PYTHON = Path(
    "/Users/twallace/source/ragsandbox/helpdesk-rag/docs/video/narration/.venv/bin/python"
)
CSS = """<meta charset=utf-8><style>
*{box-sizing:border-box;margin:0}
body{width:1920px;height:1080px;background:#14181f;color:#e8eaed;font-family:-apple-system,Helvetica,Arial,sans-serif;
padding:90px 120px;position:relative}
h1{font-size:76px;font-weight:600;margin-bottom:40px}
h2{font-size:34px;color:#8ab4f8;font-weight:500;margin-bottom:14px;letter-spacing:.04em;text-transform:uppercase}
p,li{font-size:46px;line-height:1.4;color:#d5d9de}
ul{list-style:none;padding:0} li{margin:22px 0;padding-left:44px;position:relative}
li:before{content:"";position:absolute;left:0;top:26px;width:16px;height:16px;border-radius:50%;background:#8ab4f8}
.row{display:flex;gap:60px}.col{flex:1}
.box{background:#1f2630;border-radius:18px;padding:34px 40px;margin:12px 0;font-size:40px}
.arrow{align-self:center;font-size:56px;color:#8ab4f8}
.term{background:#0c0f14;border-radius:16px;padding:36px 44px;font-family:Menlo,monospace;font-size:34px;
line-height:1.5;color:#c9d1d9;white-space:pre-wrap;border:2px solid #2a313c}
.term b{color:#7ee787;font-weight:400}.warn{color:#ffa657}.ok{color:#7ee787}
.foot{position:absolute;bottom:40px;left:120px;font-size:26px;color:#8b93a0}
.tick{color:#7ee787;margin-right:14px}
.tag{font-size:30px;color:#8ab4f8;background:#1f2630;border-radius:12px;padding:4px 16px;margin-left:18px}
</style>"""


def term(prompt: str, body: str) -> str:
    return f"<div class=term><b>$ {html.escape(prompt)}</b>\n{html.escape(body)}</div>"


def slides(root: Path) -> dict[str, tuple[str, str]]:
    """key -> (narration, html body)."""
    yaml = (root / "data/candidates/jane/interview.yaml").read_text().splitlines()
    shape = "\n".join(
        line
        for line in yaml
        if line.split(":")[0]
        in ("profile", "seniority", "background_questions", "technical_questions", "follow_ups")
    )
    cov = (root / "data/candidates/jane/questions.coverage.md").read_text().splitlines()
    rows = [r for r in cov if r.startswith("|") and ("asked" in r or "gap" in r)]
    pick = [rows[0], rows[1], rows[2], rows[7], rows[8]]
    table = "\n".join(
        " | ".join(c.strip() for c in r.strip("|").split("|")[:3]).replace("must_have", "must-have")
        for r in pick
    )
    return {
        "hook": (
            "This is a re-creation of a real job interview. The interviewer is an AI running on my "
            "laptop. No cloud models, no API keys. It joined a Google Meet, introduced itself, asked "
            "questions drawn from the candidate's resume and the job posting, listened, asked "
            "follow-ups, and afterwards wrote an assessment. A person still makes the hiring decision. "
            "Here is how it works, and what broke along the way.",
            "",
        ),
        "what": (
            "Senjin has two halves. Before the call, it plans: it reads the resume and the job "
            "description and decides what to ask. During the call, it listens with whisper dot c p p, "
            "thinks with a local Ollama model, and speaks with Piper, through virtual audio devices "
            "into Google Meet. After the call, a bigger local model scores the transcript. All of it "
            "runs on one Mac.",
            "<h2>Senjin</h2><h1>Everything runs on one Mac</h1><div class=row>"
            "<div class=box>Resume +<br>job description</div><div class=arrow>→</div>"
            "<div class=box>Plan and<br>questions</div><div class=arrow>→</div>"
            "<div class=box>Meet call<br>listen · think · speak</div><div class=arrow>→</div>"
            "<div class=box>Transcript</div><div class=arrow>→</div><div class=box>Report</div></div>"
            "<div style='margin-top:50px'><ul><li>Interview: qwen3 8B in Ollama</li>"
            "<li>Ears: whisper.cpp · Voice: Piper · Audio: BlackHole</li>"
            "<li>Scoring: a 32B judge model, also local</li></ul></div>",
        ),
        "plan": (
            "The plan comes first. One command reads the job description, picks an interview shape, "
            "here a senior full stack profile with three background questions and five technical "
            "ones, and writes the questions. Half come from a shared question bank, half are written "
            "for this job. Then it prints a coverage report: which of the job's requirements the "
            "questions actually reach, and which they don't. For this posting, five of thirteen "
            "must-haves are covered by the default five questions. That gap is visible before anyone "
            "is interviewed, and you can edit the questions file by hand. The candidate's name and "
            "contact details are removed before any model sees the resume.",
            "<h2>Before the call</h2><div class=row><div class=col>"
            + term("cat interview.yaml", shape)
            + "</div><div class=col>"
            + term("head questions.coverage.md", "Requirement | Kind | Status\n" + table)
            + "</div></div>",
        ),
        "design": (
            "The key design choice: the code runs the interview, not the model. Code picks the next "
            "question and decides when to follow up or move on. The model only phrases it. That keeps "
            "every candidate on the same plan, keeps the questions from being invented, and lets me "
            "test the flow with fakes instead of real models. The interviewer is also told never to "
            "praise or criticize, and a filter removes praise if the model slips.",
            "<h2>Design</h2><h1>Code asks. The model only phrases.</h1><div class=row>"
            "<div class=col><div class=box><b>Code decides</b><br>which question<br>when to follow up"
            "<br>when to move on<br>when to stop</div></div>"
            "<div class=col><div class=box><b>The model does</b><br>says it naturally<br>"
            "acknowledges, never praises<br>answers from the job description</div></div></div>",
        ),
        "report": (
            "After the call, the judge writes strengths, weaknesses and a score for each question. "
            "Every finding has to quote the candidate, and the code checks that the quote is really in "
            "the transcript. A finding it cannot prove is dropped. Then a fixed hiring bar applies. In "
            "the real call, the judge said Hire, the bar said technical six is below seven, and the "
            "report says so in plain words: No Hire, low confidence, and why it changed. It is "
            "advisory. Someone has to read the transcript.",
            "<h2>After the call</h2>"
            + term(
                "less data/reports/meet-20261103-132204.md",
                "Recommendation: No Hire (confidence: low)\nTechnical: 6/10 · Communication: 6/10\n"
                "Hiring bar: Hire needs technical 7+. The judge said Hire;\nchanged to No Hire because "
                "technical 6 is below 7.\n\nWeakness: explanation of automating NIST 800-53 controls\n"
                'was vague.\n  [11] "I would run a process that collected all the\n  vulnerabilities '
                'and looked at the different areas..."\n\nAdvisory only: a human makes the hiring decision.',
            ),
        ),
        "checks": (
            "I checked three things. Same transcript in, same report out: with temperature zero there "
            "was no variation. Then a fairness check: the same answers with different names and cues "
            "attached, with and without the redaction step. And speed: in one live call it took about "
            "two seconds from the candidate stopping to the first sound. The tests do not call any "
            "model. They use fakes, so they run anywhere.",
            "<h2>Checking it</h2><h1>Honest checks, small samples</h1><ul>"
            "<li>Same inputs, same report: no variation at temperature 0</li>"
            "<li>302 offline tests, none call a model</li>"
            "<li>Fairness check: names and cues swapped, with and without redaction</li>"
            "<li>One live call: about 2.1 s to first sound over 19 turns</li></ul>",
        ),
        "broke": (
            "Plenty broke. A follow-up was lost when the candidate cut in before the model spoke. "
            "Praise leaked through when the model used a curly apostrophe. A quote check failed "
            "because the judge wrote a period where the candidate had a comma. The coverage report "
            "called questions a gap when they were only cut for time. And my own prompt change made the "
            "greeting drop the notice, which is why it is fixed text now. I also tried a small decision "
            "model to detect when the candidate is finished. It was more accurate on my small test set, "
            "but slower, so it is off by default.",
            "<h2>What went wrong</h2><ul>"
            "<li><span class=tick>✓</span>Follow-up lost when the candidate cut in</li>"
            "<li><span class=tick>✓</span>Praise leaked through a curly apostrophe</li>"
            "<li><span class=tick>✓</span>A quote check failed on a period versus a comma</li>"
            "<li><span class=tick>✓</span>Coverage said “gap” when a question was only cut for time</li>"
            "<li><span class=tick>✓</span>My prompt change dropped the greeting's notice</li>"
            "<li>Decision model: more accurate, slower, off by default</li></ul>",
        ),
        "next": (
            "Four ideas are written down but not on the roadmap. First, small decision models. Ollama "
            "serves tiny models that answer questions like, has the candidate finished, with a "
            "probability. I wired one in as a second opinion on when the candidate's questions are "
            "over. On seventy hand-labelled turns it beat my rules on that decision, forty-three right "
            "to forty, but it added about six hundred milliseconds a call, so it is off by default. "
            "Second, scheduling. After the review, Senjin could draft a Google Meet invite, show it to "
            "a person, send nothing until they say yes, then start the call at the right time and wait "
            "for the candidate. The invite would say an AI conducts and transcribes the interview. I "
            "have not tested any of it, and whether a candidate has to knock to enter is still an open "
            "question. Third, a clearer voice. I compared eight voices by ear, and Senjin now speaks "
            "twenty percent slower. Speech recognition mishearing names has been a bigger problem than "
            "the candidate hearing Senjin. Fourth, report upload to Drive or Slack, only when a person "
            "asks. It is not built, and sharing the file by hand needs no code.",
            "<h2>What could come next</h2><h1>Written down, not on the roadmap</h1><ul>"
            "<li><b>Decision models</b> <span class=tag>built · opt-in · off</span></li>"
            "<li><b>Scheduling and hosting the Meet</b> <span class=tag>proposed · untested</span></li>"
            "<li><b>A clearer voice</b> <span class=tag>slower Piper voice built</span></li>"
            "<li><b>Report upload</b> <span class=tag>proposed</span></li></ul>"
            "<div class=foot>Not built unless marked · not legal advice · see docs/ENHANCEMENTS.md</div>",
        ),
        "end": (
            "Limits. Speech recognition mishears names and terms. The voice is still obviously "
            "synthetic. Rules about automated interviews differ by place, so check yours. Three things "
            "I would pass on. One: let code run the process and keep the model to language. Two: make "
            "every claim in a report point to evidence you can check. Three: read the actual "
            "transcripts, because the averages never showed me any of these bugs. The code and the docs "
            "are on GitHub.",
            "<h2>Takeaways</h2><ul><li>Code runs the process; the model handles language</li>"
            "<li>Every claim in a report points to checkable evidence</li>"
            "<li>Read the transcripts: averages hid every bug</li></ul>"
            "<div class=foot>Advisory only · local only · not legal advice · github.com/tuwallace/senjin</div>",
        ),
    }


def run(*cmd: str) -> None:
    subprocess.run(list(cmd), check=True)


def duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(out.stdout)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--call", type=Path, required=True, help="clip whose picture opens the video")
    ap.add_argument("--out", type=Path, default=Path("data/video/senjin-youtube.mp4"))
    ap.add_argument("--engine", choices=["kokoro", "say"], default="kokoro")
    ap.add_argument("--voice", default=NARRATOR)
    ap.add_argument("--kokoro-python", type=Path, default=KOKORO_PYTHON)
    args = ap.parse_args()
    root = Path.cwd()
    work = Path(tempfile.mkdtemp(prefix="senjin-video-"))

    from call_animation import CALL_SCRIPT
    from call_animation import page as call_page
    from playwright.sync_api import sync_playwright

    def narrate(text: str, name: str, pad: float) -> Path:
        """Speak `text`, trim the voice's own trailing silence, then pad `pad` seconds."""
        aiff, wav = work / f"{name}.aiff", work / f"{name}.wav"
        if args.engine == "say":
            run("say", "-v", args.voice, "-r", str(RATE), "-o", str(aiff), text)
        else:
            subprocess.run(
                [
                    str(args.kokoro_python),
                    str(Path(__file__).with_name("kokoro_say.py")),
                    args.voice,
                    str(aiff),
                ],
                input=text,
                text=True,
                check=True,
                stderr=subprocess.DEVNULL,
            )
        run(
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(aiff),
            "-ar",
            "44100",
            "-ac",
            "2",
            "-af",
            "areverse,silenceremove=start_periods=1:start_threshold=-50dB,areverse,"
            f"apad=pad_dur={pad}",
            str(wav),
        )
        return wav

    data = slides(root)
    order = ["hook", "what", "plan", "call", "design", "report", "checks", "broke", "next", "end"]
    parts: list[Path] = []
    srt: list[tuple[float, float, str]] = []
    t = 0.0

    def captions(narration: str, start: float, dur: float) -> None:
        sentences = re.split(r"(?<=[.?!])\s+", narration)
        total = sum(len(x) for x in sentences)
        cur = start
        for x in sentences:
            share = (dur - 0.5) * len(x) / total
            srt.append((cur, cur + share, x))
            cur += share

    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome")
        page = browser.new_page(viewport={"width": 1920, "height": 1080})
        for key in order:
            seg = work / f"{key}.mp4"
            if key == "call":
                # one narrated sentence at a time, so every highlight lands on its phrase
                wavs, starts, cur = [], [], 0.0
                for i, (text, _, _) in enumerate(CALL_SCRIPT):
                    last = i == len(CALL_SCRIPT) - 1
                    w = narrate(text, f"call{i}", 0.5 if last else 0.3)
                    wavs.append(w)
                    starts.append(cur)
                    cur += duration(w)
                    srt.append((t + starts[-1], t + cur - (0.5 if last else 0.3), text))
                listing = work / "call-audio.txt"
                listing.write_text("".join(f"file '{w}'\n" for w in wavs))
                audio = work / "call-audio.wav"
                run(
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
                    str(audio),
                )
                dur, fps = duration(audio), 12
                page.set_content(call_page(CSS))
                frames = work / "frames"
                frames.mkdir()
                for n in range(int(dur * fps)):
                    tt = n / fps
                    i = max(k for k, st in enumerate(starts) if st <= tt)
                    _, stage, line = CALL_SCRIPT[i]
                    page.evaluate(f"frame({stage},{tt - starts[i]:.3f},{line})")
                    page.screenshot(path=str(frames / f"{n:05d}.jpg"), type="jpeg", quality=92)
                run(
                    "ffmpeg",
                    "-y",
                    "-loglevel",
                    "error",
                    "-framerate",
                    str(fps),
                    "-i",
                    str(frames / "%05d.jpg"),
                    "-i",
                    str(audio),
                    "-r",
                    "30",
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-c:a",
                    "aac",
                    "-t",
                    f"{dur:.3f}",
                    str(seg),
                )
                t += dur
                parts.append(seg)
                continue
            narration, body = data[key]
            wav = narrate(narration, key, 0.5)
            dur = duration(wav)
            if key == "hook":
                run(
                    "ffmpeg",
                    "-y",
                    "-loglevel",
                    "error",
                    "-i",
                    str(args.call),
                    "-i",
                    str(wav),
                    "-map",
                    "0:v",
                    "-map",
                    "1:a",
                    "-t",
                    f"{dur:.3f}",
                    "-r",
                    "30",
                    "-vf",
                    "scale=1920:1080,format=yuv420p",
                    "-c:v",
                    "libx264",
                    "-c:a",
                    "aac",
                    str(seg),
                )
            else:
                page.set_content(CSS + body)
                png = work / f"{key}.png"
                page.screenshot(path=str(png))
                run(
                    "ffmpeg",
                    "-y",
                    "-loglevel",
                    "error",
                    "-loop",
                    "1",
                    "-framerate",
                    "30",
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
                    "-t",
                    f"{dur:.3f}",
                    str(seg),
                )
            captions(narration, t, dur)
            t += dur
            parts.append(seg)
        # the thumbnail
        page.set_content(
            CSS.replace("1920px", "1280px").replace("1080px", "720px")
            + "<body style='padding:60px'><h2>Local AI interviewer</h2>"
            "<h1 style='font-size:92px;line-height:1.1'>I built an AI interviewer that runs "
            "entirely on my Mac</h1><p style='font-size:40px;color:#8ab4f8'>"
            "Google Meet · Ollama · whisper.cpp · Piper</p></body>"
        )
        page.set_viewport_size({"width": 1280, "height": 720})
        page.screenshot(path=str(args.out.with_suffix(".thumbnail.png")))
        browser.close()

    def stamp(x: float) -> str:
        return f"{int(x // 3600):02d}:{int(x % 3600 // 60):02d}:{int(x % 60):02d},{int(x % 1 * 1000):03d}"

    args.out.with_suffix(".srt").write_text(
        "".join(f"{i}\n{stamp(a)} --> {stamp(b)}\n{x}\n\n" for i, (a, b, x) in enumerate(srt, 1))
    )
    # The concat filter (not the demuxer) joins the segments, so the audio of each lines up with its
    # own picture; the demuxer left silent gaps at the joins.
    inputs = [x for seg in parts for x in ("-i", str(seg))]
    graph = (
        "".join(f"[{i}:v][{i}:a]" for i in range(len(parts)))
        + f"concat=n={len(parts)}:v=1:a=1[v][a]"
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    run(
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        *inputs,
        "-filter_complex",
        graph,
        "-map",
        "[v]",
        "-map",
        "[a]",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        str(args.out),
    )
    print(f"wrote {args.out} ({duration(args.out) / 60:.1f} min)")


if __name__ == "__main__":
    main()
